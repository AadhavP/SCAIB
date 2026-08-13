"""Run an agent's code as a local subprocess in a pinned workspace.

This is the always-available tier. It gives real execution with modest
isolation: working directory pinned to the workspace, environment reduced to an
allowlist, POSIX resource limits where the platform has them. It does not give
network denial or filesystem confinement, and it says so in its
:class:`IsolationReport` rather than leaving a reader to assume otherwise.
Anything needing a real boundary uses the container tier.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from agent_evals.core.exceptions import SandboxExecutionError
from agent_evals.environment.execution.backend import (
    CommandOutcome,
    CommandRequest,
    environment_allowlist,
    interpreter_argv,
)
from agent_evals.environment.execution.fingerprint import (
    WorkspaceFingerprint,
    fingerprint_workspace,
)
from agent_evals.environment.execution.isolation import (
    IsolationControlReport,
    IsolationReport,
    IsolationRequest,
    build_limit_setter,
    describe_process_limits,
    environment_report,
    filesystem_report,
    network_report,
)
from agent_evals.environment.models import ExecutionStatus, ResourceUsage

BACKEND_NAME = "local"

#: How often the peak-memory sampler reads the child's high-water mark. Short
#: enough to catch a spike in a multi-second scientific step, long enough that
#: the sampling itself is not the cost.
_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.05

_STREAM_CHUNK_BYTES = 64 * 1024


class _CappedBuffer:
    """Drain a stream while storing at most ``limit`` bytes.

    Draining past the limit matters as much as the limit: a child that fills its
    stdout pipe blocks forever, so output has to keep being read even once it has
    stopped being kept. Holding the buffer in an object rather than returning it
    also means a cancelled read -- which is what a timeout is -- leaves the
    partial output behind, and the output of a timed-out execution is the most
    diagnostic thing about it.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.truncated = False
        self._chunks: list[bytes] = []
        self._kept = 0

    def feed(self, chunk: bytes) -> None:
        """Store what still fits and flag the rest as dropped."""
        room = self.limit - self._kept
        if room <= 0:
            self.truncated = True
            return
        if len(chunk) > room:
            self._chunks.append(chunk[:room])
            self._kept += room
            self.truncated = True
            return
        self._chunks.append(chunk)
        self._kept += len(chunk)

    def text(self) -> str:
        """Decode what was kept, tolerating a split multi-byte character."""
        return b"".join(self._chunks).decode("utf-8", errors="replace")


async def _drain(stream: asyncio.StreamReader | None, buffer: _CappedBuffer) -> None:
    """Read a stream to EOF, keeping only what the buffer has room for."""
    if stream is None:  # pragma: no cover - defensive
        return
    while chunk := await stream.read(_STREAM_CHUNK_BYTES):
        buffer.feed(chunk)


async def _feed_stdin(stream: asyncio.StreamWriter | None, payload: bytes) -> None:
    """Write source to the interpreter, tolerating an early exit."""
    if stream is None:  # pragma: no cover - defensive
        return
    try:
        stream.write(payload)
        await stream.drain()
    except (BrokenPipeError, ConnectionResetError):
        # The interpreter exited before reading its source -- a syntax error in
        # the first line does this. Its own stderr is the useful record.
        pass
    finally:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
            stream.close()


def _read_peak_rss_mb(pid: int) -> float | None:
    """Read a running process's high-water resident memory, on Linux only.

    Deliberately narrow. ``getrusage(RUSAGE_CHILDREN)`` would report a number on
    every POSIX host, but it is a maximum over every child ever reaped, so
    attributing it to *this* execution over-reports and would fail runs against
    a memory ceiling for something an earlier step did.
    """
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


async def _sample_peak_memory(pid: int, sink: list[float]) -> None:
    """Poll a child's peak resident memory until the task is cancelled."""
    while True:
        sample = _read_peak_rss_mb(pid)
        if sample is not None:
            sink.append(sample)
        await asyncio.sleep(_MEMORY_SAMPLE_INTERVAL_SECONDS)


def local_isolation_controls(
    request: IsolationRequest,
    *,
    platform_name: str = sys.platform,
) -> tuple[IsolationControlReport, ...]:
    """Report every control this tier touches, enforced or not.

    Shared with the ``env`` CLI rather than written once here and once there,
    because the two must agree exactly and a disagreement fails *silently*: the
    CLI reports what an operator decides to trust before a run, the run record
    reports what actually held, and nothing compares them. The CLI used to
    hand-list ``filesystem_scope`` among the controls this tier imposes while
    this backend reported it ``unenforceable`` on every host, so the pre-run
    summary asserted confinement the run record then denied.

    ``filesystem_scope`` and ``environment`` are always reported, unrequestable
    and unconditional: this tier pins the working directory and reduces the
    environment on every execution, and neither is something a benchmark can
    ask for or decline. Reporting them only when asked would leave the honest
    ``unenforceable`` verdict on the first one absent from the record.
    """
    return (
        network_report(network_access=request.network_access, enforceable=False),
        filesystem_report(enforceable=False),
        environment_report(allowlisted=environment_allowlist(platform_name)),
        *describe_process_limits(request, platform_name=platform_name),
    )


class LocalProcessBackend:
    """Execute agent-authored code in a local subprocess."""

    def __init__(
        self,
        root: Path,
        *,
        isolation: IsolationRequest | None = None,
        python_executable: str | None = None,
        shell_executable: str = "bash",
        extra_environment: dict[str, str] | None = None,
        platform_name: str = sys.platform,
    ) -> None:
        self.root = root
        self.name = BACKEND_NAME
        self.isolation_request = isolation or IsolationRequest()
        self.python_executable = python_executable or sys.executable
        self.shell_executable = shell_executable
        self.platform_name = platform_name
        self._extra_environment = dict(extra_environment or {})
        self._allowlist = environment_allowlist(platform_name)
        self._limit_setter: Callable[[], None] | None = None
        #: Keyed off the real platform rather than ``platform_name``, which only
        #: steers what the isolation report *claims*. Spawning has to follow the
        #: host it is actually spawning on.
        self._new_session = sys.platform != "win32"
        self._started = False

    async def start(self) -> IsolationReport:
        """Create the workspace root and report enforceable controls."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._limit_setter = build_limit_setter(self.isolation_request)
        self._started = True
        return self.isolation_report()

    def isolation_report(self) -> IsolationReport:
        """Report what this backend does and does not guarantee."""
        return IsolationReport(
            backend=self.name,
            platform=self.platform_name,
            controls=list(
                local_isolation_controls(
                    self.isolation_request,
                    platform_name=self.platform_name,
                )
            ),
        )

    async def fingerprint(self) -> WorkspaceFingerprint:
        """Fingerprint the workspace tree."""
        return await asyncio.to_thread(fingerprint_workspace, self.root)

    async def close(self) -> None:
        """Mark the backend unusable. Idempotent; nothing is deleted."""
        self._started = False

    async def run(self, request: CommandRequest) -> CommandOutcome:
        """Execute one request, converting every failure into an outcome."""
        if not self._started:
            raise SandboxExecutionError(
                "LocalProcessBackend.run called before start(); the workspace "
                "root and resource limits are not prepared"
            )
        argv = interpreter_argv(
            request.language,
            python_executable=self.python_executable,
            shell_executable=self.shell_executable,
        )
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.root),
                env=self._environment(request),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Both are POSIX-only and inert on Windows, where the limit
                # setter is always ``None`` and the session flag is ignored.
                # Passing them unconditionally keeps one spawn call rather than
                # a platform branch whose Windows half CI never exercises.
                start_new_session=self._new_session,
                preexec_fn=self._limit_setter,
            )
        except (OSError, ValueError) as error:
            return self._failed_to_start(request, argv, error, started)
        return await self._supervise(request, process, started)

    async def _supervise(
        self,
        request: CommandRequest,
        process: asyncio.subprocess.Process,
        started: float,
    ) -> CommandOutcome:
        """Drive stdin, capture output, and enforce the wall-clock ceiling."""
        stdout = _CappedBuffer(request.max_output_bytes)
        stderr = _CappedBuffer(request.max_output_bytes)
        memory_samples: list[float] = []
        sampler = (
            asyncio.create_task(_sample_peak_memory(process.pid, memory_samples))
            if self.platform_name.startswith("linux")
            else None
        )
        payload = request.command.encode("utf-8")

        async def pump() -> int:
            await asyncio.gather(
                _feed_stdin(process.stdin, payload),
                _drain(process.stdout, stdout),
                _drain(process.stderr, stderr),
            )
            return await process.wait()

        timed_out = False
        exit_code: int | None = None
        try:
            exit_code = await asyncio.wait_for(pump(), timeout=request.timeout_seconds)
        except TimeoutError:
            timed_out = True
            await self._terminate(process)
        finally:
            if sampler is not None:
                sampler.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sampler
        usage = ResourceUsage(
            wall_time_seconds=max(time.monotonic() - started, 0.0),
            peak_memory_mb=max(memory_samples) if memory_samples else None,
        )
        return self._outcome(
            request=request,
            exit_code=exit_code,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            usage=usage,
        )

    def _outcome(
        self,
        *,
        request: CommandRequest,
        exit_code: int | None,
        timed_out: bool,
        stdout: _CappedBuffer,
        stderr: _CappedBuffer,
        usage: ResourceUsage,
    ) -> CommandOutcome:
        """Classify how the execution ended without guessing beyond the evidence."""
        stderr_text = stderr.text()
        status, error = self._classify(
            exit_code=exit_code,
            timed_out=timed_out,
            timeout_seconds=request.timeout_seconds,
            stderr_text=stderr_text,
        )
        return CommandOutcome(
            status=status,
            exit_code=exit_code,
            stdout=stdout.text(),
            stderr=stderr_text,
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
            resource_usage=usage,
            isolation=self.isolation_report(),
            error=error,
        )

    def _classify(
        self,
        *,
        exit_code: int | None,
        timed_out: bool,
        timeout_seconds: float,
        stderr_text: str,
    ) -> tuple[ExecutionStatus, str | None]:
        """Map process evidence to an execution status.

        Every inference is recorded in the returned message, because the
        distinction between "ran out of memory" and "crashed" is scored, and a
        silent guess would be indistinguishable from a measurement.
        """
        if timed_out:
            return (
                ExecutionStatus.TIMEOUT,
                f"execution exceeded its {timeout_seconds:g}s limit and was killed",
            )
        if exit_code is None:  # pragma: no cover - defensive
            return ExecutionStatus.ERROR, "execution ended without an exit code"
        if exit_code == 0:
            return ExecutionStatus.SUCCESS, None
        memory_limited = self.isolation_request.max_memory_mb is not None
        if memory_limited and "MemoryError" in stderr_text:
            return (
                ExecutionStatus.OOM,
                "execution raised MemoryError under an address-space limit of "
                f"{self.isolation_request.max_memory_mb} MB",
            )
        if exit_code < 0:
            signal_number = -exit_code
            if memory_limited and signal_number == 9:
                return (
                    ExecutionStatus.OOM,
                    "execution was killed by SIGKILL while an address-space "
                    f"limit of {self.isolation_request.max_memory_mb} MB was in "
                    "force; inferred from the signal, not measured",
                )
            return (
                ExecutionStatus.TERMINATED,
                f"execution was killed by signal {signal_number}",
            )
        return ExecutionStatus.ERROR, f"execution exited with code {exit_code}"

    def _failed_to_start(
        self,
        request: CommandRequest,
        argv: Sequence[str],
        error: Exception,
        started: float,
    ) -> CommandOutcome:
        """Report a missing interpreter as an outcome, naming the cause.

        A missing ``bash`` is the harness's problem, not the agent's, so the
        message says which executable was not runnable instead of leaving an
        exit code for a reader to interpret.
        """
        interpreter = argv[0] if argv else "?"
        return CommandOutcome(
            status=ExecutionStatus.ERROR,
            stderr="",
            resource_usage=ResourceUsage(
                wall_time_seconds=max(time.monotonic() - started, 0.0)
            ),
            isolation=self.isolation_report(),
            error=(
                f"could not start {request.language.value} interpreter "
                f"'{interpreter}': {type(error).__name__}: {error}"
            ),
        )

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        """Kill a timed-out execution and everything it spawned.

        Killing only the direct child leaves its workers running, which on a
        long benchmark accumulates until the host is unusable. The process group
        covers them on POSIX; Windows needs ``taskkill /T`` because
        ``TerminateProcess`` does not touch descendants.
        """
        if process.returncode is not None:  # pragma: no cover - race
            return
        if sys.platform == "win32":  # pragma: no cover - platform-conditional
            await self._kill_windows_tree(process)
        else:
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(os.getpgid(process.pid), 9)
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=5.0)

    @staticmethod
    async def _kill_windows_tree(
        process: asyncio.subprocess.Process,
    ) -> None:  # pragma: no cover - platform-conditional
        """Best-effort process-tree kill on Windows."""
        with contextlib.suppress(OSError, ValueError):
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(process.pid),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(killer.wait(), timeout=5.0)

    def _environment(self, request: CommandRequest) -> dict[str, str]:
        """Build the execution environment from an allowlist, never a denylist.

        An allowlist is the difference between a leak and a design: a denylist
        has to enumerate every credential variable that exists now and every one
        added later, and missing one hands the agent the evaluator's API keys --
        which it could spend, or use to fetch answers.
        """
        environment = {
            name: os.environ[name] for name in self._allowlist if name in os.environ
        }
        environment.update(self._extra_environment)
        environment.update(request.env)
        return environment

    def resolve_in_workspace(self, candidate: str) -> Path | None:
        """Resolve a workspace-relative path, or ``None`` if it escapes.

        The local tier cannot confine writes, but it can refuse to *account* for
        anything outside the workspace. Without this, ``produces:
        ["../../reference.csv"]`` would let a declared artifact name a file the
        evaluator owns.
        """
        return resolve_within(self.root, candidate)


def resolve_within(root: Path, candidate: str) -> Path | None:
    """Return ``candidate`` resolved under ``root``, or ``None`` if it escapes.

    Resolution happens before the containment check so that ``..`` segments and
    symlinks are both normalised away; checking the literal string would pass a
    path that only escapes once the filesystem resolves it.
    """
    if not candidate or candidate.strip() != candidate:
        return None
    resolved_root = root.resolve()
    try:
        resolved = (resolved_root / candidate).resolve()
    except (OSError, RuntimeError):
        return None
    if resolved == resolved_root:
        return None
    if resolved_root not in resolved.parents:
        return None
    return resolved


__all__ = [
    "BACKEND_NAME",
    "LocalProcessBackend",
    "local_isolation_controls",
    "resolve_within",
]
