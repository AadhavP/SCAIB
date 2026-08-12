"""Run an agent's code inside a container, where isolation is real.

This is the tier that can honestly claim what the local tier cannot: a denied
network is a namespace with no interfaces, a memory ceiling is a cgroup bounding
resident memory rather than address space, and the filesystem the execution sees
is a mount set instead of the whole host.

One long-lived container is started per workspace and each request is an ``exec``
into it. The alternative -- a container per command -- would restart the Python
interpreter and lose every in-memory intermediate between steps, which for a
single-cell workflow means re-reading and re-normalising the matrix on every
step.

The Docker CLI is driven over ``subprocess`` rather than through the SDK. It
avoids a dependency whose type stubs are incomplete, and the argv is a pure
function of the request, which makes the part most likely to be wrong -- the
flags -- testable without a container runtime present.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from agent_evals.core.exceptions import SandboxExecutionError
from agent_evals.environment.execution.backend import (
    CommandOutcome,
    CommandRequest,
    Language,
    interpreter_argv,
)
from agent_evals.environment.execution.fingerprint import (
    WorkspaceFingerprint,
    fingerprint_workspace,
)
from agent_evals.environment.execution.isolation import (
    IsolationControl,
    IsolationControlReport,
    IsolationOutcome,
    IsolationReport,
    IsolationRequest,
    environment_report,
    filesystem_report,
    network_report,
)
from agent_evals.environment.models import ExecutionStatus, ResourceUsage

BACKEND_NAME = "container"

#: Mount point of the workspace inside the container.
CONTAINER_WORKSPACE = "/workspace"

#: Exit code Docker reports for a container killed by its memory cgroup.
OOM_EXIT_CODE = 137

#: Grace period for lifecycle commands, which should be near-instant.
_LIFECYCLE_TIMEOUT_SECONDS = 60.0

#: Interpreter paths inside the image, which need not match the host's.
_CONTAINER_PYTHON = "python"
_CONTAINER_SHELL = "bash"


def docker_available(executable: str = "docker") -> bool:
    """Whether a Docker CLI is on PATH."""
    return shutil.which(executable) is not None


def build_run_argv(
    *,
    image: str,
    workspace: Path,
    isolation: IsolationRequest,
    input_dir: Path | None = None,
    executable: str = "docker",
    cpu_limit: float | None = None,
) -> tuple[str, ...]:
    """Build the argv that starts the idle container.

    ``--memory-swap`` is set equal to ``--memory`` on purpose. Without it Docker
    grants swap equal to the memory limit, so a run nominally capped at 8 GB may
    use 16 GB and the ceiling recorded in the run record is not the one that
    applied.
    """
    argv: list[str] = [executable, "run", "--detach", "--rm"]
    if not isolation.network_access:
        argv.append("--network=none")
    if isolation.max_memory_mb is not None:
        argv.extend(
            [
                f"--memory={isolation.max_memory_mb}m",
                f"--memory-swap={isolation.max_memory_mb}m",
            ]
        )
    if isolation.max_processes is not None:
        argv.append(f"--pids-limit={isolation.max_processes}")
    if cpu_limit is not None:
        argv.append(f"--cpus={cpu_limit:g}")
    argv.extend(["--volume", f"{workspace.resolve()}:{CONTAINER_WORKSPACE}"])
    if input_dir is not None:
        # Read-only so the agent cannot rewrite the dataset it is scored on.
        argv.extend(
            [
                "--volume",
                f"{input_dir.resolve()}:{CONTAINER_WORKSPACE}/inputs:ro",
            ]
        )
    argv.extend(["--workdir", CONTAINER_WORKSPACE, "--entrypoint", "sleep"])
    argv.extend([image, "infinity"])
    return tuple(argv)


def build_exec_argv(
    *,
    container_id: str,
    language: Language,
    env: dict[str, str],
    executable: str = "docker",
) -> tuple[str, ...]:
    """Build the argv that runs one request inside the started container."""
    argv: list[str] = [executable, "exec", "--interactive"]
    for name, value in sorted(env.items()):
        argv.extend(["--env", f"{name}={value}"])
    argv.extend(["--workdir", CONTAINER_WORKSPACE, container_id])
    argv.extend(
        interpreter_argv(
            language,
            python_executable=_CONTAINER_PYTHON,
            shell_executable=_CONTAINER_SHELL,
        )
    )
    return tuple(argv)


class ContainerBackend:
    """Execute agent-authored code inside a long-lived container."""

    def __init__(
        self,
        root: Path,
        *,
        image: str,
        isolation: IsolationRequest | None = None,
        input_dir: Path | None = None,
        executable: str = "docker",
        cpu_limit: float | None = None,
        extra_environment: dict[str, str] | None = None,
    ) -> None:
        self.root = root
        self.name = BACKEND_NAME
        self.image = image
        self.isolation_request = isolation or IsolationRequest()
        self.input_dir = input_dir
        self.executable = executable
        self.cpu_limit = cpu_limit
        self._extra_environment = dict(extra_environment or {})
        self._container_id: str | None = None

    @property
    def container_id(self) -> str | None:
        """Identifier of the running container, if one has been started."""
        return self._container_id

    async def start(self) -> IsolationReport:
        """Start the idle container and report what it enforces."""
        if not docker_available(self.executable):
            raise SandboxExecutionError(
                f"container backend requires '{self.executable}' on PATH; "
                "install a container runtime or select the local backend"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        argv = build_run_argv(
            image=self.image,
            workspace=self.root,
            isolation=self.isolation_request,
            input_dir=self.input_dir,
            executable=self.executable,
            cpu_limit=self.cpu_limit,
        )
        code, stdout, stderr = await self._lifecycle(argv)
        if code != 0 or not stdout.strip():
            raise SandboxExecutionError(
                f"could not start container from image '{self.image}' "
                f"(exit {code}): {stderr.strip() or stdout.strip() or 'no output'}"
            )
        self._container_id = stdout.strip().splitlines()[-1]
        return self.isolation_report()

    def isolation_report(self) -> IsolationReport:
        """Report what this backend guarantees, per control."""
        controls = [
            network_report(
                network_access=self.isolation_request.network_access,
                enforceable=True,
                mechanism="docker --network=none",
            ),
            filesystem_report(
                enforceable=True,
                mechanism="docker bind mounts",
            ),
            environment_report(
                allowlisted=tuple(self._extra_environment),
                mechanism="container image environment",
            ),
            self._memory_control(),
        ]
        if self.isolation_request.max_processes is not None:
            controls.append(
                IsolationControlReport(
                    control=IsolationControl.PROCESS_COUNT,
                    outcome=IsolationOutcome.ENFORCED,
                    mechanism="docker --pids-limit",
                    requested=str(self.isolation_request.max_processes),
                )
            )
        return IsolationReport(
            backend=self.name,
            platform=sys.platform,
            controls=controls,
        )

    def _memory_control(self) -> IsolationControlReport:
        """Report the resident-memory ceiling, which a cgroup can truly impose."""
        if self.isolation_request.max_memory_mb is None:
            return IsolationControlReport(
                control=IsolationControl.RESIDENT_MEMORY,
                outcome=IsolationOutcome.NOT_REQUESTED,
            )
        return IsolationControlReport(
            control=IsolationControl.RESIDENT_MEMORY,
            outcome=IsolationOutcome.ENFORCED,
            mechanism="docker --memory with --memory-swap pinned to it",
            requested=f"{self.isolation_request.max_memory_mb} MB",
        )

    async def fingerprint(self) -> WorkspaceFingerprint:
        """Fingerprint the host side of the workspace bind mount."""
        return await asyncio.to_thread(fingerprint_workspace, self.root)

    async def close(self) -> None:
        """Remove the container. Idempotent, and never raises on cleanup."""
        container = self._container_id
        self._container_id = None
        if container is None:
            return
        with contextlib.suppress(OSError, ValueError):
            await self._lifecycle((self.executable, "rm", "--force", container))

    async def run(self, request: CommandRequest) -> CommandOutcome:
        """Execute one request inside the container."""
        container = self._container_id
        if container is None:
            raise SandboxExecutionError(
                "ContainerBackend.run called before start(); no container exists"
            )
        env = {**self._extra_environment, **request.env}
        argv = build_exec_argv(
            container_id=container,
            language=request.language,
            env=env,
            executable=self.executable,
        )
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, ValueError) as error:
            return CommandOutcome(
                status=ExecutionStatus.ERROR,
                isolation=self.isolation_report(),
                resource_usage=ResourceUsage(
                    wall_time_seconds=max(time.monotonic() - started, 0.0)
                ),
                error=f"could not exec into container: {type(error).__name__}: {error}",
            )
        return await self._supervise(request, process, started)

    async def _supervise(
        self,
        request: CommandRequest,
        process: asyncio.subprocess.Process,
        started: float,
    ) -> CommandOutcome:
        """Capture output under the wall-clock ceiling and classify the result."""
        payload = request.command.encode("utf-8")
        timed_out = False
        exit_code: int | None = None
        raw_out = b""
        raw_err = b""
        try:
            raw_out, raw_err = await asyncio.wait_for(
                process.communicate(input=payload),
                timeout=request.timeout_seconds,
            )
            exit_code = process.returncode
        except TimeoutError:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5.0)
        limit = request.max_output_bytes
        stdout, stdout_truncated = _clip(raw_out, limit)
        stderr, stderr_truncated = _clip(raw_err, limit)
        status, error = self._classify(
            exit_code=exit_code,
            timed_out=timed_out,
            timeout_seconds=request.timeout_seconds,
        )
        return CommandOutcome(
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            resource_usage=ResourceUsage(
                wall_time_seconds=max(time.monotonic() - started, 0.0)
            ),
            isolation=self.isolation_report(),
            error=error,
        )

    def _classify(
        self,
        *,
        exit_code: int | None,
        timed_out: bool,
        timeout_seconds: float,
    ) -> tuple[ExecutionStatus, str | None]:
        """Map the container's exit code to an execution status."""
        if timed_out:
            return (
                ExecutionStatus.TIMEOUT,
                f"execution exceeded its {timeout_seconds:g}s limit and was killed",
            )
        if exit_code is None:  # pragma: no cover - defensive
            return ExecutionStatus.ERROR, "execution ended without an exit code"
        if exit_code == 0:
            return ExecutionStatus.SUCCESS, None
        if exit_code == OOM_EXIT_CODE and self.isolation_request.max_memory_mb:
            return (
                ExecutionStatus.OOM,
                "container was killed by its memory cgroup at "
                f"{self.isolation_request.max_memory_mb} MB",
            )
        return ExecutionStatus.ERROR, f"execution exited with code {exit_code}"

    async def _lifecycle(
        self,
        argv: tuple[str, ...],
    ) -> tuple[int, str, str]:
        """Run a short Docker lifecycle command and return its output."""
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            raw_out, raw_err = await asyncio.wait_for(
                process.communicate(),
                timeout=_LIFECYCLE_TIMEOUT_SECONDS,
            )
        except TimeoutError:  # pragma: no cover - rare
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            return -1, "", f"'{' '.join(argv[:2])}' did not return in time"
        return (
            process.returncode or 0,
            raw_out.decode("utf-8", errors="replace"),
            raw_err.decode("utf-8", errors="replace"),
        )


def _clip(raw: bytes, limit: int) -> tuple[str, bool]:
    """Decode output, reporting whether the size ceiling dropped any of it."""
    if len(raw) <= limit:
        return raw.decode("utf-8", errors="replace"), False
    return raw[:limit].decode("utf-8", errors="replace"), True


__all__ = [
    "BACKEND_NAME",
    "CONTAINER_WORKSPACE",
    "OOM_EXIT_CODE",
    "ContainerBackend",
    "build_exec_argv",
    "build_run_argv",
    "docker_available",
]
