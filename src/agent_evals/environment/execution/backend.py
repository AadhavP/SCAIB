"""The contract for running an agent's own code inside a workspace.

SCAIB does not execute the agent's science.  It executes whatever source the
agent submitted, captures what came out, and then observes what changed on
disk.  That inversion is the whole point of the layer: a benchmark that runs the
pipeline itself measures how well an agent fills in someone else's pipeline.

One rule governs every implementation of this Protocol: **a failing execution is
a result, not an exception.**  An agent whose script raises has made a
scientific mistake, and that mistake is data.  If a backend let it propagate,
``ScientificEnvironment.step`` would catch it and record ``executor error``,
which attributes the agent's bug to the harness and destroys the distinction
the benchmark exists to measure.  Exceptions are therefore reserved for
lifecycle faults that are genuinely SCAIB's problem -- a backend used before it
was started, a container runtime that is not installed.

Source is fed to the interpreter on standard input rather than written to a
script file.  A file would appear in the workspace as a change SCAIB itself
caused, contaminating exactly the before/after comparison that provenance is
derived from, and it would need cleanup that a timeout could skip.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import Field

from agent_evals.environment.execution.fingerprint import WorkspaceFingerprint
from agent_evals.environment.execution.isolation import IsolationReport
from agent_evals.environment.models import (
    ExecutionStatus,
    ResourceUsage,
    RuntimeModel,
)

#: Captured stdout/stderr ceiling per stream. Output above this is dropped but
#: still drained, so a chatty execution cannot exhaust harness memory and cannot
#: deadlock on a full pipe either. Truncation is always recorded.
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024

#: Per-command wall-clock ceiling when the benchmark declares none.
DEFAULT_TIMEOUT_SECONDS = 300.0

#: Variables that survive the environment scrub on POSIX hosts.
POSIX_ENVIRONMENT_ALLOWLIST = (
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "SHELL",
    "TERM",
    "TMPDIR",
    "TZ",
    "USER",
)

#: Variables that survive the environment scrub on Windows hosts. Python needs
#: ``SYSTEMROOT`` to initialise sockets and TLS, so a stricter list breaks the
#: interpreter rather than the agent.
WINDOWS_ENVIRONMENT_ALLOWLIST = (
    "COMSPEC",
    "NUMBER_OF_PROCESSORS",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
)


class Language(StrEnum):
    """Languages an execution may be written in.

    R is deliberately absent rather than stubbed: it is neither installed nor in
    any image here, and a registered-but-broken language would fail as a harness
    error and be scored against the agent. Adding it later is one entry in
    :func:`interpreter_argv`.
    """

    PYTHON = "python"
    BASH = "bash"


def interpreter_argv(
    language: Language,
    *,
    python_executable: str,
    shell_executable: str = "bash",
) -> tuple[str, ...]:
    """Return the argv that reads ``language`` source from standard input."""
    if language is Language.PYTHON:
        # -u keeps output ordered when a run is truncated or times out, which is
        # when the captured output matters most.
        return (python_executable, "-u", "-")
    return (shell_executable, "-s")


class CommandRequest(RuntimeModel):
    """One execution to perform in the workspace."""

    command: str = Field(min_length=1)
    language: Language = Language.PYTHON
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0)
    #: Extra variables layered over the allowlisted base environment. Never a
    #: replacement for it: the base is an allowlist so that evaluator
    #: credentials cannot reach the execution by default.
    env: dict[str, str] = Field(default_factory=dict)
    max_output_bytes: int = Field(default=DEFAULT_MAX_OUTPUT_BYTES, gt=0)
    #: Human-facing tag for traces; never affects execution.
    label: str | None = None


class CommandOutcome(RuntimeModel):
    """Everything observed about one execution."""

    status: ExecutionStatus
    #: ``None`` when the process never started or was killed before reporting.
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    isolation: IsolationReport
    #: Why the execution did not succeed, in harness terms. Distinct from
    #: ``stderr``, which is the agent's own diagnostic output.
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether the execution ran to completion without error."""
        return self.status is ExecutionStatus.SUCCESS

    @property
    def truncated(self) -> bool:
        """Whether either captured stream lost output to the size ceiling."""
        return self.stdout_truncated or self.stderr_truncated


@runtime_checkable
class WorkspaceBackend(Protocol):
    """Port for a place an agent's code can run and be observed.

    Implemented by a local subprocess and a container today; a remote or
    cluster-backed implementation satisfies the same four calls. Consumers see
    only this Protocol, so no benchmark, executor, or scoring code contains a
    branch on which backend is in use.
    """

    #: Workspace root. Commands run with this as their working directory and it
    #: is the tree ``fingerprint`` covers.
    root: Path
    #: Stable identifier recorded in the isolation report.
    name: str

    async def start(self) -> IsolationReport:
        """Prepare the backend and report what it can actually enforce."""

    async def run(self, request: CommandRequest) -> CommandOutcome:
        """Execute one request, returning failures as outcomes not exceptions."""

    async def fingerprint(self) -> WorkspaceFingerprint:
        """Capture current workspace state for before/after comparison."""

    async def close(self) -> None:
        """Release backend resources. Must be safe to call more than once."""


def environment_allowlist(platform_name: str) -> tuple[str, ...]:
    """Return the variables that survive the scrub on the named platform."""
    if platform_name == "win32":
        return WINDOWS_ENVIRONMENT_ALLOWLIST
    return POSIX_ENVIRONMENT_ALLOWLIST


__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "POSIX_ENVIRONMENT_ALLOWLIST",
    "WINDOWS_ENVIRONMENT_ALLOWLIST",
    "CommandOutcome",
    "CommandRequest",
    "Language",
    "WorkspaceBackend",
    "environment_allowlist",
    "interpreter_argv",
]
