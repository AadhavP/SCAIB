"""What the host can actually enforce, reported rather than assumed.

A benchmark declares constraints -- no network, a memory ceiling, a CPU budget.
Whether those are *enforced* depends entirely on the platform and the backend:
POSIX resource limits do not exist on Windows, and a bare subprocess cannot be
denied a network without privileges no benchmark runner should hold.

The failure mode this module exists to prevent is a paper claiming isolation
its runs did not have.  So nothing here silently degrades.  Every control is
reported per run as ``not_requested``, ``enforced``, ``unenforceable``, or
``failed``, that report travels with the execution result, and a reader can ask
which controls were real instead of trusting a configuration file that merely
asked.

The controls are deliberately named after the mechanism rather than the
intent. ``address_space`` is not ``resident_memory``: ``RLIMIT_AS`` bounds
virtual address space, and a BLAS library that reserves per-thread arenas can
trip a 16 GB address-space limit while resident usage is a few gigabytes. Only
a cgroup -- which is what the container tier's ``--memory`` gives -- bounds
resident memory. Calling both "memory" would collapse a distinction that
decides whether a run failed for a scientific reason or an accounting one.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field

from agent_evals.environment.models import RuntimeModel


class IsolationControl(StrEnum):
    """One enforceable property of an execution, named after its mechanism."""

    #: Outbound network reachability.
    NETWORK = "network"
    #: Virtual address space ceiling (``RLIMIT_AS``). Not resident memory.
    ADDRESS_SPACE = "address_space"
    #: Resident memory ceiling, which only a cgroup can impose.
    RESIDENT_MEMORY = "resident_memory"
    #: CPU-seconds ceiling (``RLIMIT_CPU``), independent of wall clock.
    CPU_TIME = "cpu_time"
    #: Ceiling on processes/threads the execution may spawn.
    PROCESS_COUNT = "process_count"
    #: Ceiling on the size of any single file the execution may write.
    FILE_SIZE = "file_size"
    #: Whether writes are confined to the workspace, not merely aimed at it.
    FILESYSTEM_SCOPE = "filesystem_scope"
    #: Whether the execution's environment was reduced to an allowlist.
    ENVIRONMENT = "environment"
    #: Whether Linux capabilities were dropped from the execution process.
    CAPABILITIES = "capabilities"
    #: Whether privilege escalation was disabled by the runtime.
    PRIVILEGE_ESCALATION = "privilege_escalation"
    #: Whether the container root filesystem was mounted read-only.
    ROOT_FILESYSTEM = "root_filesystem"
    #: Whether temporary writes use a controlled temporary filesystem.
    TEMPORARY_FILESYSTEM = "temporary_filesystem"
    #: Whether the process runs as a non-root identity.
    NON_ROOT = "non_root"


class IsolationOutcome(StrEnum):
    """Whether a control was actually imposed on this execution."""

    #: The benchmark did not ask for this control.
    NOT_REQUESTED = "not_requested"
    #: Requested, and a real mechanism was applied.
    ENFORCED = "enforced"
    #: Requested, but this platform or backend has no mechanism for it. The
    #: execution still ran; the guarantee is absent and must not be claimed.
    UNENFORCEABLE = "unenforceable"
    #: Requested, a mechanism exists, and applying it errored.
    FAILED = "failed"


class IsolationControlReport(RuntimeModel):
    """Per-control record of what was asked for and what was imposed."""

    control: IsolationControl
    outcome: IsolationOutcome
    #: The concrete mechanism, e.g. ``RLIMIT_AS`` or ``docker --network=none``.
    mechanism: str | None = None
    #: The requested value, rendered for the record.
    requested: str | None = None
    detail: str | None = None


class IsolationReport(RuntimeModel):
    """Everything one execution's isolation did and did not guarantee."""

    backend: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    controls: list[IsolationControlReport] = Field(default_factory=list)

    def outcome_for(self, control: IsolationControl) -> IsolationOutcome:
        """Return the outcome for one control, defaulting to not-requested."""
        for report in self.controls:
            if report.control is control:
                return report.outcome
        return IsolationOutcome.NOT_REQUESTED

    @property
    def enforced(self) -> frozenset[IsolationControl]:
        """Controls that were genuinely imposed."""
        return frozenset(
            report.control
            for report in self.controls
            if report.outcome is IsolationOutcome.ENFORCED
        )

    @property
    def unenforced(self) -> frozenset[IsolationControl]:
        """Controls the benchmark asked for and did not get."""
        return frozenset(
            report.control
            for report in self.controls
            if report.outcome
            in (IsolationOutcome.UNENFORCEABLE, IsolationOutcome.FAILED)
        )

    @property
    def is_complete(self) -> bool:
        """Whether every requested control was enforced."""
        return not self.unenforced


@dataclass(frozen=True)
class IsolationRequest:
    """Backend-neutral statement of the limits to impose.

    Deliberately free of any benchmark-schema import: the execution layer is a
    port, and a port that knows about ``ConstraintSpecification`` cannot be
    reused by anything that does not speak the benchmark DSL. Translation from
    the DSL happens one layer up, in the executor.
    """

    #: ``False`` means the benchmark asked for the network to be denied.
    network_access: bool = True
    max_memory_mb: int | None = None
    max_cpu_seconds: int | None = None
    max_processes: int | None = None
    max_file_size_mb: int | None = None

    @property
    def requests_any_limit(self) -> bool:
        """Whether any resource ceiling was requested at all."""
        return any(
            value is not None
            for value in (
                self.max_memory_mb,
                self.max_cpu_seconds,
                self.max_processes,
                self.max_file_size_mb,
            )
        )


@dataclass(frozen=True)
class _LimitSpec:
    """One requested POSIX limit, resolved before the child is forked."""

    control: IsolationControl
    mechanism: str
    requested: str
    #: Name of the ``resource.RLIMIT_*`` constant, held as a string so this
    #: module imports and type-checks on a platform that has no ``resource``.
    rlimit_name: str
    value: int


def _requested_limits(request: IsolationRequest) -> tuple[_LimitSpec, ...]:
    """Translate a request into the POSIX limits that would express it."""
    specs: list[_LimitSpec] = []
    if request.max_memory_mb is not None:
        specs.append(
            _LimitSpec(
                control=IsolationControl.ADDRESS_SPACE,
                mechanism="RLIMIT_AS",
                requested=f"{request.max_memory_mb} MB",
                rlimit_name="RLIMIT_AS",
                value=request.max_memory_mb * 1024 * 1024,
            )
        )
    if request.max_cpu_seconds is not None:
        specs.append(
            _LimitSpec(
                control=IsolationControl.CPU_TIME,
                mechanism="RLIMIT_CPU",
                requested=f"{request.max_cpu_seconds} s",
                rlimit_name="RLIMIT_CPU",
                value=request.max_cpu_seconds,
            )
        )
    if request.max_processes is not None:
        specs.append(
            _LimitSpec(
                control=IsolationControl.PROCESS_COUNT,
                mechanism="RLIMIT_NPROC",
                requested=str(request.max_processes),
                rlimit_name="RLIMIT_NPROC",
                value=request.max_processes,
            )
        )
    if request.max_file_size_mb is not None:
        specs.append(
            _LimitSpec(
                control=IsolationControl.FILE_SIZE,
                mechanism="RLIMIT_FSIZE",
                requested=f"{request.max_file_size_mb} MB",
                rlimit_name="RLIMIT_FSIZE",
                value=request.max_file_size_mb * 1024 * 1024,
            )
        )
    return tuple(specs)


def supports_resource_limits(platform_name: str = sys.platform) -> bool:
    """Whether POSIX resource limits exist on the named platform."""
    return platform_name != "win32"


def describe_process_limits(
    request: IsolationRequest,
    *,
    platform_name: str = sys.platform,
) -> tuple[IsolationControlReport, ...]:
    """Report which requested resource limits this platform can impose.

    Kept separate from applying them so the honesty of the report is testable
    on any host: the Windows-degradation path is the one most likely to be
    wrong and the one least likely to be exercised by Linux CI.
    """
    specs = _requested_limits(request)
    if supports_resource_limits(platform_name):
        return tuple(
            IsolationControlReport(
                control=spec.control,
                outcome=IsolationOutcome.ENFORCED,
                mechanism=spec.mechanism,
                requested=spec.requested,
            )
            for spec in specs
        )
    return tuple(
        IsolationControlReport(
            control=spec.control,
            outcome=IsolationOutcome.UNENFORCEABLE,
            requested=spec.requested,
            detail=(
                f"POSIX resource limits are unavailable on {platform_name}; "
                f"{spec.mechanism} was not applied. Use the container backend "
                "for an enforced limit."
            ),
        )
        for spec in specs
    )


def build_limit_setter(
    request: IsolationRequest,
) -> Callable[[], None] | None:
    """Return a child-side callable installing the requested POSIX limits.

    Returns ``None`` when there is nothing to install or the platform has no
    mechanism, in which case the caller must not pass a ``preexec_fn``.

    Both the soft and hard limit are set, so the execution cannot raise its own
    ceiling back afterwards. Each requested value is clamped to the parent's
    existing hard limit, because a child may lower a hard limit but never raise
    one -- attempting to would make the whole execution fail to start, turning a
    tighter-than-expected host policy into an error blamed on the agent.

    ``resource`` is imported here rather than at module scope because it does not
    exist on Windows. Guarding the import at module scope would leave the name
    undefined for a type-checker running on Windows while the code that uses it
    stayed visible; behind this early return the whole block is platform-dead, so
    both the import and its uses are checked exactly where they are real.
    """
    if sys.platform == "win32":  # pragma: no cover - platform-conditional
        return None
    import resource

    specs = _requested_limits(request)
    if not specs:
        return None
    resolved: list[tuple[int, tuple[int, int]]] = []
    for spec in specs:
        rlimit = getattr(resource, spec.rlimit_name, None)
        if rlimit is None:  # pragma: no cover - platform-conditional
            continue
        _, current_hard = resource.getrlimit(rlimit)
        ceiling = spec.value
        if current_hard != resource.RLIM_INFINITY:
            ceiling = min(ceiling, current_hard)
        resolved.append((rlimit, (ceiling, ceiling)))
    if not resolved:  # pragma: no cover - platform-conditional
        return None

    def apply_limits() -> None:
        """Run in the forked child before exec; kept to bare syscalls."""
        for rlimit, bounds in resolved:
            resource.setrlimit(rlimit, bounds)

    return apply_limits


def environment_report(
    *,
    allowlisted: Sequence[str],
    mechanism: str = "environment allowlist",
) -> IsolationControlReport:
    """Report the environment scrub, which every backend can always enforce."""
    return IsolationControlReport(
        control=IsolationControl.ENVIRONMENT,
        outcome=IsolationOutcome.ENFORCED,
        mechanism=mechanism,
        requested=f"{len(allowlisted)} variable(s) passed through",
        detail=(
            "the execution cannot read the evaluator's environment, so provider "
            "API keys and evaluator credentials are not reachable from it"
        ),
    )


def network_report(
    *,
    network_access: bool,
    enforceable: bool,
    mechanism: str | None = None,
) -> IsolationControlReport:
    """Report the network control, which only an isolating backend can impose."""
    if network_access:
        return IsolationControlReport(
            control=IsolationControl.NETWORK,
            outcome=IsolationOutcome.NOT_REQUESTED,
            requested="allowed",
        )
    if enforceable:
        return IsolationControlReport(
            control=IsolationControl.NETWORK,
            outcome=IsolationOutcome.ENFORCED,
            mechanism=mechanism,
            requested="denied",
        )
    return IsolationControlReport(
        control=IsolationControl.NETWORK,
        outcome=IsolationOutcome.UNENFORCEABLE,
        requested="denied",
        detail=(
            "a local subprocess inherits host network reachability; denying it "
            "needs a network namespace this backend does not create. Use the "
            "container backend for an enforced denial."
        ),
    )


def filesystem_report(
    *,
    enforceable: bool,
    mechanism: str | None = None,
) -> IsolationControlReport:
    """Report filesystem confinement, distinguishing aiming from confining."""
    if enforceable:
        return IsolationControlReport(
            control=IsolationControl.FILESYSTEM_SCOPE,
            outcome=IsolationOutcome.ENFORCED,
            mechanism=mechanism,
            requested="workspace only",
        )
    return IsolationControlReport(
        control=IsolationControl.FILESYSTEM_SCOPE,
        outcome=IsolationOutcome.UNENFORCEABLE,
        requested="workspace only",
        detail=(
            "the working directory is pinned to the workspace and artifacts "
            "resolving outside it are refused, but a local process retains the "
            "host write permissions of the user running the benchmark"
        ),
    )


__all__ = [
    "IsolationControl",
    "IsolationControlReport",
    "IsolationOutcome",
    "IsolationReport",
    "IsolationRequest",
    "build_limit_setter",
    "describe_process_limits",
    "environment_report",
    "filesystem_report",
    "network_report",
    "supports_resource_limits",
]
