"""Running an agent's own code in an observed, isolated workspace.

The package is layered so that nothing above it branches on which backend ran:

- :mod:`fingerprint` -- content identity of a workspace tree, for observing what
  an execution changed rather than asking it.
- :mod:`isolation` -- what the host can actually enforce, reported per control so
  a run never claims a guarantee it did not have.
- :mod:`backend` -- the :class:`WorkspaceBackend` port plus its request and
  outcome models.
- :mod:`local` / :mod:`container` -- the two implementations.
- :mod:`executor` -- :class:`WorkspaceActionExecutor`, which adapts all of the
  above to the pre-existing ``ActionExecutor`` port, so episodes, decisions,
  trajectories, and scoring continue to work unchanged above this layer.

:mod:`container` is imported lazily by name rather than re-exported eagerly, so
importing this package does not require a container runtime to be installed.
"""

from agent_evals.environment.execution.backend import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    CommandOutcome,
    CommandRequest,
    Language,
    WorkspaceBackend,
    environment_allowlist,
    interpreter_argv,
)
from agent_evals.environment.execution.executor import (
    EXECUTION_PARAMETERS,
    WorkspaceActionExecutor,
    WorkspaceExecutionError,
    command_timeout,
    declared_artifacts,
    deterministic_environment,
    isolation_from_constraints,
)
from agent_evals.environment.execution.fingerprint import (
    DigestMethod,
    FileFingerprint,
    WorkspaceFingerprint,
    fingerprint_file,
    fingerprint_workspace,
)
from agent_evals.environment.execution.isolation import (
    IsolationControl,
    IsolationControlReport,
    IsolationOutcome,
    IsolationReport,
    IsolationRequest,
    describe_process_limits,
    supports_resource_limits,
)
from agent_evals.environment.execution.local import LocalProcessBackend, resolve_within

__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "EXECUTION_PARAMETERS",
    "CommandOutcome",
    "CommandRequest",
    "DigestMethod",
    "FileFingerprint",
    "IsolationControl",
    "IsolationControlReport",
    "IsolationOutcome",
    "IsolationReport",
    "IsolationRequest",
    "Language",
    "LocalProcessBackend",
    "WorkspaceActionExecutor",
    "WorkspaceBackend",
    "WorkspaceExecutionError",
    "WorkspaceFingerprint",
    "command_timeout",
    "declared_artifacts",
    "describe_process_limits",
    "deterministic_environment",
    "environment_allowlist",
    "fingerprint_file",
    "fingerprint_workspace",
    "interpreter_argv",
    "isolation_from_constraints",
    "resolve_within",
    "supports_resource_limits",
]
