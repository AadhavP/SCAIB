"""Adapt free-form workspace execution to the typed ``ActionExecutor`` port.

This is the load-bearing join of the whole design. ``ScientificEnvironment``
delegates every computation through ``ActionExecutor``, so an executor that runs
the agent's own code slots in *beneath* the existing port -- episodes, action
records, decision extraction, trajectories and scoring keep working untouched,
because none of them ever knew what was on the other side of the port.

Two rules are enforced here rather than trusted:

**A declared artifact must exist.** The agent says what its code will produce;
this checks the workspace and fails the action when it did not. Verifying rather
than believing is the same principle the benchmark applies to completion claims,
applied at the point where a claim first enters the system.

**A declared path must resolve inside the workspace.** Without that,
``produces: ["../reference.csv.gz"]`` would let an agent name a file the
evaluator owns as its own output. The local tier cannot *confine* writes and
says so, but it can refuse to account for anything outside the workspace, and
that refusal is what keeps the Stage 0 reference store outside the boundary.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from agent_evals.benchmarks.schema import ConstraintSpecification
from agent_evals.core.intent_parameters import (
    CODE_PARAMETER,
    EXECUTION_PARAMETERS,
    LANGUAGE_PARAMETER,
    PRODUCES_PARAMETER,
)
from agent_evals.environment.execution.backend import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    CommandOutcome,
    CommandRequest,
    Language,
    WorkspaceBackend,
)
from agent_evals.environment.execution.isolation import IsolationRequest
from agent_evals.environment.execution.local import resolve_within
from agent_evals.environment.models import (
    ActionExecutionResult,
    ActionIntent,
    ActionStatus,
    ArtifactRecord,
    ExecutionStatus,
    Observation,
    ResourceUsage,
    utc_now,
)
from agent_evals.environment.ports import ExecutionContext

#: Fallback wall-clock ceiling per command when the benchmark caps only the
#: whole episode. A single command may not consume the entire episode budget,
#: or one runaway step leaves the agent no chance to recover from it.
_COMMAND_TIMEOUT_FRACTION = 0.5

_FORMAT_KINDS = {
    "h5ad": "dataset",
    "csv": "table",
    "tsv": "table",
    "parquet": "table",
    "json": "record",
    "png": "figure",
    "pdf": "figure",
    "svg": "figure",
    "npy": "array",
    "npz": "array",
}


class WorkspaceExecutionError(ValueError):
    """Raised when an intent cannot be turned into a runnable request."""


def isolation_from_constraints(
    constraints: ConstraintSpecification,
) -> IsolationRequest:
    """Translate benchmark constraints into a backend-neutral isolation request.

    This function is the only place the execution layer meets the benchmark DSL,
    which is what keeps the backends reusable by anything that does not speak it.
    It is also where four long-declared-but-inert constraint fields finally get a
    consumer.
    """
    return IsolationRequest(
        network_access=constraints.internet_access,
        max_memory_mb=constraints.max_memory_mb,
        max_cpu_seconds=constraints.max_runtime_seconds,
        max_processes=None,
        max_file_size_mb=None,
    )


def command_timeout(constraints: ConstraintSpecification) -> float:
    """Choose a per-command ceiling from the episode-wide runtime limit."""
    if constraints.max_runtime_seconds is None:
        return DEFAULT_TIMEOUT_SECONDS
    return max(constraints.max_runtime_seconds * _COMMAND_TIMEOUT_FRACTION, 1.0)


def deterministic_environment(constraints: ConstraintSpecification) -> dict[str, str]:
    """Return environment variables that make a run reproducible.

    ``PYTHONHASHSEED`` has to be in the environment because the interpreter reads
    it before any user code runs; a seed set from inside the script is already
    too late to affect string hashing and therefore set iteration order.
    """
    if not constraints.deterministic or constraints.random_seed is None:
        return {}
    seed = str(constraints.random_seed)
    return {
        "PYTHONHASHSEED": seed,
        "SCAIB_RANDOM_SEED": seed,
    }


def declared_artifacts(intent: ActionIntent) -> dict[str, str]:
    """Read the artifact contract from the intent, accepting both spellings.

    A list declares paths whose id is the path itself; a mapping declares
    ``id -> path`` so a benchmark can require ``clusters`` without dictating
    where the agent writes it.
    """
    raw = intent.parameters.get(PRODUCES_PARAMETER)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {str(value): str(value) for value in raw}
    raise WorkspaceExecutionError(
        f"'{PRODUCES_PARAMETER}' must be a list of paths or a mapping of "
        f"artifact id to path, got {type(raw).__name__}"
    )


def _resolve_language(intent: ActionIntent) -> Language:
    """Resolve the requested language, rejecting one that is not supported."""
    raw = intent.parameters.get(LANGUAGE_PARAMETER)
    if raw is None:
        return Language.PYTHON
    try:
        return Language(str(raw).lower())
    except ValueError as error:
        supported = ", ".join(sorted(item.value for item in Language))
        raise WorkspaceExecutionError(
            f"unsupported language '{raw}'; this environment runs {supported}"
        ) from error


def _artifact_format(path: str) -> str:
    """Return a file's declared format from its extension."""
    suffix = Path(path).suffix.lstrip(".").lower()
    return suffix or "unknown"


def _artifact_kind(fmt: str) -> str:
    """Map a format to the coarse artifact kind the benchmark schema uses."""
    return _FORMAT_KINDS.get(fmt, "file")


def _checksum(path: Path) -> str:
    """Return a file's SHA-256, so a later step cannot silently swap it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class WorkspaceActionExecutor:
    """Run agent-authored code for one action intent and report what it did."""

    def __init__(
        self,
        backend: WorkspaceBackend,
        *,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        self.backend = backend
        self.max_output_bytes = max_output_bytes

    async def execute(
        self,
        intent: ActionIntent,
        context: ExecutionContext,
    ) -> ActionExecutionResult:
        """Execute one intent, returning a typed result for any outcome."""
        started = utc_now()
        try:
            request = self._build_request(intent, context)
            expected = declared_artifacts(intent)
            escapes = self._escaping_paths(expected)
        except WorkspaceExecutionError as error:
            return self._failure(intent, str(error), started=started)
        if escapes:
            return self._failure(
                intent,
                "declared artifact path(s) resolve outside the workspace and "
                f"were refused: {', '.join(escapes)}",
                started=started,
            )
        outcome = await self.backend.run(request)
        return self._result(intent, outcome, expected, started=started)

    def _build_request(
        self,
        intent: ActionIntent,
        context: ExecutionContext,
    ) -> CommandRequest:
        """Turn an intent into a command request, or explain why it cannot be."""
        source = intent.parameters.get(CODE_PARAMETER)
        if not isinstance(source, str) or not source.strip():
            raise WorkspaceExecutionError(
                f"action '{intent.action_id}' requires a non-empty "
                f"'{CODE_PARAMETER}' parameter holding the source to run"
            )
        constraints = context.constraints
        return CommandRequest(
            command=source,
            language=_resolve_language(intent),
            timeout_seconds=command_timeout(constraints),
            env=deterministic_environment(constraints),
            max_output_bytes=self.max_output_bytes,
            label=intent.action_id,
        )

    def _escaping_paths(self, expected: dict[str, str]) -> list[str]:
        """Return declared paths that do not resolve inside the workspace."""
        return sorted(
            f"{artifact_id} -> {path}"
            for artifact_id, path in expected.items()
            if resolve_within(self.backend.root, path) is None
        )

    def _result(
        self,
        intent: ActionIntent,
        outcome: CommandOutcome,
        expected: dict[str, str],
        *,
        started: datetime,
    ) -> ActionExecutionResult:
        """Build the typed result, verifying declared artifacts really exist."""
        observations = self._observations(intent, outcome)
        if not outcome.succeeded:
            return ActionExecutionResult(
                intent_id=intent.intent_id,
                action_id=intent.action_id,
                status=ActionStatus.FAILED,
                execution_status=outcome.status,
                observations=observations,
                resource_usage=outcome.resource_usage,
                error=outcome.error or f"execution failed: {outcome.status.value}",
                started_at=started,
            )
        artifacts, missing = self._collect_artifacts(expected)
        if missing:
            return ActionExecutionResult(
                intent_id=intent.intent_id,
                action_id=intent.action_id,
                status=ActionStatus.FAILED,
                # The code ran; the contract it declared went unmet.
                execution_status=ExecutionStatus.PARTIAL,
                observations=observations,
                artifacts=artifacts,
                resource_usage=outcome.resource_usage,
                error=(
                    "execution succeeded but declared artifact(s) were not "
                    f"produced: {', '.join(missing)}"
                ),
                started_at=started,
            )
        return ActionExecutionResult(
            intent_id=intent.intent_id,
            action_id=intent.action_id,
            status=ActionStatus.SUCCEEDED,
            execution_status=outcome.status,
            outputs={record.artifact_id: expected[record.artifact_id] for record in artifacts},
            observations=observations,
            artifacts=artifacts,
            resource_usage=outcome.resource_usage,
            started_at=started,
        )

    def _collect_artifacts(
        self,
        expected: dict[str, str],
    ) -> tuple[list[ArtifactRecord], list[str]]:
        """Record every declared artifact that exists; name those that do not."""
        records: list[ArtifactRecord] = []
        missing: list[str] = []
        for artifact_id, declared in sorted(expected.items()):
            resolved = resolve_within(self.backend.root, declared)
            if resolved is None or not resolved.is_file():
                missing.append(f"{artifact_id} at '{declared}'")
                continue
            fmt = _artifact_format(declared)
            records.append(
                ArtifactRecord(
                    artifact_id=artifact_id,
                    kind=_artifact_kind(fmt),
                    format=fmt,
                    uri=resolved.as_uri(),
                    checksum=_checksum(resolved),
                    # Existence and integrity are verified here; whether the
                    # contents satisfy the benchmark's validation rules is the
                    # evaluator's judgement, not this layer's.
                    validated=False,
                    metadata={
                        "declared_path": declared,
                        "size_bytes": resolved.stat().st_size,
                    },
                )
            )
        return records, missing

    def _observations(
        self,
        intent: ActionIntent,
        outcome: CommandOutcome,
    ) -> list[Observation]:
        """Expose the execution's own output to the agent, isolation to us.

        The agent needs its stdout and stderr back -- that is how it debugs, and
        withholding it would make the benchmark a test of blind coding. The
        isolation report is evaluator-only: it describes the harness's
        guarantees, and an agent that can read which controls went unenforced has
        been handed a map of what it can get away with.
        """
        source = f"execution:{intent.action_id}"
        return [
            Observation(
                observation_id="execution-stdout",
                value=outcome.stdout,
                source=source,
                metadata={"truncated": outcome.stdout_truncated},
            ),
            Observation(
                observation_id="execution-stderr",
                value=outcome.stderr,
                source=source,
                metadata={"truncated": outcome.stderr_truncated},
            ),
            Observation(
                observation_id="execution-status",
                value=outcome.status.value,
                source=source,
                metadata={"exit_code": outcome.exit_code},
            ),
            Observation(
                observation_id="execution-isolation",
                value=outcome.isolation.model_dump(mode="json"),
                source=source,
                visible_to_agent=False,
            ),
        ]

    def _failure(
        self,
        intent: ActionIntent,
        message: str,
        *,
        started: datetime,
    ) -> ActionExecutionResult:
        """Report a malformed intent without ever running anything."""
        return ActionExecutionResult(
            intent_id=intent.intent_id,
            action_id=intent.action_id,
            status=ActionStatus.FAILED,
            execution_status=ExecutionStatus.ERROR,
            resource_usage=ResourceUsage(),
            error=message,
            started_at=started,
        )


__all__ = [
    "CODE_PARAMETER",
    "EXECUTION_PARAMETERS",
    "LANGUAGE_PARAMETER",
    "PRODUCES_PARAMETER",
    "WorkspaceActionExecutor",
    "WorkspaceExecutionError",
    "command_timeout",
    "declared_artifacts",
    "deterministic_environment",
    "isolation_from_constraints",
]
