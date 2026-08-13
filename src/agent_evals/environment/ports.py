"""Dependency-injection ports for scientific environment implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from agent_evals.benchmarks.schema import (
    ActionSpecification,
    BenchmarkSpecification,
    ConstraintSpecification,
    TaskSpecification,
    ValidationRule,
)
from agent_evals.environment.models import (
    ActionExecutionResult,
    ActionIntent,
    ActionValidationResult,
    ArtifactRecord,
    ArtifactValidation,
    EpisodeSnapshot,
    Observation,
    ResourceUsage,
    RewardRecord,
)


class ExecutionContext:
    """Context handed to an action executor without exposing environment internals."""

    def __init__(self, snapshot: EpisodeSnapshot, constraints: ConstraintSpecification) -> None:
        self.snapshot = snapshot
        self.constraints = constraints


@runtime_checkable
class ActionExecutor(Protocol):
    """Port implemented by local, containerized, remote, or simulated tools."""

    async def execute(
        self,
        intent: ActionIntent,
        context: ExecutionContext,
    ) -> ActionExecutionResult:
        """Execute a validated intent and return typed outputs."""


@runtime_checkable
class ObservationBuilder(Protocol):
    """Port that derives agent-visible observations from episode state."""

    async def build(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
    ) -> Sequence[Observation]:
        """Build observations without mutating the episode directly."""


class DeclaredObservationBuilder:
    """Serve the observation values a benchmark declares for itself.

    ``ObservationSpecification.schema_definition`` (spelled ``schema:`` in YAML) is
    where a benchmark states a value it *owns* rather than one the harness
    measures.  The differential-expression benchmark uses it to declare
    ``comparison-definition`` -- the shape of the contrast the agent is supposed to
    test.  Nothing read that field, so the contrast arrived as ``{}`` and the task
    was unrunnable as written: an agent was asked to test a comparison it was never
    told.

    Whatever a benchmark puts here is published to the agent **verbatim**, which is
    the constraint on what may be put here at all.  That benchmark's declaration
    originally named the two reference populations to contrast, and this builder is
    what turned it from an inert YAML field into a live disclosure of two withheld
    class names; it now declares only ``{grouping: agent_defined, contrast:
    one_versus_rest, direction: up_in_group}``, which states the contrast's shape
    without naming anything the evaluator holds back.

    Deliberately lowest precedence in :class:`CompositeObservationBuilder`. A
    declaration is what the benchmark intends; a measurement is what is true, and
    where both exist the measurement is the observation.
    """

    async def build(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
    ) -> list[Observation]:
        """Build one observation per declared value the task asks for."""
        declared = {item.id: item for item in specification.observations}
        built: list[Observation] = []
        for observation_id in dict.fromkeys(task.observations):
            definition = declared.get(observation_id)
            if definition is None or not definition.schema_definition:
                continue
            built.append(
                Observation(
                    observation_id=observation_id,
                    value=dict(definition.schema_definition),
                    source="benchmark-declaration",
                    step=snapshot.state.current_step,
                    metadata={
                        "declared_type": definition.type,
                        "declared_source": definition.source,
                    },
                )
            )
        return built


class CompositeObservationBuilder:
    """Run several observation builders and merge what each of them serves.

    ``ScientificEnvironment`` holds exactly one builder, so a benchmark whose
    observations come from more than one place -- its own declaration, the
    scientific state, the workspace on disk -- needs something in that slot to
    combine them.  This is itself an :class:`ObservationBuilder`, so nothing above
    the port learns that several coexist; the same shape ``ActionKindRouter`` uses
    to let typed and free execution share one ``ActionExecutor``.

    Later builders win on a shared id, which makes the argument order the
    precedence order.  It has to be explicit because the alternative was implicit
    and wrong: observations are stored by id, and a builder that emitted a
    placeholder for an id it did not serve overwrote the real value a moment after
    the executor committed it.
    """

    def __init__(self, *builders: ObservationBuilder) -> None:
        self.builders = builders

    async def build(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
    ) -> list[Observation]:
        """Merge every builder's output, last writer winning per observation id."""
        merged: dict[str, Observation] = {}
        for builder in self.builders:
            for observation in await builder.build(specification, task, snapshot):
                merged[observation.observation_id] = observation
        return list(merged.values())


@runtime_checkable
class ArtifactValidator(Protocol):
    """Port that checks a produced artifact against its declared rules.

    A port rather than a call inside an executor, because both execution tiers
    have to reach the same verdict about the same file.  Under typed execution
    the harness wrote the artifact and under free execution the agent's own code
    did, and if each tier decided validity for itself the two would drift --
    which matters because ``validated`` is scored, so a tier that judged itself
    leniently would score better for reasons that are not scientific.

    Asynchronous because an implementation has to read files, and an ``.h5ad``
    read is slow enough to stall the event loop that feeds progress to the UI.
    """

    async def validate(
        self,
        artifact: ArtifactRecord,
        rules: Sequence[ValidationRule],
        parameters: Mapping[str, Any],
    ) -> ArtifactValidation:
        """Return what checking this artifact established, without raising.

        ``parameters`` are the producing intent's parameters, which is where a
        rule's referenced vocabulary is declared; a rule naming one that is not
        there is unevaluated rather than failed.
        """


@runtime_checkable
class RewardEvaluator(Protocol):
    """Port for metric and reward computation owned by the evaluation layer."""

    async def evaluate(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
        result: ActionExecutionResult,
    ) -> RewardRecord | None:
        """Return a reward record or ``None`` when no reward is emitted."""


class DeclarativeActionValidator:
    """Validate intents using only the benchmark specification and episode snapshot."""

    def validate(
        self,
        intent: ActionIntent,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
    ) -> ActionValidationResult:
        """Return all permission, input, parameter, and constraint errors."""
        errors: list[str] = []
        action = next((item for item in specification.actions if item.id == intent.action_id), None)
        if action is None:
            errors.append(f"unknown action '{intent.action_id}'")
            return ActionValidationResult(valid=False, errors=errors)
        if intent.action_id not in task.allowed_actions:
            errors.append(f"action '{intent.action_id}' is not allowed for task '{task.id}'")
        errors.extend(self._validate_inputs(action, snapshot))
        errors.extend(self._validate_parameters(action, intent))
        return ActionValidationResult(valid=not errors, errors=errors)

    @staticmethod
    def apply_defaults(
        intent: ActionIntent,
        specification: BenchmarkSpecification,
    ) -> ActionIntent:
        """Materialize declared parameter defaults before execution.

        Defaults used to be documentation only: validation accepted an omitted
        value, but the executor saw its own unrelated fallback. That makes a
        benchmark's advertised method/threshold contract different from the
        experiment that actually ran. Materializing them once at the boundary
        keeps the submitted intent, execution, and trajectory aligned.
        """
        action = next((item for item in specification.actions if item.id == intent.action_id), None)
        if action is None:
            return intent
        parameters = dict(intent.parameters)
        for parameter in action.parameters:
            if parameter.name not in parameters and parameter.default is not None:
                parameters[parameter.name] = parameter.default
        return intent.model_copy(update={"parameters": parameters})

    @staticmethod
    def _validate_inputs(
        action: ActionSpecification,
        snapshot: EpisodeSnapshot,
    ) -> list[str]:
        """Ensure all declaratively required observations and artifacts exist."""
        available = set(snapshot.state.observations) | set(snapshot.state.artifacts)
        missing = sorted(set(action.required_inputs) - available)
        return [f"action '{action.id}' is missing input '{item}'" for item in missing]

    @staticmethod
    def _validate_parameters(
        action: ActionSpecification,
        intent: ActionIntent,
    ) -> list[str]:
        """Validate parameter names, required values, choices, and numeric bounds."""
        errors: list[str] = []
        definitions = {parameter.name: parameter for parameter in action.parameters}
        unknown = sorted(set(intent.parameters) - set(definitions))
        errors.extend(f"action '{action.id}' has unknown parameter '{item}'" for item in unknown)
        for name, parameter in definitions.items():
            if parameter.required and name not in intent.parameters and parameter.default is None:
                errors.append(f"action '{action.id}' is missing required parameter '{name}'")
                continue
            if name not in intent.parameters:
                continue
            value = intent.parameters[name]
            if parameter.choices and value not in parameter.choices:
                errors.append(
                    f"parameter '{name}' must be one of {parameter.choices}, got {value!r}"
                )
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if parameter.minimum is not None and value < parameter.minimum:
                    errors.append(f"parameter '{name}' is below minimum {parameter.minimum}")
                if parameter.maximum is not None and value > parameter.maximum:
                    errors.append(f"parameter '{name}' exceeds maximum {parameter.maximum}")
        return errors

class ConstraintMonitor:
    """Check measured executor usage against declarative resource limits."""

    def check(
        self,
        constraints: ConstraintSpecification,
        usage: ResourceUsage,
        previous_usage: ResourceUsage | None = None,
    ) -> list[str]:
        """Return cumulative resource violations without mutating state."""
        cumulative = self._combine(previous_usage, usage)
        errors: list[str] = []
        if (
            constraints.max_runtime_seconds is not None
            and cumulative.wall_time_seconds > constraints.max_runtime_seconds
        ):
            errors.append("episode exceeded maximum runtime")
        if (
            constraints.max_memory_mb is not None
            and cumulative.peak_memory_mb is not None
            and cumulative.peak_memory_mb > constraints.max_memory_mb
        ):
            errors.append("episode exceeded maximum memory")
        if constraints.cpu_only and cumulative.gpu_used:
            errors.append("episode used a GPU under CPU-only constraints")
        if constraints.gpu_required and not cumulative.gpu_used:
            errors.append("episode did not use a required GPU")
        return errors

    @staticmethod
    def _combine(
        previous: ResourceUsage | None,
        current: ResourceUsage,
    ) -> ResourceUsage:
        """Combine cumulative totals with the current action measurement."""
        if previous is None:
            return current
        return ResourceUsage(
            wall_time_seconds=previous.wall_time_seconds + current.wall_time_seconds,
            cpu_seconds=(
                (previous.cpu_seconds or 0.0) + (current.cpu_seconds or 0.0)
                if previous.cpu_seconds is not None or current.cpu_seconds is not None
                else None
            ),
            peak_memory_mb=max(
                previous.peak_memory_mb or 0.0,
                current.peak_memory_mb or 0.0,
            )
            or None,
            gpu_used=previous.gpu_used or current.gpu_used,
        )


__all__ = [
    "ActionExecutor",
    "ArtifactValidator",
    "CompositeObservationBuilder",
    "ConstraintMonitor",
    "DeclarativeActionValidator",
    "DeclaredObservationBuilder",
    "ExecutionContext",
    "ObservationBuilder",
    "RewardEvaluator",
]
