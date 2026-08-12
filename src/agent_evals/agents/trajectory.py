"""Framework-independent agent runs, traces, and scientific decisions."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_evals.agents.decisions.verification import (
    DecisionVerification,
    verify_state_claim,
)
from agent_evals.core.intent_parameters import EXECUTION_PARAMETERS
from agent_evals.environment.models import (
    ActionStatus,
    ArtifactRecord,
    EpisodeSnapshot,
    EventType,
    ResourceUsage,
    StateDelta,
)


class AgentRuntimeModel(BaseModel):
    """Strict base model for persisted harness data."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AgentConfiguration(AgentRuntimeModel):
    """Provider-neutral configuration translated by an adapter backend."""

    agent_type: str = Field(min_length=1)
    model: str | None = None
    provider: str | None = None
    temperature: float | None = None
    max_steps: int | None = Field(default=None, gt=0)
    timeout_seconds: int | None = Field(default=None, gt=0)
    seed: int = 0
    workspace: dict[str, Any] = Field(default_factory=dict)
    tools: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentModelInfo(AgentRuntimeModel):
    """Provider/model identity retained for leaderboard comparison."""

    provider: str | None = None
    name: str | None = None


class AgentManifest(AgentRuntimeModel):
    """Public agent capability metadata, without private chain-of-thought."""

    name: str
    type: str
    model: AgentModelInfo = Field(default_factory=AgentModelInfo)
    capabilities: list[str] = Field(default_factory=list)
    temperature: float | None = None
    context_window: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(AgentRuntimeModel):
    """Optional model token accounting reported by an agent backend."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class EstimatedCost(AgentRuntimeModel):
    """Optional estimated monetary cost of one agent run."""

    amount: float = Field(ge=0)
    currency: str = "USD"
    source: str | None = None


class RunTerminationStatus(StrEnum):
    """Normalized terminal status for an agent harness run."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"
    INVALID_CONFIGURATION = "invalid_configuration"
    #: The agent stopped of its own accord and the declared artifact contract was
    #: not met. Distinct from :attr:`FAILED`, which is for a run that broke: an
    #: incomplete run executed cleanly and simply stopped too early, and the two
    #: call for different reading. The runtime already produced this verdict and
    #: it was being flattened to ``FAILED`` on the way into the archive, so an
    #: agent that quit early was indistinguishable from one that crashed.
    INCOMPLETE = "incomplete"
    #: The controller stopped the run because it was no longer making measurable
    #: scientific progress, or was repeating work. Distinct from :attr:`TIMEOUT`,
    #: which means a budget was consumed: a stagnated run still had budget and was
    #: not using it productively, and reading the two as one would make an agent
    #: that looped indistinguishable from one that ran out of room to finish.
    STAGNATED = "stagnated"


class FailureKind(StrEnum):
    """Failure categories retained in partial agent runs."""

    AGENT_ERROR = "agent_error"
    ADAPTER_UNAVAILABLE = "adapter_unavailable"
    INVALID_ACTION = "invalid_action"
    TOOL_ERROR = "tool_error"
    ENVIRONMENT_ERROR = "environment_error"
    WORKSPACE_ERROR = "workspace_error"
    RESOURCE_LIMIT = "resource_limit"
    TIMEOUT = "timeout"
    #: A verified-false completion claim: the agent said it was done and the
    #: required artifacts were not there. Nothing errored, so filing this under
    #: :attr:`AGENT_ERROR` misattributed a premature stop to a malfunction.
    INCOMPLETE_SUBMISSION = "incomplete_submission"
    #: The run was stopped for making no measurable progress, or for repeating
    #: itself. Not a :attr:`RESOURCE_LIMIT`: nothing was exhausted.
    STAGNATION = "stagnation"
    UNKNOWN = "unknown"


class AgentFailure(AgentRuntimeModel):
    """Structured error that does not discard preceding trajectory data."""

    kind: FailureKind
    message: str
    event_id: str | None = None
    recoverable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawTraceEvent(AgentRuntimeModel):
    """Framework-specific event preserved without forced normalization."""

    event_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    timestamp: datetime
    event_type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_event_id: str | None = None


class ParameterRange(AgentRuntimeModel):
    """Optional numeric or categorical bounds for an observed parameter."""

    minimum: float | None = None
    maximum: float | None = None
    choices: list[Any] = Field(default_factory=list)


class ParameterChoice(AgentRuntimeModel):
    """Typed, observable parameter selection made by an agent."""

    name: str = Field(min_length=1)
    value: Any
    type: str = "unknown"
    allowed_range: ParameterRange | None = None
    source: str = "action"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MethodChoice(AgentRuntimeModel):
    """Observable method selection with its associated parameters."""

    step_id: str = Field(min_length=1)
    method_id: str = Field(min_length=1)
    method_name: str = Field(min_length=1)
    implementation: str | None = None
    parameters: list[ParameterChoice] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionCategory(StrEnum):
    """Benchmark-independent scientific decision ontology."""

    DATA_LOADING = "data_loading"
    QC_STRATEGY = "qc_strategy"
    NORMALIZATION = "normalization"
    FEATURE_SELECTION = "feature_selection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    INTEGRATION = "integration"
    CLUSTERING = "clustering"
    ANNOTATION = "annotation"
    DIFFERENTIAL_EXPRESSION = "differential_expression"
    INTERPRETATION = "interpretation"
    OTHER = "other"


class NormalizedTrajectoryEvent(AgentRuntimeModel):
    """Framework-neutral scientific interaction event."""

    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    timestamp: datetime
    event_type: EventType
    source: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_event_id: str | None = None


class ScientificDecision(AgentRuntimeModel):
    """Explicitly observable scientific choice in a decision cascade.

    This model captures decisions exposed through structured actions, messages,
    tools, artifacts, or environment transitions. It never stores private
    chain-of-thought and does not infer biological correctness.
    """

    decision_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    order: int = Field(ge=0)
    decision_type: str = "step"
    action_category: str = Field(min_length=1)
    decision_category: DecisionCategory = DecisionCategory.OTHER
    intent: str | None = None
    hypothesis: str | None = None
    #: Step the agent's own plan said this was, when it said so. Recorded so a
    #: reader can ask whether the agent followed its stated plan or abandoned
    #: it, which is a finding either way. It is never checked against the plan
    #: to *reject* a decision -- a plan is an evaluation object, not a
    #: constraint, and an agent that adapts when the data contradicts its plan
    #: is doing science rather than disobeying.
    plan_reference: str | None = None
    method: str | None = None
    chosen_method: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    chosen_parameters: dict[str, Any] = Field(default_factory=dict)
    evidence_used: list[str] = Field(default_factory=list)
    rationale: str | None = None
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    execution_status: ActionStatus | None = None
    parent_decision_id: str | None = None
    predecessor_decision_ids: list[str] = Field(default_factory=list)
    dependency_decision_ids: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    timestamp: datetime
    selected_value: Any | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    expected_effect: dict[str, float] = Field(default_factory=dict)
    downstream_dependency: dict[str, Any] = Field(default_factory=dict)
    #: What the agent said this step did to the data. Recorded because a claim is
    #: evidence about the agent even when it is false -- and especially then.
    claimed_state_delta: dict[str, Any] = Field(default_factory=dict)
    #: What the harness measured the step doing, by comparing state before and
    #: after. ``None`` when nothing was observed, which is not the same as
    #: nothing having happened.
    observed_state_delta: StateDelta | None = None
    #: The comparison of the two above. Populated whenever either exists, so a
    #: reader never has to redo the comparison to find out whether it was done.
    verification: DecisionVerification | None = None
    #: What this one step cost. Present per decision, not just per run, because
    #: an efficiency claim about a *trajectory* needs to know which steps were
    #: expensive rather than only what the total was.
    resource_usage: ResourceUsage | None = None
    #: Which agent produced this decision, for runs with more than one. Opaque
    #: on purpose: topology is metadata the benchmark records, never something it
    #: scores, so this is a label rather than a structure.
    agent_origin: str | None = None
    method_choice: MethodChoice | None = None
    parameter_choice: ParameterChoice | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def synchronize_choice_views(self) -> Self:
        """Keep legacy and explicit choice fields consistent."""
        if self.chosen_method is None:
            object.__setattr__(self, "chosen_method", self.method)
        if self.method is None:
            object.__setattr__(self, "method", self.chosen_method)
        if not self.chosen_parameters:
            object.__setattr__(self, "chosen_parameters", dict(self.parameters))
        if not self.parameters:
            object.__setattr__(self, "parameters", dict(self.chosen_parameters))
        return self


class DecisionCascade(AgentRuntimeModel):
    """Hierarchical and dependency-aware collection of scientific decisions."""

    decisions: list[ScientificDecision] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        """Reject duplicate IDs, unknown references, and decision cycles."""
        decision_ids = [decision.decision_id for decision in self.decisions]
        duplicates = sorted({item for item in decision_ids if decision_ids.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate decision identifier(s): {', '.join(duplicates)}")
        known = set(decision_ids)
        for decision in self.decisions:
            references = [
                reference
                for reference in [
                    decision.parent_decision_id,
                    *decision.predecessor_decision_ids,
                    *decision.dependency_decision_ids,
                ]
                if reference is not None
            ]
            unknown = sorted(set(references) - known)
            if unknown:
                raise ValueError(
                    f"decision '{decision.decision_id}' references unknown decision(s): "
                    f"{', '.join(unknown)}"
                )
        graph = {
            decision.decision_id: [
                reference
                for reference in [
                    decision.parent_decision_id,
                    *decision.predecessor_decision_ids,
                    *decision.dependency_decision_ids,
                ]
                if reference is not None
            ]
            for decision in self.decisions
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(decision_id: str) -> None:
            if decision_id in visiting:
                raise ValueError(f"circular decision dependency at '{decision_id}'")
            if decision_id in visited:
                return
            visiting.add(decision_id)
            for dependency in graph[decision_id]:
                visit(dependency)
            visiting.remove(decision_id)
            visited.add(decision_id)

        for decision_id in graph:
            visit(decision_id)
        return self


class NormalizedTrajectory(AgentRuntimeModel):
    """Persisted framework-neutral event stream plus decision cascade."""

    trajectory_version: str = "1.0.0"
    run_id: str
    episode_id: str
    events: list[NormalizedTrajectoryEvent] = Field(default_factory=list)
    decisions: DecisionCascade = Field(default_factory=DecisionCascade)

    def to_json(self) -> str:
        """Serialize the trajectory using JSON-compatible Pydantic output."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, payload: str) -> NormalizedTrajectory:
        """Restore a trajectory from canonical JSON."""
        return cls.model_validate_json(payload)


class AgentRun(AgentRuntimeModel):
    """Normalized result of one adapter execution inside one episode."""

    run_id: str
    agent_id: str
    configuration: AgentConfiguration
    manifest: AgentManifest | None = None
    model: str | None = None
    provider: str | None = None
    adapter_name: str
    adapter_version: str
    benchmark_id: str
    task_id: str
    dataset_id: str | None = None
    dataset_version: str | None = None
    episode_id: str
    started_at: datetime
    finished_at: datetime
    termination_status: RunTerminationStatus
    termination_reason: str | None = None
    token_usage: TokenUsage | None = None
    estimated_cost: EstimatedCost | None = None
    wall_clock_seconds: float = Field(ge=0)
    step_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    raw_events: list[RawTraceEvent] = Field(default_factory=list)
    trajectory: NormalizedTrajectory
    generated_artifacts: list[ArtifactRecord] = Field(default_factory=list)
    final_environment_state: EpisodeSnapshot
    failures: list[AgentFailure] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the complete run without using pickle."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, payload: str) -> AgentRun:
        """Restore an agent run from canonical JSON."""
        return cls.model_validate_json(payload)

    @property
    def succeeded(self) -> bool:
        """Return whether the adapter completed without recorded failures."""
        return self.termination_status == RunTerminationStatus.COMPLETED and not self.failures


def decision_cascade_from_episode(snapshot: EpisodeSnapshot) -> DecisionCascade:
    """Extract explicit decisions from observable episode action records."""
    decisions: list[ScientificDecision] = []
    for _order, record in enumerate(snapshot.state.actions):
        metadata = record.intent.metadata
        step_id = f"step-{record.step}"
        decision_id = f"decision-{record.step}"
        output_artifacts = [artifact.artifact_id for artifact in record.result.artifacts]
        method_name = metadata.get("method") or record.intent.parameters.get("method")
        category = _decision_category(metadata.get("decision_category"), record.intent.action_id)
        # Coerced again here, not redundantly: only runtime turns pass through
        # the decision extraction layer. ``mock``, ``action_mapper`` and
        # ``scientific/runner`` build intents directly, so for those paths this
        # is the only coercion these two fields ever get.
        evidence_used = [str(item) for item in metadata.get("evidence_used", [])]
        # The claim comes from the agent's metadata and the observation from the
        # executor's result, and they are read from those two separate places on
        # purpose: if the observation were ever derived from the claim, the
        # verification below would be checking the agent against itself.
        claimed_state_delta = dict(metadata.get("state_claim") or {})
        observed_state_delta = record.result.observed_state_delta
        verification = (
            verify_state_claim(claimed_state_delta, observed_state_delta)
            if claimed_state_delta or observed_state_delta is not None
            else None
        )
        # ``method`` is the method choice, not a parameter of it, and the
        # execution parameters are mechanics rather than methodology: a
        # free-execution step would otherwise emit a parameter decision whose
        # selected value is the agent's entire program, scored as if choosing a
        # script were a methodological choice comparable to choosing n_pcs=50.
        parameter_choices = [
            ParameterChoice(
                name=name,
                value=value,
                type=_value_type(value),
                source="action_intent",
            )
            for name, value in record.intent.parameters.items()
            if name != "method" and name not in EXECUTION_PARAMETERS
        ]
        method_choice = (
            MethodChoice(
                step_id=step_id,
                method_id=str(metadata.get("method_id") or method_name),
                method_name=str(method_name),
                implementation=metadata.get("implementation"),
                parameters=parameter_choices,
            )
            if method_name is not None
            else None
        )
        decisions.append(
            ScientificDecision(
                decision_id=decision_id,
                episode_id=snapshot.state.episode_id,
                step_id=step_id,
                order=len(decisions),
                decision_type=str(metadata.get("decision_type", "step_selection")),
                action_category=record.intent.action_id,
                decision_category=category,
                intent=metadata.get("intent"),
                hypothesis=metadata.get("hypothesis"),
                plan_reference=_optional_str(metadata.get("plan_reference")),
                method=str(method_name) if method_name is not None else None,
                chosen_method=str(method_name) if method_name is not None else None,
                parameters=record.intent.parameters,
                chosen_parameters=record.intent.parameters,
                evidence_used=evidence_used,
                rationale=record.intent.rationale,
                input_artifacts=list(metadata.get("input_artifacts", [])),
                output_artifacts=output_artifacts,
                alternatives_considered=list(metadata.get("alternatives_considered", [])),
                execution_status=record.result.status,
                parent_decision_id=metadata.get("parent_decision_id"),
                predecessor_decision_ids=list(metadata.get("predecessor_decision_ids", [])),
                dependency_decision_ids=list(metadata.get("dependency_decision_ids", [])),
                source_event_ids=list(metadata.get("source_event_ids", [])),
                timestamp=record.recorded_at,
                selected_value=metadata.get("selected_value", record.intent.action_id),
                confidence=metadata.get("confidence"),
                expected_effect={
                    str(key): float(value)
                    for key, value in metadata.get("expected_effect", {}).items()
                    if isinstance(value, (int, float))
                },
                downstream_dependency=dict(metadata.get("downstream_dependency", {})),
                claimed_state_delta=claimed_state_delta,
                observed_state_delta=observed_state_delta,
                verification=verification,
                resource_usage=record.result.resource_usage,
                agent_origin=_optional_str(metadata.get("agent_origin")),
                method_choice=method_choice,
                metadata=metadata,
            )
        )
        method_decision_id = f"{decision_id}-method"
        # The step decision above is the only one that carries the state claim,
        # the observation, and the cost. The method and parameter decisions below
        # are facets of that same execution, so repeating any of the three would
        # make one discrepancy look like three and one step's runtime look like
        # several. ``agent_origin`` does repeat, because attribution is a label
        # rather than a measurement and nothing aggregates it.
        agent_origin = _optional_str(metadata.get("agent_origin"))
        if method_choice is not None:
            decisions.append(
                ScientificDecision(
                    decision_id=method_decision_id,
                    episode_id=snapshot.state.episode_id,
                    step_id=step_id,
                    order=len(decisions),
                    decision_type="method_selection",
                    action_category=record.intent.action_id,
                    decision_category=category,
                    method=method_choice.method_name,
                    chosen_method=method_choice.method_name,
                    parameters=record.intent.parameters,
                    chosen_parameters=record.intent.parameters,
                    rationale=record.intent.rationale,
                    output_artifacts=output_artifacts,
                    execution_status=record.result.status,
                    parent_decision_id=decision_id,
                    source_event_ids=list(metadata.get("source_event_ids", [])),
                    timestamp=record.recorded_at,
                    selected_value=method_choice.method_name,
                    agent_origin=agent_origin,
                    method_choice=method_choice,
                )
            )
        parameter_parent = method_decision_id if method_choice is not None else decision_id
        for parameter_choice in parameter_choices:
            decisions.append(
                ScientificDecision(
                    decision_id=f"{decision_id}-parameter-{parameter_choice.name}",
                    episode_id=snapshot.state.episode_id,
                    step_id=step_id,
                    order=len(decisions),
                    decision_type="parameter_selection",
                    action_category=record.intent.action_id,
                    decision_category=category,
                    method=str(method_name) if method_name is not None else None,
                    chosen_method=str(method_name) if method_name is not None else None,
                    parameters={parameter_choice.name: parameter_choice.value},
                    chosen_parameters={parameter_choice.name: parameter_choice.value},
                    rationale=record.intent.rationale,
                    output_artifacts=output_artifacts,
                    execution_status=record.result.status,
                    parent_decision_id=parameter_parent,
                    source_event_ids=list(metadata.get("source_event_ids", [])),
                    timestamp=record.recorded_at,
                    selected_value=parameter_choice.value,
                    agent_origin=agent_origin,
                    parameter_choice=parameter_choice,
                )
            )
    return DecisionCascade(decisions=decisions)


def _decision_category(value: Any, action_id: str) -> DecisionCategory:
    """Map environment actions to the stable ontology without private inference."""
    if value is not None:
        try:
            return DecisionCategory(str(value))
        except ValueError:
            pass
    mapping = {
        "load": DecisionCategory.DATA_LOADING,
        "qc": DecisionCategory.QC_STRATEGY,
        "normalize": DecisionCategory.NORMALIZATION,
        "select_hvg": DecisionCategory.FEATURE_SELECTION,
        "pca": DecisionCategory.DIMENSIONALITY_REDUCTION,
        "harmony": DecisionCategory.INTEGRATION,
        "batch_correct": DecisionCategory.INTEGRATION,
        "cluster": DecisionCategory.CLUSTERING,
        "annotate": DecisionCategory.ANNOTATION,
        "marker-genes": DecisionCategory.DIFFERENTIAL_EXPRESSION,
        "differential-expression": DecisionCategory.DIFFERENTIAL_EXPRESSION,
    }
    return mapping.get(action_id, DecisionCategory.OTHER)


def _optional_str(value: Any) -> str | None:
    """Coerce an untyped metadata value to text, keeping absence as absence."""
    return None if value is None else str(value)


def _value_type(value: Any) -> str:
    """Return a stable generic type name for an observed parameter value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


__all__ = [
    "AgentConfiguration",
    "AgentFailure",
    "AgentManifest",
    "AgentModelInfo",
    "AgentRun",
    "DecisionCascade",
    "DecisionCategory",
    "EstimatedCost",
    "FailureKind",
    "MethodChoice",
    "NormalizedTrajectory",
    "NormalizedTrajectoryEvent",
    "ParameterChoice",
    "ParameterRange",
    "RawTraceEvent",
    "RunTerminationStatus",
    "ScientificDecision",
    "TokenUsage",
    "decision_cascade_from_episode",
]
