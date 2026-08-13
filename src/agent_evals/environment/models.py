"""Typed runtime values exchanged by the scientific environment.

These models are the runtime counterpart to the declarative benchmark
specification.  They describe what happened during an episode without
implementing any scientific tool or sandbox backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return an aware UTC timestamp for reproducible episode records."""
    return datetime.now(UTC)


class RuntimeModel(BaseModel):
    """Strict base model shared by environment and episode records."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EpisodeStatus(StrEnum):
    """Lifecycle state of a scientific episode."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionStatus(StrEnum):
    """Outcome of a submitted action intent."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ExecutionStatus(StrEnum):
    """How an execution ended, at a finer grain than accepted/rejected.

    ``ActionStatus`` is the coarse gate the environment branches on and stays
    binary on purpose.  This enum records *why* an execution ended, which the
    binary gate cannot express: an agent whose script was killed by a memory
    limit made a different mistake from one whose script raised, and a run cut
    short by a timeout is not evidence about the science at all.  Reported
    alongside ``ActionStatus`` rather than replacing it, so persisted runs and
    the environment's own control flow are unaffected.
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    TIMEOUT = "timeout"
    OOM = "oom"
    TERMINATED = "terminated"


class EventType(StrEnum):
    """Event categories stored in the append-only episode trace."""

    CREATED = "episode.created"
    STARTED = "episode.started"
    OBSERVATIONS_UPDATED = "observations.updated"
    ACTION_SUBMITTED = "action.submitted"
    ACTION_REJECTED = "action.rejected"
    ACTION_COMPLETED = "action.completed"
    ARTIFACTS_UPDATED = "artifacts.updated"
    REWARD_RECORDED = "reward.recorded"
    TERMINATED = "episode.terminated"
    AGENT_MESSAGE = "agent.message"
    #: The agent stated a plan. Distinct from a message because a plan is an
    #: evaluation object: it is what a later decision can be compared against.
    #: Folded into ``AGENT_MESSAGE`` it becomes unfindable in the trace.
    PLAN_DECLARED = "agent.plan_declared"
    OBSERVATION_RECEIVED = "agent.observation_received"
    TOOL_CALL = "agent.tool_call"
    TOOL_RESULT = "agent.tool_result"
    ACTION_PROPOSED = "agent.action_proposed"
    ACTION_EXECUTED = "agent.action_executed"
    ARTIFACT_CREATED = "agent.artifact_created"
    ARTIFACT_MODIFIED = "agent.artifact_modified"
    COMMAND_EXECUTED = "agent.command_executed"
    ENVIRONMENT_STATE_CHANGED = "agent.environment_state_changed"
    AGENT_ERROR = "agent.error"


class ResourceUsage(RuntimeModel):
    """Measured resources reported by an action executor."""

    wall_time_seconds: float = Field(default=0.0, ge=0)
    cpu_seconds: float | None = Field(default=None, ge=0)
    peak_memory_mb: float | None = Field(default=None, ge=0)
    gpu_used: bool = False


class ActionIntent(RuntimeModel):
    """A typed request to perform one action declared by a benchmark.

    Agents submit intents rather than Python or shell source.  The environment
    validates the action ID, parameters, required inputs, and task permissions
    before an executor is allowed to see it.
    """

    intent_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    action_id: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    submitted_at: datetime = Field(default_factory=utc_now)


class Observation(RuntimeModel):
    """One value produced by the environment, agent-visible unless marked otherwise."""

    observation_id: str = Field(min_length=1)
    value: Any
    source: str = Field(min_length=1)
    step: int = Field(default=0, ge=0)
    #: Whether an agent may read this. Evaluator-only observations are recorded in
    #: the episode -- they are evidence -- but must be projected out of anything
    #: handed to a policy. Enforce it with :func:`agent_visible_observations`
    #: rather than by hand: a producer that sets this flag has no way to check
    #: that the consumer honoured it, and for six stages nothing did.
    visible_to_agent: bool = True
    produced_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


def agent_visible_observations(
    observations: Mapping[str, Observation],
) -> dict[str, Observation]:
    """Drop every observation its producer marked evaluator-only.

    The single chokepoint for :attr:`Observation.visible_to_agent`. It exists as a
    function rather than as a filter written at each call site because the flag
    was inert for six stages: the workspace executor set it on the isolation
    report -- the per-control map of what this host failed to enforce, which is
    exactly a map of what an agent can get away with -- and every agent-facing
    projection then serialized the episode's observations wholesale. Nothing
    raised, because a leaked observation blocks nothing.
    """
    return {
        key: value for key, value in observations.items() if value.visible_to_agent
    }


class RuleOutcome(StrEnum):
    """Verdict on one declared validation rule.

    The three-way split is the point.  A rule that ran and disagreed with the
    artifact is the agent's result; a rule nobody could run is the harness's
    blind spot.  Collapsing them would let a missing reader look like bad
    science, or -- worse in the other direction -- let an artifact that failed a
    check pass as merely unmeasured.
    """

    #: The artifact was read and satisfies the rule.
    PASSED = "passed"
    #: The artifact was read and does not satisfy the rule.
    FAILED = "failed"
    #: The rule could not be evaluated at all, and why is recorded.
    UNCHECKABLE = "uncheckable"


class RuleEvaluation(RuntimeModel):
    """The outcome of evaluating one declared rule against one artifact."""

    name: str = Field(min_length=1)
    #: The rule text verbatim from the benchmark, so a finding can quote what
    #: was actually asked rather than a paraphrase of it.
    rule: str = Field(min_length=1)
    outcome: RuleOutcome
    detail: str = ""


class ArtifactValidation(RuntimeModel):
    """What checking an artifact established, and what it could not.

    ``validated`` on an artifact record is a single bit, which is all scoring
    consumes.  This is the audit trail behind that bit: an artifact marked valid
    with three rules unevaluated is a different claim from one that passed all
    three, and a paper reporting the first as the second would be wrong.
    """

    exists: bool = False
    #: ``True`` when the file's digest still matches the one recorded when it was
    #: produced, ``False`` when it has since changed, and ``None`` when no digest
    #: was recorded to compare against.  ``None`` is a harness gap rather than a
    #: verdict, so it does not block validity -- the digest is computed by the
    #: harness, never supplied by the agent, so its absence is not something an
    #: agent can arrange.
    checksum_verified: bool | None = None
    rules: list[RuleEvaluation] = Field(default_factory=list)
    #: Why any part of this check could not be completed.
    limitations: list[str] = Field(default_factory=list)

    def with_outcome(self, outcome: RuleOutcome) -> list[RuleEvaluation]:
        """Return the rules that reached one outcome."""
        return [rule for rule in self.rules if rule.outcome is outcome]

    @property
    def is_valid(self) -> bool:
        """Whether the artifact earned ``validated=True``.

        An unevaluated rule does not block validity, for the same reason a
        structurally ineligible metric is excluded rather than scored zero:
        charging the agent for a check the harness could not run would make a
        missing optional dependency indistinguishable from bad science.  What
        keeps that honest is that the unevaluated rules are recorded here, so the
        bit is always auditable rather than merely optimistic.
        """
        return (
            self.exists
            and self.checksum_verified is not False
            and not self.with_outcome(RuleOutcome.FAILED)
        )


class ArtifactRecord(RuntimeModel):
    """A materialized or referenced scientific artifact produced in an episode."""

    artifact_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    format: str = Field(min_length=1)
    uri: str | None = None
    checksum: str | None = None
    #: Whether the artifact satisfied every rule that could be checked.  Set from
    #: :attr:`ArtifactValidation.is_valid` rather than by whoever produced the
    #: file, because a producer asserting its own output is valid is the claim
    #: this benchmark exists to verify instead of believe.
    validated: bool = False
    #: The evidence behind ``validated``.  ``None`` means no check has run, which
    #: is why ``validated`` defaults to ``False``: unvalidated, not invalid.
    validation: ArtifactValidation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KeyDelta(RuntimeModel):
    """Names that appeared, vanished, or changed value between two observations.

    ``unproven`` is the honesty field.  Some identities are established by a
    complete digest and some by a cheaper proxy, so a verdict about a name in
    that second group is *evidence* rather than proof.  Listing those names
    separately lets a reader distinguish "this did not change" from "nothing
    suggests this changed", which are different claims to put in a paper.
    """

    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    unproven: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether nothing observable happened to this namespace."""
        return not (self.added or self.removed or self.changed)

    @property
    def touched(self) -> list[str]:
        """Every name this delta says something happened to."""
        return sorted({*self.added, *self.removed, *self.changed})


#: Namespaces a ``StateDelta`` can report on, and therefore the vocabulary of
#: ``StateDelta.unobserved``.  Kept as a named constant because a producer and a
#: consumer that disagree about these spellings fail *silently*: a namespace
#: misspelled in ``unobserved`` reads as observed, turning "we could not look"
#: into "nothing happened", which is the one confusion this whole layer exists
#: to prevent.
STATE_NAMESPACES = frozenset({"obs", "var", "obsm", "layers", "files", "matrix"})


class StateDelta(RuntimeModel):
    """What observing state before and after an execution says it did.

    Once the agent runs its own code, the harness cannot learn what a step did
    by reading the step -- it can only compare what it observed before with what
    it observes after.  This model is that comparison, and it is deliberately
    independent of anything the agent said: it is the evidence a claim gets
    checked against, so it must not be derived from the claim.

    Absent fields mean "not observed", never "did not change".  Because an empty
    ``KeyDelta`` is also the correct result for a step that changed nothing, the
    two cases cannot be told apart by inspection -- which is what ``unobserved``
    is for.  A producer that could not see a namespace must name it there, and a
    consumer must ask :meth:`is_observed` before reading an emptiness as a fact.
    """

    n_obs_before: int | None = Field(default=None, ge=0)
    n_obs_after: int | None = Field(default=None, ge=0)
    n_vars_before: int | None = Field(default=None, ge=0)
    n_vars_after: int | None = Field(default=None, ge=0)
    obs: KeyDelta = Field(default_factory=KeyDelta)
    var: KeyDelta = Field(default_factory=KeyDelta)
    obsm: KeyDelta = Field(default_factory=KeyDelta)
    layers: KeyDelta = Field(default_factory=KeyDelta)
    files: KeyDelta = Field(default_factory=KeyDelta)
    #: Whether the expression matrix itself changed. ``None`` when unobserved.
    matrix_changed: bool | None = None
    #: Whether the set or order of cell barcodes changed. Tracked separately
    #: from the counts because a substitution or a reordering leaves ``n_obs``
    #: untouched while still breaking the barcode join that scoring rejoins the
    #: hidden reference on.
    obs_names_changed: bool | None = None
    #: Whether the set or order of gene names changed.
    var_names_changed: bool | None = None
    #: Namespaces from :data:`STATE_NAMESPACES` this delta could not look at.
    #: Their fields are therefore empty for lack of evidence, not for lack of
    #: change, and no verdict may be drawn from them.
    unobserved: list[str] = Field(default_factory=list)
    #: Why this delta is less than complete: a sampled digest, an unreadable
    #: file, a dataset the executor could not open. Recorded rather than
    #: silently narrowing what the delta appears to cover.
    limitations: list[str] = Field(default_factory=list)

    def is_observed(self, namespace: str) -> bool:
        """Whether this delta actually looked at ``namespace``."""
        return namespace not in self.unobserved

    @property
    def is_empty(self) -> bool:
        """Whether observation detected no change at all."""
        return (
            self.obs.is_empty
            and self.var.is_empty
            and self.obsm.is_empty
            and self.layers.is_empty
            and self.files.is_empty
            and not self.matrix_changed
            and not self.obs_names_changed
            and not self.var_names_changed
            and self.cells_removed in (None, 0)
            and self.genes_removed in (None, 0)
        )

    @property
    def cells_removed(self) -> int | None:
        """Cells the step dropped, negative when it added them."""
        if self.n_obs_before is None or self.n_obs_after is None:
            return None
        return self.n_obs_before - self.n_obs_after

    @property
    def genes_removed(self) -> int | None:
        """Genes the step dropped, negative when it added them."""
        if self.n_vars_before is None or self.n_vars_after is None:
            return None
        return self.n_vars_before - self.n_vars_after

    def summary(self) -> dict[str, Any]:
        """Return a compact description for the episode trace."""
        return {
            "cells_removed": self.cells_removed,
            "genes_removed": self.genes_removed,
            "obs_columns": self.obs.touched,
            "var_columns": self.var.touched,
            "obsm_keys": self.obsm.touched,
            "layers": self.layers.touched,
            "files": self.files.touched,
            "matrix_changed": self.matrix_changed,
            "unobserved": self.unobserved,
            "limitations": self.limitations,
        }


class ActionExecutionResult(RuntimeModel):
    """Executor response for a validated action intent.

    Executors may be backed by a local process, container, remote service, or
    simulator.  The environment only consumes this stable result contract.
    """

    intent_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    status: ActionStatus
    #: Finer-grained reason the execution ended. Optional because executors that
    #: cannot distinguish a timeout from a crash must not claim they can; a
    #: persisted run recorded before this field existed loads as ``None``.
    execution_status: ExecutionStatus | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    #: What the executor observed the execution do, from comparing state before
    #: and after. ``None`` from an executor that cannot observe state -- which
    #: must read as "unknown", not as "nothing happened".
    observed_state_delta: StateDelta | None = None
    error: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)


class ActionRecord(RuntimeModel):
    """Immutable-in-practice history entry pairing an intent with its result."""

    step: int = Field(ge=1)
    intent: ActionIntent
    result: ActionExecutionResult
    recorded_at: datetime = Field(default_factory=utc_now)


class RewardRecord(RuntimeModel):
    """Reward emitted by a pluggable evaluator for one episode step."""

    value: float
    strategy_id: str | None = None
    metric_values: dict[str, float] = Field(default_factory=dict)
    step: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeEvent(RuntimeModel):
    """Append-only event used for debugging, audit, and deterministic replay."""

    event_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    episode_id: str = Field(min_length=1)
    event_type: EventType
    step: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class EpisodeState(RuntimeModel):
    """Current derived state of an episode."""

    episode_id: str
    benchmark_id: str
    benchmark_version: str
    task_id: str
    seed: int
    status: EpisodeStatus = EpisodeStatus.CREATED
    current_step: int = Field(default=0, ge=0)
    dataset_id: str | None = None
    observations: dict[str, Observation] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
    actions: list[ActionRecord] = Field(default_factory=list)
    rewards: list[RewardRecord] = Field(default_factory=list)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    specification_digest: str


class EpisodeSnapshot(RuntimeModel):
    """Read-only transport snapshot passed to ports and returned to callers."""

    state: EpisodeState
    events: list[EpisodeEvent] = Field(default_factory=list)


def agent_visible_state(state: EpisodeState) -> EpisodeState:
    """Return the episode state with every evaluator-only observation removed.

    Observations reach an agent through *two* paths in this model, and redacting
    one is worth nothing while the other stands. The obvious one is
    :attr:`EpisodeState.observations`. The second is the action history: every
    :class:`ActionRecord` embeds its executor's :class:`ActionExecutionResult`,
    which carries the observations that execution produced -- so the workspace
    executor's isolation report, naming each sandbox control this host failed to
    enforce, is reachable at ``actions[i].result.observations`` even after the
    top-level map is filtered.

    Redaction happens on the typed model rather than on a serialized dict so that
    adding an observation-bearing field to any of these models is a type error
    here instead of a silent leak wherever it is dumped.
    """
    # Reward records are evaluator-owned. In the scientific loop they may carry
    # reference-derived metrics and stage progress, neither of which is visible
    # to a real scientist working from the supplied data. Returning them here
    # leaked the answer channel through ``AgentObservation.state.rewards`` even
    # though the same values were correctly marked absent from observations.
    return state.model_copy(
        update={
            "observations": agent_visible_observations(state.observations),
            "rewards": [],
            "actions": [
                record.model_copy(
                    update={
                        "result": record.result.model_copy(
                            update={
                                "observations": [
                                    observation
                                    for observation in record.result.observations
                                    if observation.visible_to_agent
                                ]
                            }
                        )
                    }
                )
                for record in state.actions
            ],
        }
    )


class ActionValidationResult(RuntimeModel):
    """Structured result of validating an action intent."""

    valid: bool
    errors: list[str] = Field(default_factory=list)


class EnvironmentStep(RuntimeModel):
    """Result returned by one environment interaction."""

    episode_id: str
    accepted: bool
    validation: ActionValidationResult
    execution: ActionExecutionResult | None = None
    reward: RewardRecord | None = None
    observation: EpisodeSnapshot


__all__ = [
    "STATE_NAMESPACES",
    "ActionExecutionResult",
    "ActionIntent",
    "ActionRecord",
    "ActionStatus",
    "ActionValidationResult",
    "ArtifactRecord",
    "ArtifactValidation",
    "EnvironmentStep",
    "EpisodeEvent",
    "EpisodeSnapshot",
    "EpisodeState",
    "EpisodeStatus",
    "EventType",
    "ExecutionStatus",
    "KeyDelta",
    "Observation",
    "ResourceUsage",
    "RewardRecord",
    "RuleEvaluation",
    "RuleOutcome",
    "StateDelta",
    "agent_visible_observations",
    "agent_visible_state",
]
