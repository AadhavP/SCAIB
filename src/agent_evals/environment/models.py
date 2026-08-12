"""Typed runtime values exchanged by the scientific environment.

These models are the runtime counterpart to the declarative benchmark
specification.  They describe what happened during an episode without
implementing any scientific tool or sandbox backend.
"""

from __future__ import annotations

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
    """One agent-visible value produced by the environment."""

    observation_id: str = Field(min_length=1)
    value: Any
    source: str = Field(min_length=1)
    step: int = Field(default=0, ge=0)
    visible_to_agent: bool = True
    produced_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactRecord(RuntimeModel):
    """A materialized or referenced scientific artifact produced in an episode."""

    artifact_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    format: str = Field(min_length=1)
    uri: str | None = None
    checksum: str | None = None
    validated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    "ActionExecutionResult",
    "ActionIntent",
    "ActionRecord",
    "ActionStatus",
    "ActionValidationResult",
    "ArtifactRecord",
    "EnvironmentStep",
    "EpisodeEvent",
    "EpisodeSnapshot",
    "EpisodeState",
    "EpisodeStatus",
    "EventType",
    "ExecutionStatus",
    "Observation",
    "ResourceUsage",
    "RewardRecord",
]
