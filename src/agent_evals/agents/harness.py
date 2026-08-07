"""Framework-neutral agent adapter and run orchestration contracts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import ClassVar, Protocol, runtime_checkable
from uuid import uuid4

from agent_evals.agents.trajectory import (
    AgentConfiguration,
    AgentFailure,
    AgentManifest,
    AgentModelInfo,
    AgentRun,
    EstimatedCost,
    FailureKind,
    NormalizedTrajectory,
    NormalizedTrajectoryEvent,
    RawTraceEvent,
    RunTerminationStatus,
    TokenUsage,
    decision_cascade_from_episode,
)
from agent_evals.benchmarks.schema import TaskSpecification
from agent_evals.environment.models import (
    EpisodeEvent,
    EpisodeSnapshot,
    EpisodeStatus,
    EventType,
)
from agent_evals.environment.runtime import ScientificEnvironment


@runtime_checkable
class AgentAdapter(Protocol):
    """Narrow contract implemented by any external agent framework."""

    adapter_name: str
    adapter_version: str

    async def run(
        self,
        task: TaskSpecification,
        environment: ScientificEnvironment,
        configuration: AgentConfiguration,
    ) -> AgentRun:
        """Run one task and return a normalized agent execution."""


@runtime_checkable
class TraceNormalizer(Protocol):
    """Convert framework-specific raw events into normalized events."""

    def normalize(
        self,
        raw_events: Sequence[RawTraceEvent],
        *,
        run_id: str,
        episode_id: str,
    ) -> list[NormalizedTrajectoryEvent]:
        """Normalize raw events while preserving their payloads and order."""


class DefaultTraceNormalizer:
    """Conservative event mapper that never infers private reasoning."""

    _EVENT_MAP: ClassVar[dict[str, EventType]] = {
        "message": EventType.AGENT_MESSAGE,
        "assistant_message": EventType.AGENT_MESSAGE,
        "observation": EventType.OBSERVATION_RECEIVED,
        "tool_call": EventType.TOOL_CALL,
        "tool_result": EventType.TOOL_RESULT,
        "action": EventType.ACTION_PROPOSED,
        "action_proposed": EventType.ACTION_PROPOSED,
        "action_executed": EventType.ACTION_EXECUTED,
        "artifact_created": EventType.ARTIFACT_CREATED,
        "artifact_modified": EventType.ARTIFACT_MODIFIED,
        "command": EventType.COMMAND_EXECUTED,
        "command_executed": EventType.COMMAND_EXECUTED,
        "error": EventType.AGENT_ERROR,
    }

    def normalize(
        self,
        raw_events: Sequence[RawTraceEvent],
        *,
        run_id: str,
        episode_id: str,
    ) -> list[NormalizedTrajectoryEvent]:
        """Map known event names and conservatively classify unknown events."""
        normalized: list[NormalizedTrajectoryEvent] = []
        for event in raw_events:
            event_type = self._EVENT_MAP.get(event.event_type.lower(), EventType.AGENT_MESSAGE)
            normalized.append(
                NormalizedTrajectoryEvent(
                    event_id=event.event_id,
                    run_id=run_id,
                    episode_id=episode_id,
                    sequence=event.sequence,
                    timestamp=event.timestamp,
                    event_type=event_type,
                    source=event.source,
                    payload=event.payload,
                    parent_event_id=event.parent_event_id,
                )
            )
        return normalized


class AgentHarness:
    """Run adapters and preserve valid partial results on adapter failure."""

    def __init__(self, *, normalizer: TraceNormalizer | None = None) -> None:
        self.normalizer = normalizer or DefaultTraceNormalizer()

    async def run(
        self,
        adapter: AgentAdapter,
        environment: ScientificEnvironment,
        configuration: AgentConfiguration,
    ) -> AgentRun:
        """Execute one adapter and normalize unexpected harness failures."""
        started_at = datetime.now(UTC)
        try:
            result = await adapter.run(environment.task, environment, configuration)
            return result
        except Exception as error:
            snapshot = await self._failure_snapshot(environment, configuration)
            finished_at = datetime.now(UTC)
            run_id = str(uuid4())
            failure = AgentFailure(kind=FailureKind.AGENT_ERROR, message=str(error))
            return build_agent_run(
                adapter_name=getattr(adapter, "adapter_name", type(adapter).__name__),
                adapter_version=getattr(adapter, "adapter_version", "unknown"),
                configuration=configuration,
                task=environment.task,
                snapshot=snapshot,
                raw_events=[],
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                termination_status=RunTerminationStatus.FAILED,
                termination_reason=str(error),
                failures=[failure],
                normalizer=self.normalizer,
            )

    async def _failure_snapshot(
        self,
        environment: ScientificEnvironment,
        configuration: AgentConfiguration,
    ) -> EpisodeSnapshot:
        """Ensure even pre-reset adapter failures have a persisted episode."""
        if environment.episode is None:
            await environment.reset(
                seed=configuration.seed,
                dataset_id=environment.task.datasets[0] if environment.task.datasets else None,
            )
        snapshot = await environment.observe()
        if environment.episode is not None:
            try:
                snapshot = environment.terminate(status=EpisodeStatus.FAILED)
            except Exception:
                pass
        return snapshot


def episode_events_to_normalized(
    events: Sequence[EpisodeEvent],
    *,
    run_id: str,
    episode_id: str,
    sequence_offset: int,
) -> list[NormalizedTrajectoryEvent]:
    """Represent existing episode events in the same normalized event stream."""
    return [
        NormalizedTrajectoryEvent(
            event_id=event.event_id,
            run_id=run_id,
            episode_id=episode_id,
            sequence=sequence_offset + index,
            timestamp=event.timestamp,
            event_type=event.event_type,
            source="scientific_environment",
            payload=event.payload,
        )
        for index, event in enumerate(events)
    ]


def build_agent_run(
    *,
    adapter_name: str,
    adapter_version: str,
    configuration: AgentConfiguration,
    task: TaskSpecification,
    snapshot: EpisodeSnapshot,
    raw_events: Sequence[RawTraceEvent],
    run_id: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    termination_status: RunTerminationStatus = RunTerminationStatus.COMPLETED,
    termination_reason: str | None = None,
    failures: list[AgentFailure] | None = None,
    normalizer: TraceNormalizer | None = None,
    token_usage: TokenUsage | None = None,
    estimated_cost: EstimatedCost | None = None,
    metadata: dict[str, object] | None = None,
    manifest: AgentManifest | None = None,
) -> AgentRun:
    """Build a complete run from adapter events and the final episode snapshot."""
    resolved_run_id = run_id or str(uuid4())
    resolved_started = started_at or datetime.now(UTC)
    resolved_finished = finished_at or datetime.now(UTC)
    trace_normalizer = normalizer or DefaultTraceNormalizer()
    normalized = trace_normalizer.normalize(
        raw_events,
        run_id=resolved_run_id,
        episode_id=snapshot.state.episode_id,
    )
    normalized.extend(
        episode_events_to_normalized(
            snapshot.events,
            run_id=resolved_run_id,
            episode_id=snapshot.state.episode_id,
            sequence_offset=len(normalized),
        )
    )
    normalized.sort(key=lambda event: (event.sequence, event.timestamp))
    trajectory = NormalizedTrajectory(
        run_id=resolved_run_id,
        episode_id=snapshot.state.episode_id,
        events=normalized,
        decisions=decision_cascade_from_episode(snapshot),
    )
    return AgentRun(
        run_id=resolved_run_id,
        agent_id=configuration.agent_type,
        configuration=configuration,
        manifest=manifest
        or AgentManifest(
            name=configuration.agent_type,
            type=configuration.agent_type,
            model=AgentModelInfo(provider=configuration.provider, name=configuration.model),
            temperature=configuration.temperature,
        ),
        model=configuration.model,
        provider=configuration.provider,
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        benchmark_id=snapshot.state.benchmark_id,
        task_id=task.id,
        dataset_id=snapshot.state.dataset_id,
        episode_id=snapshot.state.episode_id,
        started_at=resolved_started,
        finished_at=resolved_finished,
        termination_status=termination_status,
        termination_reason=termination_reason,
        token_usage=token_usage,
        estimated_cost=estimated_cost,
        wall_clock_seconds=max(0.0, (resolved_finished - resolved_started).total_seconds()),
        step_count=snapshot.state.current_step,
        tool_call_count=sum(1 for event in normalized if event.event_type == EventType.TOOL_CALL),
        raw_events=list(raw_events),
        trajectory=trajectory,
        generated_artifacts=list(snapshot.state.artifacts.values()),
        final_environment_state=snapshot,
        failures=failures or [],
        metadata=metadata or {},
    )


__all__ = [
    "AgentAdapter",
    "AgentHarness",
    "DefaultTraceNormalizer",
    "TraceNormalizer",
    "build_agent_run",
    "episode_events_to_normalized",
]
