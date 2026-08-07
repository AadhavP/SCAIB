"""Runtime orchestration and compatibility bridge to the existing harness."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.events import AgentEventType, AgentTrajectory
from agent_evals.agents.runtime.protocol import (
    AgentAction,
    AgentContext,
    AgentObservation,
    FinalSubmission,
)
from agent_evals.agents.tools import ToolExecutor
from agent_evals.benchmarks.schema import TaskSpecification
from agent_evals.environment.models import ActionStatus, EpisodeSnapshot, EpisodeStatus
from agent_evals.environment.runtime import ScientificEnvironment


class RuntimeRun(BaseModel):
    """Serializable universal-runtime result before legacy harness conversion."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    benchmark_id: str
    task_id: str
    started_at: datetime
    finished_at: datetime
    termination_status: str
    termination_reason: str | None = None
    step_count: int = Field(default=0, ge=0)
    trajectory: AgentTrajectory
    final_submission: FinalSubmission | None = None
    final_snapshot: EpisodeSnapshot


class AgentRuntimeManager:
    """Drive any runtime through the typed scientific environment."""

    def __init__(
        self,
        *,
        terminal_actions: set[str] | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self.terminal_actions = terminal_actions or {
            "terminate",
            "final_submission",
            "finish",
            "done",
        }
        self.tool_executor = tool_executor

    async def run(  # noqa: C901
        self,
        runtime: AgentRuntime,
        environment: ScientificEnvironment,
        context: AgentContext,
        *,
        seed: int = 0,
        dataset_id: str | None = None,
        max_steps: int | None = None,
    ) -> RuntimeRun:
        """Run a universal runtime while preserving partial trajectories."""
        started_at = datetime.now(UTC)
        initial = await environment.reset(seed=seed, dataset_id=dataset_id)
        trajectory = AgentTrajectory()
        observation = _observation_from_snapshot(initial, environment.task)
        trajectory.observations.append(observation)
        trajectory.record(AgentEventType.OBSERVATION, observation.model_dump(mode="json"))
        session = await runtime.initialize(context)
        status = "completed"
        reason = "agent runtime completed"
        submission: FinalSubmission | None = None
        steps = 0
        try:
            while max_steps is None or steps < max_steps:
                raw_action = await runtime.act(session, observation)
                action = AgentAction.model_validate(raw_action)
                trajectory.actions.append(action)
                action_event = trajectory.record(
                    AgentEventType.ACTION,
                    action.model_dump(mode="json"),
                )
                if action.reasoning_metadata.get("summary"):
                    trajectory.record(
                        AgentEventType.REASONING_SUMMARY,
                        {"summary": action.reasoning_metadata["summary"]},
                        parent_event_id=action_event.event_id,
                    )
                if self.tool_executor is not None and _is_registered_tool(self.tool_executor, action.action_type):
                    trajectory.record(
                        AgentEventType.TOOL_CALL,
                        {"tool": action.action_type, "arguments": action.parameters},
                        parent_event_id=action_event.event_id,
                    )
                    try:
                        tool_result = await self.tool_executor.execute(
                            action.action_type,
                            action.parameters,
                            context=session,
                        )
                        trajectory.record(
                            AgentEventType.TOOL_RESULT,
                            {"tool": action.action_type, "result": tool_result},
                            parent_event_id=action_event.event_id,
                        )
                        if isinstance(tool_result, AgentAction) or (
                            isinstance(tool_result, dict) and "action_type" in tool_result
                        ):
                            action = AgentAction.model_validate(tool_result)
                        else:
                            steps += 1
                            continue
                    except Exception as error:
                        trajectory.record(
                            AgentEventType.FAILURE,
                            {"tool": action.action_type, "error": str(error)},
                            parent_event_id=action_event.event_id,
                        )
                        steps += 1
                        continue
                if action.action_type in self.terminal_actions:
                    submission = await runtime.terminate(session, observation)
                    trajectory.final_submission = submission
                    trajectory.record(
                        AgentEventType.FINAL_SUBMISSION,
                        submission.model_dump(mode="json"),
                        parent_event_id=action_event.event_id,
                    )
                    break
                intent = _action_to_intent(action)
                result = await environment.step(intent)
                trajectory.record(
                    AgentEventType.ENVIRONMENT_RESPONSE,
                    result.model_dump(mode="json"),
                    parent_event_id=action_event.event_id,
                )
                if not result.accepted or result.execution is None or result.execution.status != ActionStatus.SUCCEEDED:
                    trajectory.record(
                        AgentEventType.FAILURE,
                        {
                            "action_type": action.action_type,
                            "error": result.execution.error if result.execution else result.validation.errors,
                        },
                        parent_event_id=action_event.event_id,
                    )
                steps += 1
                observation = _observation_from_snapshot(result.observation, environment.task)
                trajectory.observations.append(observation)
                trajectory.record(
                    AgentEventType.OBSERVATION,
                    observation.model_dump(mode="json"),
                    parent_event_id=action_event.event_id,
                )
            else:
                status = "timeout"
                reason = "maximum runtime steps reached"
            if submission is None:
                submission = await runtime.terminate(session, observation)
                trajectory.final_submission = submission
                trajectory.record(
                    AgentEventType.FINAL_SUBMISSION,
                    submission.model_dump(mode="json"),
                )
        except Exception as error:
            status = "failed"
            reason = str(error)
            trajectory.record(AgentEventType.FAILURE, {"error": str(error)})
            try:
                submission = await runtime.terminate(session, observation)
                trajectory.final_submission = submission
            except Exception as termination_error:
                trajectory.record(AgentEventType.FAILURE, {"error": str(termination_error)})
        if environment.episode is not None and environment.episode.status not in {
            EpisodeStatus.COMPLETED,
            EpisodeStatus.FAILED,
            EpisodeStatus.CANCELLED,
        }:
            environment.terminate(
                status=EpisodeStatus.COMPLETED if status == "completed" else EpisodeStatus.FAILED,
                reason=reason,
            )
        final_snapshot = await environment.observe()
        finished_at = datetime.now(UTC)
        return RuntimeRun(
            agent_id=runtime.agent_id,
            benchmark_id=final_snapshot.state.benchmark_id,
            task_id=final_snapshot.state.task_id,
            started_at=started_at,
            finished_at=finished_at,
            termination_status=status,
            termination_reason=reason,
            step_count=steps,
            trajectory=trajectory,
            final_submission=submission,
            final_snapshot=final_snapshot,
        )


class RuntimeAgentAdapter:
    """Expose a universal runtime through the existing AgentAdapter protocol."""

    adapter_name = "universal-runtime"
    adapter_version = "2.0.0"

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def run(
        self,
        task: TaskSpecification,
        environment: ScientificEnvironment,
        configuration: Any,
    ) -> Any:
        """Run and convert a universal result into the normalized AgentRun model."""
        from agent_evals.agents.harness import build_agent_run
        from agent_evals.agents.trajectory import (
            AgentFailure,
            FailureKind,
            RawTraceEvent,
            RunTerminationStatus,
        )
        from agent_evals.agents.trajectory import (
            AgentManifest as LegacyAgentManifest,
        )
        from agent_evals.agents.trajectory import (
            AgentModelInfo as LegacyAgentModelInfo,
        )

        root = str(configuration.workspace.get("root", "."))
        constraints = _model_dump(environment.task.constraints)
        tools = configuration.tools.get("definitions", []) if isinstance(configuration.tools, dict) else []
        context = AgentContext(
            benchmark_id=environment.specification.metadata.id,
            task_id=task.id,
            workspace=root,
            tools=list(tools) if isinstance(tools, list) else [],
            constraints=constraints,
            metadata=configuration.metadata,
        )
        universal = await AgentRuntimeManager().run(
            self.runtime,
            environment,
            context,
            seed=configuration.seed,
            dataset_id=configuration.metadata.get("dataset_id")
            or (task.datasets[0] if task.datasets else None),
            max_steps=configuration.max_steps,
        )
        raw_events = [
            RawTraceEvent(
                event_id=event.event_id,
                source=self.runtime.agent_id,
                sequence=event.sequence,
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                payload=event.payload,
                parent_event_id=event.parent_event_id,
            )
            for event in universal.trajectory.events
        ]
        status = {
            "completed": RunTerminationStatus.COMPLETED,
            "timeout": RunTerminationStatus.TIMEOUT,
            "failed": RunTerminationStatus.FAILED,
        }.get(universal.termination_status, RunTerminationStatus.FAILED)
        failures = (
            [AgentFailure(kind=FailureKind.AGENT_ERROR, message=universal.termination_reason or "runtime failed")]
            if status != RunTerminationStatus.COMPLETED
            else []
        )
        return build_agent_run(
            adapter_name=self.adapter_name,
            adapter_version=self.adapter_version,
            configuration=configuration,
            task=task,
            snapshot=universal.final_snapshot,
            raw_events=raw_events,
            run_id=universal.run_id,
            started_at=universal.started_at,
            finished_at=universal.finished_at,
            termination_status=status,
            termination_reason=universal.termination_reason,
            failures=failures,
            manifest=LegacyAgentManifest(
                name=self.runtime.manifest.name,
                type=self.runtime.manifest.type,
                model=LegacyAgentModelInfo(
                    provider=self.runtime.manifest.model.provider,
                    name=self.runtime.manifest.model.name,
                ),
                capabilities=self.runtime.manifest.capabilities,
                temperature=self.runtime.manifest.temperature,
                context_window=self.runtime.manifest.context_window,
                metadata=self.runtime.manifest.metadata,
            ),
            metadata={
                "agent_manifest": self.runtime.manifest.model_dump(mode="json"),
                "final_submission": universal.final_submission.model_dump(mode="json")
                if universal.final_submission is not None
                else None,
            },
        )


def _observation_from_snapshot(snapshot: EpisodeSnapshot, task: TaskSpecification) -> AgentObservation:
    """Project an environment snapshot into the universal observation shape."""
    return AgentObservation(
        state=snapshot.state.model_dump(mode="json"),
        available_actions=list(task.allowed_actions),
        artifacts=[artifact.model_dump(mode="json") for artifact in snapshot.state.artifacts.values()],
        metadata={
            "episode_id": snapshot.state.episode_id,
            "step": snapshot.state.current_step,
            "observations": {
                key: value.model_dump(mode="json") for key, value in snapshot.state.observations.items()
            },
        },
    )


def _action_to_intent(action: AgentAction) -> Any:
    """Map a universal action to the existing typed environment request."""
    from agent_evals.environment.models import ActionIntent

    parameters = dict(action.parameters)
    action_id = str(parameters.pop("action_id", action.action_type))
    rationale = action.reasoning_metadata.get("explanation") or action.reasoning_metadata.get("summary")
    metadata = {
        "runtime_action_type": action.action_type,
        **action.reasoning_metadata,
    }
    return ActionIntent(
        action_id=action_id,
        parameters=parameters,
        rationale=str(rationale) if rationale is not None else None,
        metadata=metadata,
    )


def _model_dump(value: Any) -> dict[str, Any]:
    """Convert optional Pydantic constraints to a plain dictionary."""
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    return dump(mode="json") if callable(dump) else {}


def _is_registered_tool(executor: ToolExecutor, name: str) -> bool:
    """Check tool membership without turning ordinary environment actions into tools."""
    try:
        executor.registry.get(name)
    except KeyError:
        return False
    return True


__all__ = ["AgentRuntimeManager", "RuntimeAgentAdapter", "RuntimeRun"]
