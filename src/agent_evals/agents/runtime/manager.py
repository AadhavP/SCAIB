"""Runtime orchestration and compatibility bridge to the existing harness."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
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
    AgentPlan,
    FinalSubmission,
)
from agent_evals.agents.tools import ToolExecutor
from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
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
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> None:
        self.terminal_actions = terminal_actions or {
            "terminate",
            "final_submission",
            "finish",
            "done",
        }
        self.tool_executor = tool_executor
        self.event_callback = event_callback

    async def _emit(self, event: dict[str, Any]) -> None:
        """Publish runtime lifecycle events without coupling the manager to the API."""
        if self.event_callback is None:
            return
        result = self.event_callback(event)
        if inspect.isawaitable(result):
            await result

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
        await self._emit(
            {
                "type": "agent_planning",
                "step": 0,
                "message": "Asking the agent for an overall scientific plan.",
            }
        )
        try:
            plan = await runtime.plan(context, observation)
            plan_source = "agent"
        except Exception as error:
            plan = None
            plan_source = f"benchmark_fallback: {type(error).__name__}"
        if plan is None:
            metadata = observation.metadata
            plan = AgentPlan(
                goal=str(metadata.get("goal", "Complete the scientific benchmark objective.")),
                steps=[
                    "Inspect the current dataset and quality signals.",
                    "Choose the next evidence-producing benchmark action.",
                    "Reassess the plan after each result and stop when success criteria are met.",
                ],
                success_criteria=[
                    str(item.get("description", item.get("condition", "")))
                    for item in metadata.get("scenario", {}).get("success_criteria", [])
                    if isinstance(item, dict)
                ],
                adaptation_policy="After every result, keep, revise, or end the plan based on the new evidence.",
            )
        session.state["plan"] = plan.model_dump(mode="json")
        observation = observation.model_copy(
            update={
                "metadata": {
                    **observation.metadata,
                    "active_plan": plan.model_dump(mode="json"),
                    "plan_review": "After each result, decide whether to keep, revise, or end this plan.",
                }
            }
        )
        await self._emit(
            {
                "type": "agent_plan",
                "step": 0,
                "message": "Initial scientific plan is ready.",
                "plan": plan.model_dump(mode="json"),
                "source": plan_source,
            }
        )
        trajectory.record(AgentEventType.PLAN, plan.model_dump(mode="json"))
        status = "completed"
        reason = "agent runtime completed"
        submission: FinalSubmission | None = None
        steps = 0
        try:
            while max_steps is None or steps < max_steps:
                current_step = steps + 1
                await self._emit(
                    {
                        "type": "agent_prompt",
                        "step": current_step,
                        "message": "Environment observation sent to the agent.",
                        "observation": observation.model_dump(mode="json"),
                    }
                )
                await self._emit(
                    {
                        "type": "agent_waiting",
                        "step": current_step,
                        "message": "Waiting for the agent to return its next action.",
                    }
                )
                raw_action = await runtime.act(session, observation)
                action = AgentAction.model_validate(raw_action)
                await self._emit(
                    {
                        "type": "agent_response",
                        "step": current_step,
                        "message": "Agent returned a structured action.",
                        "action_type": action.action_type,
                        "parameters": action.parameters,
                        "reasoning_metadata": {
                            key: value
                            for key, value in action.reasoning_metadata.items()
                            if key in {"summary", "explanation"}
                        },
                    }
                )
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
                    # A terminal action is a claim of completion, not proof of it.
                    # Accepting it unverified lets an agent score "completed" by
                    # finishing immediately without producing any artifact.
                    missing = _missing_required_artifacts(environment)
                    submission = await runtime.terminate(session, observation)
                    trajectory.final_submission = submission
                    trajectory.record(
                        AgentEventType.FINAL_SUBMISSION,
                        submission.model_dump(mode="json"),
                        parent_event_id=action_event.event_id,
                    )
                    if missing:
                        status = "incomplete"
                        reason = (
                            "agent submitted a terminal action while required "
                            f"benchmark artifacts were missing: {sorted(missing)}"
                        )
                        trajectory.record(
                            AgentEventType.FAILURE,
                            {
                                "action_type": action.action_type,
                                "error": reason,
                                "missing_artifacts": sorted(missing),
                            },
                            parent_event_id=action_event.event_id,
                        )
                    break
                intent = _action_to_intent(action, environment.specification)
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
                next_observation = _observation_from_snapshot(result.observation, environment.task)
                observation = next_observation.model_copy(
                    update={
                        "metadata": {
                            **next_observation.metadata,
                            "active_plan": plan.model_dump(mode="json"),
                            "plan_review": "After this result, decide whether to keep, revise, or end the plan.",
                        }
                    }
                )
                trajectory.observations.append(observation)
                trajectory.record(
                    AgentEventType.OBSERVATION,
                    observation.model_dump(mode="json"),
                    parent_event_id=action_event.event_id,
                )
            else:
                produced_artifacts = set(environment.episode.snapshot().state.artifacts) if environment.episode else set()
                required_artifacts = set(environment.task.artifacts)
                if required_artifacts and required_artifacts.issubset(produced_artifacts):
                    status = "completed"
                    reason = "required benchmark artifacts were produced within the step budget"
                else:
                    status = "timeout"
                    reason = "maximum runtime steps reached before the benchmark goal was satisfied"
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

    def __init__(
        self,
        runtime: AgentRuntime,
        event_callback: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.event_callback = event_callback

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
        universal = await AgentRuntimeManager(event_callback=self.event_callback).run(
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
    """Project an environment snapshot and scientific goal into agent-visible context."""
    termination = [
        {
            "name": condition.name,
            "description": condition.description,
            "condition": condition.condition,
        }
        for condition in task.termination
    ]
    return AgentObservation(
        state=snapshot.state.model_dump(mode="json"),
        available_actions=list(task.allowed_actions),
        artifacts=[artifact.model_dump(mode="json") for artifact in snapshot.state.artifacts.values()],
        metadata={
            "episode_id": snapshot.state.episode_id,
            "step": snapshot.state.current_step,
            "scenario": {
                "name": task.name,
                "objective": task.objective,
                "description": task.description,
                "success_criteria": termination,
                "required_artifacts": list(task.artifacts),
                "required_metrics": list(task.metrics),
            },
            "goal": task.objective,
            "observations": {
                key: value.model_dump(mode="json") for key, value in snapshot.state.observations.items()
            },
        },
    )


def _action_to_intent(
    action: AgentAction,
    specification: BenchmarkSpecification,
) -> Any:
    """Map a universal action to the typed request expected by the environment.

    Universal runtimes do not otherwise see the benchmark's artifact contract.
    Carrying it on the intent lets scientific executors translate their native
    artifact names (for example ``normalized_anndata``) to the benchmark's
    stable IDs (for example ``normalized-anndata``) before environment output
    validation runs.
    """
    from agent_evals.environment.models import ActionIntent

    parameters = dict(action.parameters)
    action_id = str(parameters.pop("action_id", action.action_type))
    declared_action = next(
        (item for item in specification.actions if item.id == action_id),
        None,
    )
    rationale = action.reasoning_metadata.get("explanation") or action.reasoning_metadata.get("summary")
    metadata = {
        "runtime_action_type": action.action_type,
        **action.reasoning_metadata,
    }
    if declared_action is not None:
        metadata.update(
            {
                "expected_inputs": list(declared_action.required_inputs),
                "expected_outputs": list(declared_action.expected_outputs),
            }
        )
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


def _missing_required_artifacts(environment: ScientificEnvironment) -> set[str]:
    """Return the task's required artifact IDs that have not been produced."""
    required = set(environment.task.artifacts)
    if not required:
        return set()
    produced = (
        set(environment.episode.snapshot().state.artifacts)
        if environment.episode is not None
        else set()
    )
    return required - produced


def _is_registered_tool(executor: ToolExecutor, name: str) -> bool:
    """Check tool membership without turning ordinary environment actions into tools."""
    try:
        executor.registry.get(name)
    except KeyError:
        return False
    return True


__all__ = ["AgentRuntimeManager", "RuntimeAgentAdapter", "RuntimeRun"]
