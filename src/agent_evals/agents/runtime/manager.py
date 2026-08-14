"""Runtime orchestration and compatibility bridge to the existing harness."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from time import monotonic
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents.decisions import extract_decision
from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.events import AgentEventType, AgentTrajectory
from agent_evals.agents.runtime.protocol import (
    AgentAction,
    AgentContext,
    AgentObservation,
    AgentPlan,
    AgentSession,
    AgentUsage,
    FinalSubmission,
)
from agent_evals.agents.tools import ToolExecutor

# The rest of ``agents.trajectory`` is imported inside the conversion method to
# avoid the cycle through ``agents.harness``. These two are enums with no onward
# dependencies, and the status tables below have to be module-level constants to
# read as the mapping they are.
from agent_evals.agents.trajectory import (
    EstimatedCost,
    FailureKind,
    RunTerminationStatus,
    TokenUsage,
)
from agent_evals.benchmarks.agent_package import build_agent_task_package
from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.core.progress_keys import PROGRESS_DELTA_KEY
from agent_evals.environment.cutoff import (
    CutoffController,
    CutoffDecision,
    CutoffReason,
    CutoffReport,
    StepObservation,
    budget_from_specification,
)
from agent_evals.environment.models import (
    ActionStatus,
    EnvironmentStep,
    EpisodeSnapshot,
    EpisodeStatus,
    agent_visible_observations,
    agent_visible_state,
)
from agent_evals.environment.runtime import ScientificEnvironment


class RuntimeVerdict(StrEnum):
    """How the universal runtime loop decided a run ended.

    A vocabulary rather than four bare literals because the loop writes these and
    the adapter below reads them as lookup keys, and a spelling that only one side
    knows fails *silently*: the lookup falls back to ``FAILED`` and the verdict
    disappears without anything raising. That is how ``INCOMPLETE`` was being lost
    before it ever reached a result file.
    """

    #: The run met its declared artifact contract.
    COMPLETED = "completed"
    #: The step budget ran out with the contract unmet.
    TIMEOUT = "timeout"
    #: Something raised, in the harness or in the agent.
    FAILED = "failed"
    #: The agent claimed completion and the contract was not met. A clean run
    #: that stopped early, which is not the same finding as a broken one.
    INCOMPLETE = "incomplete"
    #: The controller stopped a run that was no longer making measurable
    #: progress, or was repeating itself. One verdict rather than one per
    #: mechanism -- ``CutoffReport.reason`` carries which fired, the same
    #: argument that put ``ExecutionStatus`` beside ``ActionStatus`` instead of
    #: widening it.
    STAGNATED = "stagnated"


class _RuntimeWallTimeout(TimeoutError):
    """A provider call exceeded the run's declared wall-clock budget."""


class RuntimeRun(BaseModel):
    """Serializable universal-runtime result before legacy harness conversion."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    benchmark_id: str
    task_id: str
    started_at: datetime
    finished_at: datetime
    termination_status: RuntimeVerdict
    termination_reason: str | None = None
    step_count: int = Field(default=0, ge=0)
    trajectory: AgentTrajectory
    final_submission: FinalSubmission | None = None
    final_snapshot: EpisodeSnapshot
    token_usage: TokenUsage | None = None
    estimated_cost: EstimatedCost | None = None
    #: Optional so persisted runs from before controller-owned termination still
    #: load. ``None`` means the run predates the cutoff layer, which is a
    #: different fact from a run whose budgets were all undeclared -- that one
    #: still gets a report, with every reason marked ``UNDECLARED``.
    cutoff: CutoffReport | None = None
    #: Secret-free HTTP/endpoint exchange provenance when the runtime exposes it.
    #: Native runtimes leave this empty; the boundary adapter records hashes and
    #: structural metadata rather than persisting prompts, credentials, or raw
    #: responses a second time.
    boundary_exchanges: list[dict[str, Any]] = Field(default_factory=list)


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
        run_id = str(uuid4())
        # Make the public scientific brief available from initialization onward,
        # not only after the first environment observation. This gives every
        # runtime the same contract and lets a custom agent inspect it before it
        # implements its first call.
        context = context.model_copy(
            update={
                "run_id": run_id,
                "task_package": build_agent_task_package(
                    environment.specification,
                    environment.task,
                )
            }
        )
        # Built here rather than accepted as a parameter so every existing call
        # site -- including ``RuntimeAgentAdapter``, which constructs this manager
        # itself -- gets controller-owned stopping without being changed.
        controller = CutoffController(
            budget_from_specification(
                environment.specification.cutoff,
                environment.specification.constraints,
                caller_max_steps=max_steps,
            )
        )
        # A monotonic origin, not ``started_at``: a wall-clock budget must not be
        # voided or doubled because the host adjusted its clock mid-run.
        run_origin = monotonic()
        input_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens: int | None = None
        cost_usd: float | None = None

        def consume_usage(usage: AgentUsage | None) -> None:
            """Accumulate per-request usage and expose only totals to cutoffs."""
            nonlocal input_tokens, output_tokens, total_tokens, cost_usd
            if usage is None:
                return
            if usage.input_tokens is not None:
                input_tokens = (input_tokens or 0) + usage.input_tokens
            if usage.output_tokens is not None:
                output_tokens = (output_tokens or 0) + usage.output_tokens
            reported_total = usage.total_tokens
            if reported_total is None and (
                usage.input_tokens is not None or usage.output_tokens is not None
            ):
                reported_total = (usage.input_tokens or 0) + (usage.output_tokens or 0)
            if reported_total is not None:
                total_tokens = (total_tokens or 0) + reported_total
            if usage.cost_usd is not None:
                cost_usd = (cost_usd or 0.0) + usage.cost_usd
            controller.observe_usage(total_tokens=total_tokens, cost_usd=cost_usd)

        def token_usage_model() -> TokenUsage | None:
            if all(value is None for value in (input_tokens, output_tokens, total_tokens)):
                return None
            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        def cost_model() -> EstimatedCost | None:
            return (
                None
                if cost_usd is None
                else EstimatedCost(amount=cost_usd, source="agent-reported")
            )

        def visible_budget() -> dict[str, float | int | None]:
            """This instant's headroom. Hard budgets only, by construction --
            see ``CutoffController.agent_visible_budget``. An agent that cannot
            see its step and time limits cannot plan against them, and would be
            judged on a horizon it was never told."""
            return controller.agent_visible_budget(
                elapsed_seconds=monotonic() - run_origin
            )

        initial = await environment.reset(seed=seed, dataset_id=dataset_id)
        trajectory = AgentTrajectory()
        # The budget goes on before the planning call, not after it: choosing how
        # many steps a plan should span is exactly when an agent needs to know how
        # many it has.
        opening = _observation_from_snapshot(
            initial, environment.task, environment.specification
        )
        observation = opening.model_copy(
            update={"metadata": {**opening.metadata, "budget": visible_budget()}}
        )
        trajectory.observations.append(observation)
        trajectory.record(AgentEventType.OBSERVATION, observation.model_dump(mode="json"))
        try:
            session = await _await_with_wall_budget(
                runtime.initialize(context), controller, run_origin, "initialization"
            )
        except _RuntimeWallTimeout as error:
            trajectory.record(AgentEventType.FAILURE, {"error": str(error)})
            await _close_runtime(runtime)
            environment.terminate(status=EpisodeStatus.FAILED, reason=str(error))
            final_snapshot = await environment.observe()
            return RuntimeRun(
                run_id=run_id,
                agent_id=runtime.agent_id,
                benchmark_id=final_snapshot.state.benchmark_id,
                task_id=final_snapshot.state.task_id,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                termination_status=RuntimeVerdict.TIMEOUT,
                termination_reason=str(error),
                step_count=0,
                trajectory=trajectory,
                final_submission=None,
                final_snapshot=final_snapshot,
                token_usage=token_usage_model(),
                estimated_cost=cost_model(),
                cutoff=controller.report(),
                boundary_exchanges=_runtime_exchange_log(runtime),
            )
        except Exception as error:
            # Initialization is part of the episode boundary. A transport or
            # provider failure here must still produce a persisted partial run;
            # allowing it to escape would erase the run id, opening observation,
            # and the fact that the endpoint was unavailable.
            message = f"agent initialization failed: {type(error).__name__}: {error}"
            trajectory.record(AgentEventType.FAILURE, {"error": message})
            await _close_runtime(runtime)
            environment.terminate(status=EpisodeStatus.FAILED, reason=message)
            final_snapshot = await environment.observe()
            return RuntimeRun(
                run_id=run_id,
                agent_id=runtime.agent_id,
                benchmark_id=final_snapshot.state.benchmark_id,
                task_id=final_snapshot.state.task_id,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                termination_status=RuntimeVerdict.FAILED,
                termination_reason=message,
                step_count=0,
                trajectory=trajectory,
                final_submission=None,
                final_snapshot=final_snapshot,
                token_usage=token_usage_model(),
                estimated_cost=cost_model(),
                cutoff=controller.report(),
                boundary_exchanges=_runtime_exchange_log(runtime),
            )
        initial_usage = session.state.pop("_boundary_usage", None)
        if isinstance(initial_usage, AgentUsage):
            consume_usage(initial_usage)
        await self._emit(
            {
                "type": "agent_planning",
                "step": 0,
                "message": "Asking the agent for an overall scientific plan.",
            }
        )
        plan_timeout: _RuntimeWallTimeout | None = None
        try:
            plan = await _await_with_wall_budget(
                runtime.plan(context, observation), controller, run_origin, "planning"
            )
            plan_source = "agent"
        except _RuntimeWallTimeout as error:
            plan = None
            plan_source = f"timeout: {error}"
            plan_timeout = error
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
                stopping_criteria=[
                    str(item.get("description", item.get("condition", "")))
                    for item in metadata.get("scenario", {}).get("stopping_criteria", [])
                    if isinstance(item, dict)
                ],
                adaptation_policy="After every result, keep, revise, or end the plan based on the new evidence.",
            )
        consume_usage(plan.usage)
        session.state["plan"] = plan.model_dump(mode="json")
        observation = observation.model_copy(
            update={
                "metadata": {
                    **observation.metadata,
                    "active_plan": plan.model_dump(mode="json"),
                    "plan_review": "After each result, decide whether to keep, revise, or end this plan.",
                    # Re-read rather than reuse the opening snapshot: planning is
                    # the one call that can take minutes, and a stale reading
                    # would hand the agent a horizon it no longer has.
                    "budget": visible_budget(),
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
        status = RuntimeVerdict.TIMEOUT if plan_timeout is not None else RuntimeVerdict.COMPLETED
        reason = str(plan_timeout) if plan_timeout is not None else "agent runtime completed"
        submission: FinalSubmission | None = None
        termination_attempted = False
        steps = 0
        try:
            while True:
                if status is RuntimeVerdict.TIMEOUT:
                    break
                cutoff = controller.decide(elapsed_seconds=monotonic() - run_origin)
                if cutoff.stop:
                    termination = cutoff_termination(cutoff, environment)
                    status, reason = termination.verdict, termination.reason
                    trajectory.record(
                        AgentEventType.FAILURE
                        if status is not RuntimeVerdict.COMPLETED
                        else AgentEventType.OBSERVATION,
                        {"cutoff": cutoff.model_dump(mode="json"), "reason": reason},
                    )
                    await self._emit(
                        {
                            "type": "run_cutoff",
                            "step": steps,
                            "message": reason,
                            "cutoff": cutoff.model_dump(mode="json"),
                        }
                    )
                    break
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
                try:
                    raw_action = await _await_with_wall_budget(
                        runtime.act(session, observation),
                        controller,
                        run_origin,
                        "action",
                    )
                except _RuntimeWallTimeout as error:
                    status = RuntimeVerdict.TIMEOUT
                    reason = str(error)
                    trajectory.record(AgentEventType.FAILURE, {"error": reason})
                    await self._emit(
                        {"type": "run_cutoff", "step": steps, "message": reason}
                    )
                    break
                action = AgentAction.model_validate(raw_action)
                consume_usage(action.usage)
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
                if action.plan_update is not None:
                    # A plan is an observable working hypothesis, not private
                    # chain-of-thought. Persist revisions so the trajectory can
                    # distinguish a scientist adapting to evidence from one
                    # repeating an obsolete opening plan.
                    plan = action.plan_update
                    session.state["plan"] = plan.model_dump(mode="json")
                    trajectory.record(
                        AgentEventType.PLAN,
                        plan.model_dump(mode="json"),
                        parent_event_id=action_event.event_id,
                    )
                    await self._emit(
                        {
                            "type": "agent_plan_revision",
                            "step": current_step,
                            "message": "Agent revised the working scientific plan.",
                            "plan": plan.model_dump(mode="json"),
                        }
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
                        tool_result = await _await_with_wall_budget(
                            self.tool_executor.execute(
                                action.action_type,
                                action.parameters,
                                context=session,
                            ),
                            controller,
                            run_origin,
                            "tool call",
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
                            consume_usage(action.usage)
                        else:
                            steps += 1
                            # A tool call that returned data rather than an action
                            # still spends a step, so the controller has to see it
                            # or its step count drifts below the loop's. No
                            # signature: a tool call is not a scientific decision.
                            controller.observe(
                                StepObservation(
                                    step=steps,
                                    succeeded=True,
                                    total_tokens=total_tokens,
                                    cost_usd=cost_usd,
                                )
                            )
                            continue
                    except _RuntimeWallTimeout as error:
                        status = RuntimeVerdict.TIMEOUT
                        reason = str(error)
                        trajectory.record(
                            AgentEventType.FAILURE,
                            {"tool": action.action_type, "error": reason},
                            parent_event_id=action_event.event_id,
                        )
                        await self._emit(
                            {"type": "run_cutoff", "step": steps, "message": reason}
                        )
                        break
                    except Exception as error:
                        trajectory.record(
                            AgentEventType.FAILURE,
                            {"tool": action.action_type, "error": str(error)},
                            parent_event_id=action_event.event_id,
                        )
                        steps += 1
                        controller.observe(
                            StepObservation(
                                step=steps,
                                succeeded=False,
                                total_tokens=total_tokens,
                                cost_usd=cost_usd,
                            )
                        )
                        continue
                if action.action_type in self.terminal_actions:
                    # A terminal action is a claim of completion, not proof of it.
                    # Accepting it unverified lets an agent score "completed" by
                    # finishing immediately without producing any artifact.
                    missing = _missing_required_artifacts(environment)
                    # Termination is a lifecycle side effect, not a safe retryable
                    # read. Mark the attempt before calling the provider so an
                    # error in the closing exchange cannot cause the outer failure
                    # handler to POST ``terminate`` a second time.
                    termination_attempted = True
                    submission = await _terminate_with_grace(runtime, session, observation)
                    consume_usage(submission.usage)
                    trajectory.final_submission = submission
                    trajectory.record(
                        AgentEventType.FINAL_SUBMISSION,
                        submission.model_dump(mode="json"),
                        parent_event_id=action_event.event_id,
                    )
                    if missing:
                        status = RuntimeVerdict.INCOMPLETE
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
                try:
                    result = await _await_with_wall_budget(
                        environment.step(intent),
                        controller,
                        run_origin,
                        "environment step",
                    )
                except _RuntimeWallTimeout as error:
                    # A scientific executor is part of the episode's wall-clock
                    # budget too. Force the controller to observe the boundary
                    # before the timeout branch archives the run, otherwise the
                    # run would say it timed out while its cutoff report claimed
                    # no cutoff fired.
                    wall_limit = controller.budget.max_wall_time_seconds
                    elapsed = monotonic() - run_origin
                    cutoff = controller.decide(
                        elapsed_seconds=max(elapsed, wall_limit or elapsed)
                    )
                    if not cutoff.stop:
                        cutoff = CutoffDecision(
                            stop=True,
                            reason=CutoffReason.WALL_TIME,
                            detail=str(error),
                        )
                    termination = cutoff_termination(cutoff, environment)
                    status, reason = termination.verdict, termination.reason
                    trajectory.record(
                        AgentEventType.FAILURE,
                        {
                            "action_type": action.action_type,
                            "error": reason,
                            "cutoff": cutoff.model_dump(mode="json"),
                        },
                        parent_event_id=action_event.event_id,
                    )
                    await self._emit(
                        {
                            "type": "run_cutoff",
                            "step": steps,
                            "message": reason,
                            "cutoff": cutoff.model_dump(mode="json"),
                        }
                    )
                    break
                trajectory.record(
                    AgentEventType.ENVIRONMENT_RESPONSE,
                    result.model_dump(mode="json"),
                    parent_event_id=action_event.event_id,
                )
                step_succeeded = (
                    result.accepted
                    and result.execution is not None
                    and result.execution.status == ActionStatus.SUCCEEDED
                )
                if not step_succeeded:
                    trajectory.record(
                        AgentEventType.FAILURE,
                        {
                            "action_type": action.action_type,
                            "error": result.execution.error if result.execution else result.validation.errors,
                        },
                        parent_event_id=action_event.event_id,
                    )
                steps += 1
                controller.observe(
                    StepObservation(
                        step=steps,
                        succeeded=step_succeeded,
                        signature=decision_signature(action.action_type, action.parameters),
                        progress_delta=progress_delta(result),
                        total_tokens=total_tokens,
                        cost_usd=cost_usd,
                    )
                )
                next_observation = _observation_from_snapshot(
                    result.observation, environment.task, environment.specification
                )
                observation = next_observation.model_copy(
                    update={
                        "metadata": {
                            **next_observation.metadata,
                            "active_plan": plan.model_dump(mode="json"),
                            "plan_review": "After this result, decide whether to keep, revise, or end the plan.",
                            "budget": visible_budget(),
                        }
                    }
                )
                trajectory.observations.append(observation)
                trajectory.record(
                    AgentEventType.OBSERVATION,
                    observation.model_dump(mode="json"),
                    parent_event_id=action_event.event_id,
                )
            if submission is None:
                termination_attempted = True
                submission = await _terminate_with_grace(runtime, session, observation)
                consume_usage(submission.usage)
                trajectory.final_submission = submission
                trajectory.record(
                    AgentEventType.FINAL_SUBMISSION,
                    submission.model_dump(mode="json"),
                )
        except Exception as error:
            status = RuntimeVerdict.FAILED
            reason = str(error)
            trajectory.record(AgentEventType.FAILURE, {"error": str(error)})
            if not termination_attempted:
                termination_attempted = True
                try:
                    submission = await _terminate_with_grace(runtime, session, observation)
                    consume_usage(submission.usage)
                    trajectory.final_submission = submission
                except Exception as termination_error:
                    trajectory.record(AgentEventType.FAILURE, {"error": str(termination_error)})
            else:
                trajectory.record(
                    AgentEventType.FAILURE,
                    {"error": "termination was already attempted; no retry was issued"},
                )
        if environment.episode is not None and environment.episode.status not in {
            EpisodeStatus.COMPLETED,
            EpisodeStatus.FAILED,
            EpisodeStatus.CANCELLED,
        }:
            environment.terminate(
                status=EpisodeStatus.COMPLETED
                if status is RuntimeVerdict.COMPLETED
                else EpisodeStatus.FAILED,
                reason=reason,
            )
        await _close_runtime(runtime)
        final_snapshot = await environment.observe()
        finished_at = datetime.now(UTC)
        return RuntimeRun(
            run_id=run_id,
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
            token_usage=token_usage_model(),
            estimated_cost=cost_model(),
            cutoff=controller.report(),
            boundary_exchanges=_runtime_exchange_log(runtime),
        )


#: Cutoff reasons that describe a *consumed budget*. These keep mapping to
#: ``TIMEOUT`` because that is what they are -- the run ran out of something.
#: Stagnation and repetition do not: the run had budget left and was not using it
#: to make progress, which is a different finding and gets its own verdict.
_BUDGET_CUTOFFS = frozenset(
    {
        CutoffReason.MAX_STEPS,
        CutoffReason.WALL_TIME,
        CutoffReason.COST,
        CutoffReason.TOKENS,
        CutoffReason.CONSECUTIVE_FAILURES,
    }
)


class CutoffTermination(BaseModel):
    """How one fired cutoff is archived.

    A single object rather than three lookups at each call site, because every
    loop that can be cut off has to reach the same verdict, status, and failure
    kind from the same cutoff -- and getting that wrong fails *silently*, filing
    a run the controller stopped as one that finished cleanly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: What the runtime loop calls this ending.
    verdict: RuntimeVerdict
    #: What the archived run records as its terminal status.
    status: RunTerminationStatus
    #: ``None`` only when the run still met its artifact contract. Every other
    #: ending keeps a failure, because the contract genuinely was not met.
    failure_kind: FailureKind | None
    #: Human-readable, and always names the cutoff detail it came from.
    reason: str


def cutoff_termination(
    cutoff: CutoffDecision, environment: ScientificEnvironment
) -> CutoffTermination:
    """Translate a fired cutoff into how the run should be recorded.

    A run that produced everything the benchmark required and then hit its step
    budget completed; it did not time out. That check is the one the old
    ``while``/``else`` clause performed, kept here so exhausting the budget means
    the same thing it always did, and extended to the reasons that did not exist
    before.

    A benchmark that requires no artifacts is *not* credited with completing --
    the ``required and`` guard is load-bearing, because an empty required set
    makes the subset test vacuously true and would file every cut-off run as
    finished. With nothing to check against, a run the controller stopped is
    recorded as stopped. This became reachable once requiredness started coming
    from ``required_task_artifacts``, which is empty for a free-execution
    benchmark whose artifacts are all optional by design.
    """
    artifacts = (
        environment.episode.snapshot().state.artifacts
        if environment.episode
        else {}
    )
    required = environment.specification.required_task_artifacts(environment.task)
    detail = cutoff.detail or "a run cutoff fired"
    validated = {
        artifact_id
        for artifact_id, artifact in artifacts.items()
        if artifact.validated
    }
    if required and required.issubset(validated):
        verdict, reason = (
            RuntimeVerdict.COMPLETED,
            f"required benchmark artifacts were produced before the run stopped: {detail}",
        )
    elif cutoff.reason in _BUDGET_CUTOFFS:
        verdict, reason = (
            RuntimeVerdict.TIMEOUT,
            f"the run stopped before the benchmark goal was satisfied: {detail}",
        )
    else:
        verdict, reason = (
            RuntimeVerdict.STAGNATED,
            f"the controller stopped an unproductive run: {detail}",
        )
    status = _TERMINATION_STATUSES.get(verdict, RunTerminationStatus.FAILED)
    return CutoffTermination(
        verdict=verdict,
        status=status,
        failure_kind=_FAILURE_KINDS.get(status),
        reason=reason,
    )


def decision_signature(action_name: str, parameters: Mapping[str, Any]) -> str:
    """Fingerprint an action so a repeated decision is recognizable.

    Takes the name and parameters rather than an action object because the typed
    baseline holds an ``ActionIntent`` where the universal loop holds an
    ``AgentAction``. Both must fingerprint identically or the same loop would be
    detected on one path and not the other.

    Parameters are included and sorted, so the same method at a different
    resolution is not a repeat. ``default=str`` because a parameter value only
    has to be *comparable* here, not round-trippable -- and a fingerprint that
    raised on an exotic value would break the loop over a detail it is allowed to
    be imprecise about.
    """
    return f"{action_name}({json.dumps(dict(parameters), sort_keys=True, default=str)})"


def progress_delta(result: EnvironmentStep) -> float | None:
    """Read ``dS`` off the step's reward, or ``None`` when nobody measured it.

    Absent is not zero. The key is only written when two consecutive steps shared
    a comparable metric, and treating its absence as a flat delta is what would
    let the controller stop a run for the harness's blindness.
    """
    if result.reward is None:
        return None
    value = result.reward.metric_values.get(PROGRESS_DELTA_KEY)
    return None if value is None else float(value)


#: Each runtime verdict mapped to the status archived in the run record. Every
#: member of :class:`RuntimeVerdict` needs an entry: an unmapped one falls back to
#: ``FAILED``, which is lossy rather than wrong, so the omission would not raise
#: and would surface only as a status that never appears in any result file.
_TERMINATION_STATUSES: dict[RuntimeVerdict, RunTerminationStatus] = {
    RuntimeVerdict.COMPLETED: RunTerminationStatus.COMPLETED,
    RuntimeVerdict.TIMEOUT: RunTerminationStatus.TIMEOUT,
    RuntimeVerdict.FAILED: RunTerminationStatus.FAILED,
    RuntimeVerdict.INCOMPLETE: RunTerminationStatus.INCOMPLETE,
    RuntimeVerdict.STAGNATED: RunTerminationStatus.STAGNATED,
}

#: How a non-completed status is described in the retained failure. A failure is
#: still recorded in every case -- the run genuinely did not meet its contract --
#: but the kind now says which contract, instead of reporting a clean early stop
#: and a timeout as the same agent malfunction.
_FAILURE_KINDS: dict[RunTerminationStatus, FailureKind] = {
    RunTerminationStatus.TIMEOUT: FailureKind.TIMEOUT,
    RunTerminationStatus.STAGNATED: FailureKind.STAGNATION,
    RunTerminationStatus.INCOMPLETE: FailureKind.INCOMPLETE_SUBMISSION,
    RunTerminationStatus.UNAVAILABLE: FailureKind.ADAPTER_UNAVAILABLE,
    RunTerminationStatus.INVALID_CONFIGURATION: FailureKind.INVALID_ACTION,
}


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
        from agent_evals.agents.trajectory import AgentFailure, RawTraceEvent
        from agent_evals.agents.trajectory import (
            AgentManifest as LegacyAgentManifest,
        )
        from agent_evals.agents.trajectory import (
            AgentModelInfo as LegacyAgentModelInfo,
        )

        root = str(configuration.workspace.get("root", "."))
        constraints = _model_dump(
            environment.task.constraints or environment.specification.constraints
        )
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
        status = _TERMINATION_STATUSES.get(
            universal.termination_status, RunTerminationStatus.FAILED
        )
        failures = (
            [
                AgentFailure(
                    kind=_FAILURE_KINDS.get(status, FailureKind.AGENT_ERROR),
                    message=universal.termination_reason or "runtime failed",
                )
            ]
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
            token_usage=universal.token_usage,
            estimated_cost=universal.estimated_cost,
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
                # Persisted here because this is the only conversion between the
                # runtime result and the archived run, so a cutoff report left out
                # of it would be computed correctly every run and read by nobody.
                "cutoff": universal.cutoff.model_dump(mode="json")
                if universal.cutoff is not None
                else None,
                "boundary_exchanges": universal.boundary_exchanges,
            },
        )


def _observation_from_snapshot(
    snapshot: EpisodeSnapshot,
    task: TaskSpecification,
    specification: BenchmarkSpecification | None = None,
) -> AgentObservation:
    """Project an environment snapshot and scientific goal into agent-visible context.

    This is the only place an agent is handed the episode state wholesale, so it
    is the one place the ``visible_to_agent`` boundary has to hold. Both channels
    that carry observations -- the state itself and the convenience copy under
    ``metadata`` -- are filtered through :func:`agent_visible_state` and
    :func:`agent_visible_observations`; filtering one and not the other filters
    nothing, since the executor's isolation report would still reach the agent
    through whichever was left.

    ``declared_artifacts`` and ``required_artifacts`` are separate keys because
    they answer different questions and the task's reference list only answers the
    first. An agent needs every declared id to label its outputs with -- on the
    free tier an artifact is only recorded if the agent names it -- but publishing
    that list as *required* told the free benchmark's agent to produce four files
    when the benchmark demands none, which is a false contract that costs real
    steps. Requiredness is resolvable only with the specification, so without one
    the key is omitted rather than guessed: an absent key reads as unknown, and an
    empty list would assert that nothing is required.
    """
    visible = {
        key: value.model_dump(mode="json")
        for key, value in agent_visible_observations(snapshot.state.observations).items()
    }
    scientific_payload = visible.get("scientific-observation", {}).get("value", {})
    available_actions = (
        list(scientific_payload.get("available_actions", []))
        if isinstance(scientific_payload, dict)
        else []
    ) or list(task.allowed_actions)
    package = build_agent_task_package(specification, task) if specification is not None else {}
    previous_decision: dict[str, Any] | None = None
    state_delta: dict[str, Any] | None = None
    if snapshot.state.actions:
        last_record = snapshot.state.actions[-1]
        previous_decision = {
            "step": last_record.step,
            "action_type": last_record.intent.action_id,
            "parameters": {
                key: value
                for key, value in last_record.intent.parameters.items()
                if key not in {"code", "produces"}
            },
            "rationale": last_record.intent.rationale,
        }
        if last_record.result.observed_state_delta is not None:
            state_delta = last_record.result.observed_state_delta.model_dump(mode="json")
    return AgentObservation(
        state=agent_visible_state(snapshot.state).model_dump(mode="json"),
        available_actions=available_actions,
        artifacts=[artifact.model_dump(mode="json") for artifact in snapshot.state.artifacts.values()],
        previous_decision=previous_decision,
        state_delta=state_delta,
        metadata={
            "episode_id": snapshot.state.episode_id,
            "step": snapshot.state.current_step,
            "scenario": _scenario(task, specification),
            "goal": task.objective,
            "observations": visible,
            "task_package": package,
            "interaction": _interaction_feedback(snapshot),
        },
    )


def _scenario(
    task: TaskSpecification,
    specification: BenchmarkSpecification | None,
) -> dict[str, Any]:
    """Describe what the task asks for, without prescribing how to do it.

    ``artifact_contracts`` is the discoverable half of the output contract. The
    benchmark validates a produced artifact against declared
    :class:`ValidationRule` s and scores the verdict -- ``artifact_validity``
    carries weight inside ``trajectory_quality`` -- and until now the rules were
    published nowhere, so an agent could satisfy "columns include
    barcode,predicted_label" only by guessing the column name. A rule the agent
    cannot read is one it can only pass by luck, and luck is not what the
    benchmark means to measure.

    Publishing them leaks nothing: a rule names the shape of the agent's *own*
    output and the vocabulary of its own intent parameters, never reference
    biology, and the benchmark YAML that declares it is public anyway. This is the
    line the environment documentation is meant to hold -- specify the
    environment and the contract, never the method.
    """
    scenario: dict[str, Any] = build_agent_task_package(specification, task) if specification is not None else {
        "task": {
            "id": task.id,
            "name": task.name,
            "objective": task.objective,
            "description": task.description,
        },
    }
    # Keep these short aliases for existing runtimes and reports. The richer
    # package above is the canonical contract; the aliases avoid breaking
    # consumers that already read ``scenario.success_criteria``.
    scenario.update({
        "name": task.name,
        "objective": task.objective,
        "end_goal": scenario.get("task", {}).get("end_goal", task.objective),
        "description": task.description,
        "success_criteria": [
            {
                "name": condition.name,
                "description": condition.description,
                "condition": condition.condition,
            }
            for condition in task.termination
        ],
        "stopping_criteria": [
            {
                "name": condition.name,
                "description": condition.description,
                "condition": condition.condition,
                "terminal": condition.terminal,
            }
            for condition in task.termination
        ],
        "declared_artifacts": list(task.artifacts),
        "required_metrics": list(task.metrics),
    })
    if specification is None:
        return scenario
    required = specification.required_task_artifacts(task)
    declared = {artifact.id: artifact for artifact in specification.artifacts}
    scenario["required_artifacts"] = [
        artifact_id for artifact_id in task.artifacts if artifact_id in required
    ]
    scenario["artifact_contracts"] = [
        {
            "artifact_id": artifact.id,
            "name": artifact.name,
            "description": artifact.description,
            "kind": artifact.kind.value,
            "format": artifact.format,
            "required": artifact.id in required,
            "validation": [
                {
                    "name": rule.name,
                    "description": rule.description,
                    "rule": rule.rule,
                }
                for rule in artifact.validation
            ],
        }
        for artifact in (declared[key] for key in task.artifacts if key in declared)
    ]
    return scenario


def _interaction_feedback(snapshot: EpisodeSnapshot) -> dict[str, Any]:
    """Expose the last observable result in a compact, decision-ready form."""
    if not snapshot.state.actions:
        return {
            "phase": "initial_observation",
            "message": "Inspect the supplied data summary and task package before selecting the first action.",
            "last_action": None,
        }
    record = snapshot.state.actions[-1]
    result = record.result
    delta = result.observed_state_delta
    return {
        "phase": "post_action_review",
        "message": "Review this result, update the working plan if needed, and select the next evidence-producing action.",
        "last_action": {
            "step": record.step,
            "action": record.intent.action_id,
            "parameters": dict(record.intent.parameters),
            "status": result.status.value,
            "execution_status": result.execution_status.value if result.execution_status else None,
            "error": result.error,
            "outputs": dict(result.outputs),
            "artifacts": [artifact.artifact_id for artifact in result.artifacts],
            "state_delta": delta.summary() if delta is not None else None,
        },
    }


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
    parameter_method = parameters.get("method")
    if parameter_method is None and declared_action is not None:
        parameter_method = next(
            (
                parameter.default
                for parameter in declared_action.parameters
                if parameter.name == "method" and parameter.default is not None
            ),
            None,
        )
    reasoning = dict(action.reasoning_metadata)
    if action.state_claim:
        # Keep the agent's claim separate from the executor's observed delta. The
        # decision cascade verifies the two later; merging it into the public
        # reasoning mapping makes the claim survive the endpoint boundary without
        # making it authoritative.
        reasoning["state_claim"] = dict(action.state_claim)
    if action.next_step:
        reasoning["next_step"] = dict(action.next_step)
    rationale = (
        reasoning.get("explanation") or reasoning.get("summary") or reasoning.get("rationale")
    )
    # Coerce before the metadata reaches ``decision_cascade_from_episode``, which
    # indexes these keys by type. Unparsed, a string where it expects a mapping
    # raised, and a string where it expects a list recorded one item per
    # character -- inventing evidence the agent never offered.
    extracted = extract_decision(reasoning)
    metadata: dict[str, Any] = {
        "runtime_action_type": action.action_type,
        **extracted.metadata,
    }
    if action.extraction_evidence is not None:
        # Boundary provenance is generated by SCAIB, not claimed by the agent.
        # Carry it into the canonical intent so the evidence survives the
        # universal-runtime to AgentRun conversion and can be audited beside the
        # decision it produced.
        metadata["response_extraction"] = action.extraction_evidence.model_dump(
            mode="json"
        )
    selected_method = extracted.metadata.get("method") or parameter_method
    if selected_method is not None:
        # A method declared as an action parameter is still a scientific method
        # choice. Without this bridge the evaluator recorded the coarse action
        # id (``qc``) and silently ignored the agent's granular QC selection.
        metadata["method"] = selected_method
        metadata["method_id"] = selected_method
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
    """Return the required artifact IDs this run has not produced.

    Requiredness comes from ``BenchmarkSpecification.required_task_artifacts``
    rather than from the task's reference list, because a benchmark whose
    artifacts are deliberately optional must be finishable. A free-execution
    benchmark cannot demand a fixed set of files without dictating the pipeline
    shape it exists to measure, so per-invocation enforcement happens through the
    action's ``produces`` parameter instead.
    """
    required = environment.specification.required_task_artifacts(environment.task)
    if not required:
        return set()
    artifacts = (
        environment.episode.snapshot().state.artifacts
        if environment.episode is not None
        else {}
    )
    return {
        artifact_id
        for artifact_id in required
        if artifact_id not in artifacts or not artifacts[artifact_id].validated
    }


async def _await_with_wall_budget(
    awaitable: Awaitable[Any],
    controller: CutoffController,
    run_origin: float,
    label: str,
) -> Any:
    """Bound provider work by the same wall clock the controller reports."""
    limit = controller.budget.max_wall_time_seconds
    if limit is None:
        return await awaitable
    remaining = limit - (monotonic() - run_origin)
    if remaining <= 0:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise _RuntimeWallTimeout(
            f"{label} could not start because the {limit:g}s wall-clock budget was exhausted"
        )
    try:
        return await asyncio.wait_for(awaitable, timeout=remaining)
    except TimeoutError as error:
        raise _RuntimeWallTimeout(
            f"{label} exceeded the remaining {remaining:.2f}s of the "
            f"{limit:g}s wall-clock budget"
        ) from error


def _runtime_exchange_log(runtime: AgentRuntime) -> list[dict[str, Any]]:
    """Copy bounded transport provenance exposed by a runtime, if any."""
    value = getattr(runtime, "exchange_log", None)
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


async def _close_runtime(runtime: AgentRuntime) -> None:
    """Close a runtime transport without turning cleanup into a run failure."""
    try:
        await runtime.close()
    except Exception:
        # The scientific result and its failure reason are more valuable than a
        # provider-specific cleanup traceback. The endpoint runtime's own
        # termination path already records request failures; this is final cleanup.
        return


async def _terminate_with_grace(
    runtime: AgentRuntime,
    session: AgentSession,
    observation: AgentObservation | None,
) -> FinalSubmission:
    """Do not let cleanup hang after a provider has exhausted the run budget."""
    try:
        return await asyncio.wait_for(
            runtime.terminate(session, observation), timeout=5.0
        )
    except TimeoutError:
        return FinalSubmission(
            metadata={"termination": "agent termination exceeded 5s grace period"}
        )


def _is_registered_tool(executor: ToolExecutor, name: str) -> bool:
    """Check tool membership without turning ordinary environment actions into tools."""
    try:
        executor.registry.get(name)
    except KeyError:
        return False
    return True


__all__ = [
    "AgentRuntimeManager",
    "CutoffTermination",
    "RuntimeAgentAdapter",
    "RuntimeRun",
    "cutoff_termination",
    "decision_signature",
    "progress_delta",
]
