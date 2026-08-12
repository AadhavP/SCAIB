"""Declarative scientific environment state machine."""

from __future__ import annotations

from typing import Any

from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.core.exceptions import EnvironmentStateError
from agent_evals.environment.episode import Episode
from agent_evals.environment.models import (
    ActionExecutionResult,
    ActionIntent,
    ActionStatus,
    ActionValidationResult,
    ArtifactValidation,
    EnvironmentStep,
    EpisodeSnapshot,
    EpisodeStatus,
    ExecutionStatus,
    RewardRecord,
)
from agent_evals.environment.ports import (
    ActionExecutor,
    ArtifactValidator,
    ConstraintMonitor,
    DeclarativeActionValidator,
    ExecutionContext,
    ObservationBuilder,
    RewardEvaluator,
)


class ScientificEnvironment:
    """Run typed action intents against a benchmark-defined episode.

    The environment owns lifecycle, permissions, state transitions, visible
    observations, and episode trace management.  Scientific computation is
    delegated to ``ActionExecutor``; metric and reward computation is delegated
    to ``RewardEvaluator``.  This class therefore provides the research-facing
    world model without coupling the framework to Scanpy, Docker, or an agent
    implementation.
    """

    def __init__(
        self,
        specification: BenchmarkSpecification,
        *,
        task_id: str,
        executor: ActionExecutor,
        observation_builder: ObservationBuilder | None = None,
        reward_evaluator: RewardEvaluator | None = None,
        validator: DeclarativeActionValidator | None = None,
        constraint_monitor: ConstraintMonitor | None = None,
        artifact_validator: ArtifactValidator | None = None,
    ) -> None:
        self.specification = specification
        self.task = self._resolve_task(task_id)
        self.executor = executor
        self.observation_builder = observation_builder
        self.reward_evaluator = reward_evaluator
        self.validator = validator or DeclarativeActionValidator()
        self.constraint_monitor = constraint_monitor or ConstraintMonitor()
        #: Optional, because the environment can run without checking artifact
        #: contents.  When absent every artifact keeps ``validated=False``, which
        #: reads as unvalidated rather than as invalid.
        self.artifact_validator = artifact_validator
        self._episode: Episode | None = None

    async def reset(
        self,
        *,
        seed: int,
        dataset_id: str | None = None,
        episode_id: str | None = None,
    ) -> EpisodeSnapshot:
        """Create and start a fresh episode, optionally selecting one dataset."""
        if self._episode is not None and self._episode.status not in {
            EpisodeStatus.COMPLETED,
            EpisodeStatus.FAILED,
            EpisodeStatus.CANCELLED,
        }:
            raise EnvironmentStateError("cannot reset while an episode is running")
        self._episode = Episode.from_specification(
            self.specification,
            task_id=self.task.id,
            seed=seed,
            dataset_id=dataset_id,
            episode_id=episode_id,
        )
        self._episode.start()
        reset_reward = getattr(self.reward_evaluator, "reset", None)
        if callable(reset_reward):
            reset_reward(self._episode.snapshot())
        await self._refresh_observations()
        return self._episode.snapshot()

    async def observe(self) -> EpisodeSnapshot:
        """Return the current episode snapshot without changing state."""
        if self._episode is None:
            raise EnvironmentStateError("environment has not been reset")
        return self._episode.snapshot()

    async def step(self, intent: ActionIntent) -> EnvironmentStep:
        """Validate and execute one typed action intent.

        Invalid intents are recorded as rejected events without advancing the
        episode step.  Valid intents always advance the step, including failed
        executions, so failures remain part of the replayable scientific trace.
        Failed executions do not commit outputs and do not terminate the
        episode automatically; callers may retry or explicitly terminate.
        """
        episode = self._require_episode()
        snapshot = episode.snapshot()
        validation = self.validator.validate(
            intent,
            self.specification,
            self.task,
            snapshot,
        )
        if not validation.valid:
            episode.record_rejection(intent, validation.errors)
            return EnvironmentStep(
                episode_id=episode.episode_id,
                accepted=False,
                validation=validation,
                observation=episode.snapshot(),
            )

        episode.record_submission(intent)
        constraints = self.task.constraints or self.specification.constraints
        try:
            result = await self.executor.execute(intent, ExecutionContext(snapshot, constraints))
        except Exception as error:  # pragma: no cover - exercised by integration executors
            result = ActionExecutionResult(
                intent_id=intent.intent_id,
                action_id=intent.action_id,
                status=ActionStatus.FAILED,
                execution_status=ExecutionStatus.ERROR,
                error=f"executor error: {error}",
            )

        result = self._normalize_result(intent, result)
        result = self._apply_resource_constraints(
            result,
            constraints,
            episode.snapshot().state.resource_usage,
        )
        # Before ``record_action``, because the record is what scoring reads and
        # an artifact recorded unvalidated stays that way. Applied to failed steps
        # too: their artifacts are persisted either way, and ``artifact_validity``
        # is computed over every recorded artifact rather than only committed ones.
        result = await self._validate_artifacts(intent, result)
        episode.record_action(intent, result)
        # After ``record_action``, so the event lands at the step it describes,
        # and gated on a non-empty delta so a step that changed nothing does not
        # add a change event claiming otherwise. An unobserved delta is not empty
        # -- it carries the limitations saying why -- so it is still recorded.
        delta = result.observed_state_delta
        if delta is not None and not (delta.is_empty and not delta.unobserved):
            episode.record_state_change(delta)

        reward: RewardRecord | None = None
        if result.status == ActionStatus.SUCCEEDED:
            episode.record_outputs(result.observations, result.artifacts)
            if self.reward_evaluator is not None:
                reward = await self.reward_evaluator.evaluate(
                    self.specification,
                    self.task,
                    episode.snapshot(),
                    result,
                )
                if reward is not None:
                    reward = reward.model_copy(update={"step": episode.snapshot().state.current_step})
                    episode.record_reward(reward)
            await self._refresh_observations()

        return EnvironmentStep(
            episode_id=episode.episode_id,
            accepted=True,
            validation=ActionValidationResult(valid=True),
            execution=result,
            reward=reward,
            observation=episode.snapshot(),
        )

    def terminate(
        self,
        *,
        status: EpisodeStatus = EpisodeStatus.COMPLETED,
        reason: str | None = None,
    ) -> EpisodeSnapshot:
        """Explicitly close the current episode and return its final snapshot."""
        episode = self._require_episode()
        episode.terminate(status, reason)
        return episode.snapshot()

    @property
    def episode(self) -> Episode | None:
        """Return the active episode controller for persistence integrations."""
        return self._episode

    def _require_episode(self) -> Episode:
        """Return the active episode or raise a useful lifecycle error."""
        if self._episode is None:
            raise EnvironmentStateError("environment has not been reset")
        if self._episode.status in {
            EpisodeStatus.COMPLETED,
            EpisodeStatus.FAILED,
            EpisodeStatus.CANCELLED,
        }:
            raise EnvironmentStateError(
                f"episode '{self._episode.episode_id}' is already {self._episode.status.value}"
            )
        return self._episode

    async def _validate_artifacts(
        self,
        intent: ActionIntent,
        result: ActionExecutionResult,
    ) -> ActionExecutionResult:
        """Check every produced artifact against the rules its benchmark declared.

        ``validated`` is set from the check rather than by whoever wrote the file.
        That is the same principle applied to completion claims and to state
        deltas: a producer's assertion about its own output is the thing under
        evaluation, so it cannot also be the evidence.
        """
        if self.artifact_validator is None or not result.artifacts:
            return result
        rules = {item.id: item.validation for item in self.specification.artifacts}
        validated = []
        for artifact in result.artifacts:
            try:
                validation = await self.artifact_validator.validate(
                    artifact,
                    rules.get(artifact.artifact_id, []),
                    intent.parameters,
                )
            except Exception as error:  # a check must not fail the science
                # Recorded as a harness limitation with nothing established, which
                # leaves ``validated`` false for the honest reason: no check ran.
                validation = ArtifactValidation(
                    limitations=[
                        f"the artifact validator failed: {type(error).__name__}: {error}"
                    ]
                )
            validated.append(
                artifact.model_copy(
                    update={"validation": validation, "validated": validation.is_valid}
                )
            )
        return result.model_copy(update={"artifacts": validated})

    async def _refresh_observations(self) -> None:
        """Ask the observation port for values and commit them atomically."""
        if self.observation_builder is None:
            return
        episode = self._require_episode()
        observations = await self.observation_builder.build(
            self.specification,
            self.task,
            episode.snapshot(),
        )
        episode.record_observations(list(observations))

    def _resolve_task(self, task_id: str) -> TaskSpecification:
        """Resolve a task from the immutable benchmark specification."""
        task = next((item for item in self.specification.tasks if item.id == task_id), None)
        if task is None:
            raise EnvironmentStateError(f"unknown task '{task_id}'")
        return task

    def _normalize_result(
        self,
        intent: ActionIntent,
        result: ActionExecutionResult,
    ) -> ActionExecutionResult:
        """Turn executor protocol mismatches into a failed typed result."""
        errors: list[str] = []
        if result.intent_id != intent.intent_id:
            errors.append("executor returned a different intent_id")
        if result.action_id != intent.action_id:
            errors.append("executor returned a different action_id")
        action = next(item for item in self.specification.actions if item.id == intent.action_id)
        produced = set(result.outputs) | {artifact.artifact_id for artifact in result.artifacts}
        missing_outputs = sorted(set(action.expected_outputs) - produced)
        if result.status == ActionStatus.SUCCEEDED and missing_outputs:
            actual_outputs = ", ".join(sorted(produced)) or "none"
            errors.append(
                "executor omitted expected output(s): "
                f"{', '.join(missing_outputs)}; produced: {actual_outputs}"
            )
        if not errors:
            return result
        message = "; ".join(errors)
        return ActionExecutionResult(
            intent_id=intent.intent_id,
            action_id=intent.action_id,
            status=ActionStatus.FAILED,
            # An execution that ran cleanly but did not produce everything it
            # declared is partial, not an error; an execution that already
            # failed keeps whatever reason it reported.
            execution_status=self._contract_status(result, ExecutionStatus.PARTIAL),
            error=message,
            resource_usage=result.resource_usage,
            # Carried across the rebuild deliberately. The contract check is a
            # judgement about the result; the delta is a measurement of what the
            # code did, and discarding it here would lose exactly the evidence
            # that explains a step which ran but declared the wrong outputs.
            observed_state_delta=result.observed_state_delta,
        )

    @staticmethod
    def _contract_status(
        result: ActionExecutionResult,
        replacement: ExecutionStatus,
    ) -> ExecutionStatus | None:
        """Keep an executor's own failure reason instead of overwriting it."""
        if result.execution_status in (None, ExecutionStatus.SUCCESS):
            return replacement
        return result.execution_status

    def _apply_resource_constraints(
        self,
        result: ActionExecutionResult,
        constraints: Any,
        previous_usage: Any,
    ) -> ActionExecutionResult:
        """Convert resource violations into failed results before state commit."""
        if result.status == ActionStatus.FAILED:
            return result
        violations = self.constraint_monitor.check(
            constraints,
            result.resource_usage,
            previous_usage,
        )
        if not violations:
            return result
        return ActionExecutionResult(
            intent_id=result.intent_id,
            action_id=result.action_id,
            status=ActionStatus.FAILED,
            # The work itself may have finished; the episode ran out of budget.
            execution_status=self._contract_status(result, ExecutionStatus.TERMINATED),
            error="; ".join(violations),
            resource_usage=result.resource_usage,
            # A step that overran its budget is the case where knowing what it
            # already changed matters most: the outputs are not committed, so
            # the delta is the only record of what the data now looks like.
            observed_state_delta=result.observed_state_delta,
        )


__all__ = ["ScientificEnvironment"]
