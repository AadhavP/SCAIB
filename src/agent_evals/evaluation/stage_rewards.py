"""Reward evaluator that also records per-step scientific state.

Implements the environment's existing ``RewardEvaluator`` port, so the step loop
needs no change: ``ScientificEnvironment`` already calls ``evaluate`` on every
accepted step and already offers a duck-typed ``reset`` hook at episode start.

Two deliberate restraints.

**The reward scalar is the delegate's, untouched.** ``S_t`` is derived from the
held-out reference, and the reward value is the one number a reinforcement-
learning consumer would optimize directly. Feeding reference-derived quality into
it would turn the dense local reward into a channel for exactly the information
the benchmark withholds. Progress is recorded as evidence beside the reward
instead, where only the evaluator reads it.

**Nothing here may raise.** Scoring a step is observation, and an observation
that throws would fail an agent for the harness's inability to look. Every
failure becomes a limitation string on the signal.
"""

from __future__ import annotations

from typing import Any

from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.core.progress_keys import (
    PROGRESS_COMPARABLE_KEY,
    PROGRESS_DELTA_KEY,
    PROGRESS_LIMITATIONS_KEY,
    PROGRESS_PREFIX,
    PROGRESS_STAGE_KEY,
    PROGRESS_STATE_KEY,
)
from agent_evals.environment.models import (
    ActionExecutionResult,
    EpisodeSnapshot,
    RewardRecord,
)
from agent_evals.environment.ports import RewardEvaluator
from agent_evals.evaluation.candidates import (
    ScientificStateProvider,
    build_candidate_artifacts,
    build_reference_artifacts,
)
from agent_evals.evaluation.progress import (
    ProgressSignal,
    ScientificProgressTracker,
    infer_stage,
)
from agent_evals.evaluation.scientific import ScientificMetricEngine
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.results import MetricStatus


class StageAwareRewardEvaluator:
    """Wrap a reward evaluator and record ``S_t`` and ``dS_t`` alongside it."""

    def __init__(
        self,
        delegate: RewardEvaluator,
        tracker: ScientificProgressTracker,
        state: ScientificStateProvider,
        metric_ids: list[str],
        engine: ScientificMetricEngine | None = None,
    ) -> None:
        self._delegate = delegate
        self._tracker = tracker
        self._state = state
        self._metric_ids = list(metric_ids)
        self._engine = engine or ScientificMetricEngine()
        self._signals: list[ProgressSignal] = []

    @property
    def signals(self) -> tuple[ProgressSignal, ...]:
        """Progress signals recorded so far, oldest first."""
        return tuple(self._signals)

    def reset(self, snapshot: EpisodeSnapshot) -> None:
        """Clear per-episode state and forward to the delegate when it has one."""
        self._signals = []
        self._tracker.reset()
        delegate_reset = getattr(self._delegate, "reset", None)
        if callable(delegate_reset):
            delegate_reset(snapshot)

    async def evaluate(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
        result: ActionExecutionResult,
    ) -> RewardRecord | None:
        """Return the delegate's reward, annotated with this step's progress."""
        reward = await self._delegate.evaluate(specification, task, snapshot, result)
        step = snapshot.state.current_step
        signal = self._record(step, result)
        self._signals.append(signal)
        if reward is None:
            return None
        return reward.model_copy(
            update={
                "metric_values": {
                    **reward.metric_values,
                    **self._reward_values(signal),
                },
                "metadata": {
                    **reward.metadata,
                    PROGRESS_STAGE_KEY: (
                        signal.stage.value if signal.stage is not None else None
                    ),
                    PROGRESS_COMPARABLE_KEY: list(signal.comparable_metrics),
                    PROGRESS_LIMITATIONS_KEY: list(signal.limitations),
                },
            }
        )

    @staticmethod
    def _reward_values(signal: ProgressSignal) -> dict[str, float]:
        """Expose only the numbers that exist, so absence stays distinguishable."""
        values: dict[str, float] = {}
        if signal.scientific_state is not None:
            values[PROGRESS_STATE_KEY] = signal.scientific_state
        if signal.delta is not None:
            values[PROGRESS_DELTA_KEY] = signal.delta
        return values

    def _record(self, step: int, result: ActionExecutionResult) -> ProgressSignal:
        """Score the current scientific state, degrading to a limitation on error."""
        delta = result.observed_state_delta
        artifact_ids = [record.artifact_id for record in result.artifacts]
        stage = infer_stage(delta, artifact_ids) if delta is not None else None
        limitations: list[str] = []
        if delta is None:
            limitations.append(
                "no observed state delta for this step, so its workflow stage is "
                "unknown"
            )
        elif stage is None:
            limitations.append(
                "nothing observed identifies this step's workflow stage, so it "
                "contributes no stage attribution"
            )
        try:
            values = self._metric_values()
        # Deliberately broad. Scoring a step is observation, and any exception a
        # metric backend raises must become a recorded limitation rather than a
        # failed episode -- the alternative fails an agent for the harness's
        # inability to look at what it produced.
        except Exception as error:
            return self._tracker.record(
                step=step,
                stage=stage,
                metric_values={},
                limitations=[
                    *limitations,
                    f"scientific state could not be scored at this step: {error}",
                ],
            )
        return self._tracker.record(
            step=step,
            stage=stage,
            metric_values=values,
            limitations=limitations,
        )

    def _metric_values(self) -> dict[str, float]:
        """Compute the metrics that can be answered against the state right now.

        Only ``COMPUTED`` results are returned. A metric that failed because its
        stage has not run yet carries a failure score, and admitting those would
        make every early step look catastrophic and every later one a recovery.
        """
        adata = self._state.adata
        if adata is None:
            return {}
        context = ScientificMetricContext(
            adata=adata,
            candidate_artifacts=build_candidate_artifacts(
                adata,
                prediction_column=self._state.agent_prediction_column(),
                cluster_column=self._state.agent_cluster_column(),
            ),
            reference_artifacts=build_reference_artifacts(adata),
            agent_produced_columns=self._agent_columns(),
        )
        results, _applicability, _groups, _score = self._engine.evaluate(
            self._metric_ids,
            context,
        )
        return {
            item.metric_id: float(item.normalized_value)
            for item in results
            if item.status is MetricStatus.SCORED and item.normalized_value is not None
        }

    def _agent_columns(self) -> frozenset[str]:
        """Read agent-written column provenance when the state exposes it."""
        columns: Any = getattr(self._state, "agent_produced_columns", None)
        if columns is None:
            return frozenset()
        return frozenset(str(column) for column in columns)


__all__ = ["PROGRESS_PREFIX", "StageAwareRewardEvaluator"]
