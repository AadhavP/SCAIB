"""Local decision and global benchmark reward calculations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.environment.models import (
    ActionExecutionResult,
    ActionIntent,
    ActionRecord,
    EpisodeSnapshot,
    RewardRecord,
)
from agent_evals.scientific.metrics import (
    aggregate_objective_score,
    compute_objective_metrics,
)


class GlobalReward(BaseModel):
    """Final benchmark reward kept separate from step-level rewards."""

    model_config = ConfigDict(extra="forbid")

    value: float | None = Field(default=None, ge=0, le=1)
    components: dict[str, float] = Field(default_factory=dict)
    metric_ids: list[str] = Field(default_factory=list)
    status: str = "succeeded"


class RewardEvaluator:
    """Compute local decision rewards and satisfy the environment reward port."""

    def __init__(self) -> None:
        self._previous_snapshot: EpisodeSnapshot | None = None

    def reset(self, snapshot: EpisodeSnapshot) -> None:
        """Set the baseline state at the start of an episode."""
        self._previous_snapshot = snapshot

    def evaluate_step(
        self,
        previous_state: EpisodeSnapshot,
        action: ActionRecord | ActionIntent | ActionExecutionResult,
        resulting_state: EpisodeSnapshot,
    ) -> RewardRecord:
        """Score one action using observable quality, retention, and execution."""
        before = self._quality(previous_state)
        after = self._quality(resulting_state)
        quality_improvement = self._improvement(before.get("mean_pct_mt"), after.get("mean_pct_mt"))
        retention = self._retention(resulting_state)
        succeeded = self._action_succeeded(action)
        score = (0.6 * quality_improvement) + (0.2 * retention) + (0.2 * succeeded)
        action_id = self._action_id(action)
        return RewardRecord(
            value=max(0.0, min(1.0, score)),
            strategy_id="local-decision",
            metric_values={
                "quality_improvement": quality_improvement,
                "cell_retention": retention,
                "execution_success": succeeded,
            },
            step=resulting_state.state.current_step,
            metadata={
                "reward_type": "local",
                "action_id": action_id,
                "before_quality": before,
                "after_quality": after,
            },
        )

    async def evaluate(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
        result: ActionExecutionResult,
    ) -> RewardRecord:
        """Adapter method consumed by ScientificEnvironment."""
        previous = self._previous_snapshot or snapshot
        action = snapshot.state.actions[-1] if snapshot.state.actions else result
        reward = self.evaluate_step(previous, action, snapshot)
        self._previous_snapshot = snapshot
        return reward

    @staticmethod
    def global_reward(
        benchmark_id: str,
        adata: Any,
        *,
        pipeline_parameters: dict[str, Any] | None = None,
    ) -> GlobalReward:
        """Compute final objective metrics without merging them into local rewards."""
        metrics = compute_objective_metrics(
            benchmark_id,
            adata,
            pipeline_parameters=pipeline_parameters,
        )
        values = {
            metric.metric_id: float(metric.normalized_score)
            for metric in metrics
            if metric.normalized_score is not None and metric.status.value == "succeeded"
        }
        final_value = aggregate_objective_score(benchmark_id, metrics)
        return GlobalReward(
            value=final_value,
            components=values,
            metric_ids=[metric.metric_id for metric in metrics],
            status="succeeded" if final_value is not None else "unavailable",
        )

    @staticmethod
    def _quality(snapshot: EpisodeSnapshot) -> dict[str, float | None]:
        for observation in snapshot.state.observations.values():
            if isinstance(observation.value, dict):
                metrics = observation.value.get("quality_metrics")
                if isinstance(metrics, dict):
                    return {
                        "mean_pct_mt": (
                            float(metrics["mean_pct_mt"])
                            if isinstance(metrics.get("mean_pct_mt"), (int, float))
                            else None
                        )
                    }
        return {"mean_pct_mt": None}

    @staticmethod
    def _improvement(before: float | None, after: float | None) -> float:
        if before is None or after is None:
            return 0.5
        if before <= 0:
            return 1.0
        return max(0.0, min(1.0, (before - after) / before + 0.5))

    @staticmethod
    def _retention(snapshot: EpisodeSnapshot) -> float:
        for artifact in reversed(snapshot.state.artifacts.values()):
            before = artifact.metadata.get("cells_before")
            after = artifact.metadata.get("cells_after")
            if isinstance(before, (int, float)) and isinstance(after, (int, float)) and before > 0:
                return max(0.0, min(1.0, float(after) / float(before)))
        return 1.0

    @staticmethod
    def _action_succeeded(action: ActionRecord | ActionIntent | ActionExecutionResult) -> float:
        if isinstance(action, ActionRecord):
            return 1.0 if action.result.status.value == "succeeded" else 0.0
        if isinstance(action, ActionExecutionResult):
            return 1.0 if action.status.value == "succeeded" else 0.0
        return 1.0

    @staticmethod
    def _action_id(action: ActionRecord | ActionIntent | ActionExecutionResult) -> str:
        return action.intent.action_id if isinstance(action, ActionRecord) else action.action_id


__all__ = ["GlobalReward", "RewardEvaluator"]
