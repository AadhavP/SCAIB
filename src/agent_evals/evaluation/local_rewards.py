"""Step-level scientific decision rewards from observable evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents.trajectory import DecisionCategory, ScientificDecision


class LocalDecisionReward(BaseModel):
    """Transparent reward decomposition for one structured decision."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    category: DecisionCategory
    value: float = Field(ge=0, le=1)
    components: dict[str, float] = Field(default_factory=dict)
    formula: str
    evidence: list[str] = Field(default_factory=list)


class LocalRewardEvaluator:
    """Evaluate local decisions without using a learned or causal judge."""

    _WEIGHTS: ClassVar[dict[DecisionCategory, tuple[tuple[str, float], ...]]] = {
        DecisionCategory.QC_STRATEGY: (
            ("artifact_removal", 0.4),
            ("biological_retention", 0.3),
            ("rare_population_preservation", 0.3),
        ),
        DecisionCategory.INTEGRATION: (
            ("batch_removal", 0.5),
            ("biology_preservation", 0.5),
        ),
        DecisionCategory.CLUSTERING: (
            ("ari", 0.5),
            ("rare_cell_recovery", 0.3),
            ("stability", 0.2),
        ),
    }

    def evaluate(
        self,
        decision: ScientificDecision,
        observation_before: Mapping[str, Any] | None,
        observation_after: Mapping[str, Any] | None,
        downstream_metrics: Mapping[str, float] | None,
    ) -> LocalDecisionReward:
        """Compute a weighted local reward from before/after and metric evidence."""
        del observation_before, observation_after
        metrics = downstream_metrics or {}
        weights = self._WEIGHTS.get(decision.decision_category)
        if weights is None:
            if "decision_local_reward" in metrics:
                fallback = max(0.0, min(1.0, float(metrics["decision_local_reward"])))
                return LocalDecisionReward(
                    decision_id=decision.decision_id,
                    category=decision.decision_category,
                    value=fallback,
                    components={"decision_local_reward": fallback},
                    formula="decision_local_reward",
                    evidence=["category-specific evidence was unavailable; environment local reward used"],
                )
            weight = 1.0 / len(metrics) if metrics else 1.0
            weights = tuple((key, weight) for key in metrics)
        if not weights:
            value = 1.0 if decision.execution_status is not None and decision.execution_status.value == "succeeded" else 0.0
            return LocalDecisionReward(
                decision_id=decision.decision_id,
                category=decision.decision_category,
                value=value,
                formula="execution_success",
                evidence=["no category-specific downstream metrics were supplied"],
            )
        if not any(name in metrics for name, _weight in weights) and "decision_local_reward" in metrics:
            fallback = max(0.0, min(1.0, float(metrics["decision_local_reward"])))
            return LocalDecisionReward(
                decision_id=decision.decision_id,
                category=decision.decision_category,
                value=fallback,
                components={"decision_local_reward": fallback},
                formula="decision_local_reward",
                evidence=["category-specific evidence was unavailable; environment local reward used"],
            )
        components = {
            name: max(0.0, min(1.0, float(metrics.get(name, 0.0))))
            for name, _weight in weights
        }
        value = sum(components[name] * weight for name, weight in weights)
        formula = " + ".join(f"{weight:g}*{name}" for name, weight in weights)
        return LocalDecisionReward(
            decision_id=decision.decision_id,
            category=decision.decision_category,
            value=value,
            components=components,
            formula=formula,
            evidence=[f"component '{name}'={components[name]:.3f}" for name, _ in weights],
        )


__all__ = ["LocalDecisionReward", "LocalRewardEvaluator"]
