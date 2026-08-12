"""Step-level scientific decision rewards from observable evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents.trajectory import DecisionCategory, ScientificDecision
from agent_evals.core.decision_components import (
    OBSERVED_CELL_COUNT,
    OBSERVED_COMPONENTS,
    removed_fraction,
    resolve_metric_component,
)


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
        """Compute a weighted local reward from before/after and metric evidence.

        ``observation_before`` and ``observation_after`` report the observed cell
        and gene counts around this decision, keyed by ``OBSERVED_CELL_COUNT`` and
        ``OBSERVED_GENE_COUNT``. Either being ``None`` means nobody looked, which
        excludes the components derived from them rather than scoring them zero.

        ``downstream_metrics`` may be keyed by component name or by dotted metric
        id; the latter is what the metric registry actually produces, and failing
        to accept it is why these components went unreceived.
        """
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
        resolved = self._resolve_components(
            weights,
            metrics,
            observation_before,
            observation_after,
        )
        if not resolved and "decision_local_reward" in metrics:
            fallback = max(0.0, min(1.0, float(metrics["decision_local_reward"])))
            return LocalDecisionReward(
                decision_id=decision.decision_id,
                category=decision.decision_category,
                value=fallback,
                components={"decision_local_reward": fallback},
                formula="decision_local_reward",
                evidence=["category-specific evidence was unavailable; environment local reward used"],
            )
        # Renormalize over what was answerable instead of defaulting an absent
        # component to zero. Under the old behaviour a clustering decision taken
        # before any reference-scored metric could be computed lost 0.5 of its
        # reward to the harness having nothing to look at yet.
        total_weight = sum(weight for _value, weight, _source in resolved.values())
        if not resolved or total_weight <= 0:
            return LocalDecisionReward(
                decision_id=decision.decision_id,
                category=decision.decision_category,
                value=0.0,
                formula="no component of this decision category was answerable",
                evidence=[
                    f"component '{name}' had no observed or metric source"
                    for name, _weight in weights
                ],
            )
        components = {name: item[0] for name, item in resolved.items()}
        value = sum(
            item[0] * item[1] / total_weight for item in resolved.values()
        )
        formula = " + ".join(
            f"{weight / total_weight:g}*{name}"
            for name, (_value, weight, _source) in resolved.items()
        )
        evidence = [
            f"component '{name}'={value_:.3f} from {source}"
            for name, (value_, _weight, source) in resolved.items()
        ]
        missing = [
            name for name, _weight in weights if name not in resolved
        ]
        if missing:
            evidence.append(
                "component(s) excluded as unmeasured rather than scored zero: "
                f"{', '.join(missing)}"
            )
        return LocalDecisionReward(
            decision_id=decision.decision_id,
            category=decision.decision_category,
            value=max(0.0, min(1.0, value)),
            components=components,
            formula=formula,
            evidence=evidence,
        )

    @staticmethod
    def _resolve_components(
        weights: tuple[tuple[str, float], ...],
        metrics: Mapping[str, float],
        observation_before: Mapping[str, Any] | None,
        observation_after: Mapping[str, Any] | None,
    ) -> dict[str, tuple[float, float, str]]:
        """Find a value, weight, and source for every component that has one.

        Observed components consult the before/after state first, because that is
        the harness's own measurement; supplied evidence is only a fallback for a
        caller that computed the same quantity itself.
        """
        resolved: dict[str, tuple[float, float, str]] = {}
        for name, weight in weights:
            if name in OBSERVED_COMPONENTS:
                observed = removed_fraction(
                    observation_before,
                    observation_after,
                    OBSERVED_CELL_COUNT,
                )
                if observed is not None:
                    resolved[name] = (observed, weight, "observed cell counts")
                    continue
            found = resolve_metric_component(name, metrics)
            if found is not None:
                resolved[name] = (
                    max(0.0, min(1.0, found[0])),
                    weight,
                    found[1],
                )
        return resolved


__all__ = ["LocalDecisionReward", "LocalRewardEvaluator"]
