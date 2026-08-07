"""Execution engine for generic scientific metric adapters."""

from __future__ import annotations

from collections.abc import Sequence

from agent_evals.evaluation.metrics.base import (
    ArtifactBundle,
    EvaluationContext,
    MetricStatus,
    ReferenceBundle,
    ScientificMetricResult,
    ScoreAnchors,
)
from agent_evals.evaluation.metrics.registry import MetricRegistry, metric_registry


class ScientificMetricEngine:
    """Evaluate a frozen metric set without candidate-specific renormalization."""

    def __init__(self, registry: MetricRegistry = metric_registry) -> None:
        self.registry = registry

    def evaluate(
        self,
        metric_names: Sequence[str],
        prediction: ArtifactBundle,
        reference: ReferenceBundle,
        context: EvaluationContext,
    ) -> list[ScientificMetricResult]:
        """Return one explicit result per requested metric."""
        results: list[ScientificMetricResult] = []
        for name in metric_names:
            metric = self.registry.get(name)
            applicability = metric.applicability(context)
            if applicability.structurally_ineligible:
                results.append(
                    ScientificMetricResult(
                        metric_name=metric.name,
                        category=metric.category,
                        applicable=False,
                        role=metric.role,
                        status=MetricStatus.STRUCTURALLY_INELIGIBLE,
                        implementation_version=metric.implementation_version,
                        metadata={"reason": applicability.reason},
                    )
                )
                continue
            try:
                raw = metric.compute(prediction, reference, context)
                if raw.value is None:
                    raise ValueError(raw.metadata.get("failure_reason", "backend returned no value"))
                numeric = float(raw.value)
                normalized = metric.normalize(numeric, ScoreAnchors())
                results.append(
                    ScientificMetricResult(
                        metric_name=metric.name,
                        category=metric.category,
                        raw_value=raw.value,
                        normalized_value=normalized,
                        applicable=applicability.applicable,
                        role=metric.role,
                        status=MetricStatus.COMPUTED,
                        implementation_version=metric.implementation_version,
                        metadata={"applicability": applicability.reason, **raw.metadata},
                    )
                )
            except Exception as error:
                results.append(
                    ScientificMetricResult(
                        metric_name=metric.name,
                        category=metric.category,
                        applicable=True,
                        role=metric.role,
                        status=MetricStatus.FAILED,
                        implementation_version=metric.implementation_version,
                        metadata={
                            "reason": f"{type(error).__name__}: {error}",
                            "applicability": applicability.reason,
                        },
                    )
                )
        return results


__all__ = ["ScientificMetricEngine"]
