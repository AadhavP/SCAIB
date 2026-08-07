"""Applicability behavior for generic metrics."""

from agent_evals.evaluation.metrics import (
    ArtifactBundle,
    EvaluationContext,
    MetricApplicability,
    MetricRole,
    RawMetricResult,
    ReferenceBundle,
    ScientificMetric,
    ScoreAnchors,
)
from agent_evals.evaluation.metrics.registry import MetricRegistry


class ApplicabilityMetric(ScientificMetric):
    name = "test.applicability"
    category = "test"
    direction = "maximize"
    role = MetricRole.PRIMARY
    implementation_version = "test-1"

    def applicability(self, context: EvaluationContext) -> MetricApplicability:
        return MetricApplicability(applicable="required" in context.available_artifacts, reason="test")

    def compute(
        self,
        prediction: ArtifactBundle,
        reference: ReferenceBundle,
        context: EvaluationContext,
    ) -> RawMetricResult:
        del prediction, reference, context
        return RawMetricResult(value=0.8)

    def normalize(self, value: float, anchors: ScoreAnchors) -> float:
        del anchors
        return value


def test_missing_structure_is_not_computed_as_a_candidate_failure() -> None:
    registry = MetricRegistry()
    registry.register(ApplicabilityMetric())
    result = registry.get("test.applicability").applicability(EvaluationContext())

    assert result.applicable is False
    assert result.structurally_ineligible is False
    assert "required" in result.reason or result.reason == "test"
