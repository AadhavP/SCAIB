"""Generic metric registry and engine tests."""

from agent_evals.evaluation.metrics import (
    ArtifactBundle,
    EvaluationContext,
    MetricApplicability,
    MetricRole,
    MetricStatus,
    RawMetricResult,
    ReferenceBundle,
    ScientificMetric,
    ScientificMetricEngine,
    ScoreAnchors,
)
from agent_evals.evaluation.metrics.registry import MetricRegistry


class FakeMetric(ScientificMetric):
    name = "fake.metric"
    category = "test"
    direction = "maximize"
    role = MetricRole.PRIMARY
    implementation_version = "test-1"

    def applicability(self, context: EvaluationContext) -> MetricApplicability:
        return MetricApplicability(applicable="required" in context.available_artifacts, reason="test")

    def compute(self, prediction: ArtifactBundle, reference: ReferenceBundle, context: EvaluationContext) -> RawMetricResult:
        del prediction, reference, context
        return RawMetricResult(value=0.8)

    def normalize(self, value: float, anchors: ScoreAnchors) -> float:
        del anchors
        return value


def test_registry_supports_registration_search_and_engine() -> None:
    registry = MetricRegistry()
    registry.register(FakeMetric())
    assert registry.get("fake.metric").implementation_version == "test-1"
    assert registry.search(category="test")[0].name == "fake.metric"
    result = ScientificMetricEngine(registry).evaluate(
        ["fake.metric"],
        ArtifactBundle(),
        ReferenceBundle(),
        EvaluationContext(available_artifacts={"required"}),
    )[0]
    assert result.status == MetricStatus.COMPUTED
    assert result.normalized_value == 0.8
