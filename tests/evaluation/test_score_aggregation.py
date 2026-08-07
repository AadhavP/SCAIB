"""Domain geometric aggregation tests."""

from agent_evals.evaluation.profiles.base import MetricGroupProfile, MetricProfileEntry
from agent_evals.evaluation.scoring import (
    MetricScoreInput,
    WeightedGeometricAggregator,
    aggregate_domains,
)


def test_geometric_aggregation_keeps_failed_weight_and_excludes_structural() -> None:
    profile = MetricGroupProfile(
        weight=1,
        metrics={
            "good": MetricProfileEntry(weight=0.5),
            "failed": MetricProfileEntry(weight=0.5),
            "not_applicable": MetricProfileEntry(weight=1, required=False),
        },
    )
    domain = WeightedGeometricAggregator().aggregate(
        "biology",
        profile,
        [
            MetricScoreInput(name="good", value=0.8),
            MetricScoreInput(name="failed", value=0, status="failed"),
            MetricScoreInput(name="not_applicable", structurally_ineligible=True),
        ],
    )
    score = aggregate_domains([domain])

    assert domain.value == 0.0
    assert "not_applicable" in domain.excluded_metrics
    assert score.value == 0.0


def test_external_profile_score_is_aggregated() -> None:
    profile = MetricGroupProfile(weight=0.2, external_score="robustness.seed_stability")
    domain = WeightedGeometricAggregator().aggregate(
        "robustness",
        profile,
        [MetricScoreInput(name="robustness.seed_stability", value=0.88)],
    )

    assert domain.value == 0.88
    assert domain.included_metrics == ["robustness.seed_stability"]
