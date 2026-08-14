"""Regression tests for metric evidence/status invariants."""

from __future__ import annotations

import math

from agent_evals.metrics.models import MetricDirection, MetricRole
from agent_evals.metrics.results import MetricResult, MetricStatus
from agent_evals.research.golden import validate_metric_result


def _result(
    status: MetricStatus,
    *,
    eligible: bool = False,
    normalized: float | None = None,
) -> MetricResult:
    return MetricResult(
        metric_id="toy.metric",
        version="1.0",
        metric_name="Toy metric",
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_value=normalized,
        normalized_value=normalized,
        eligible=eligible,
        status=status,
        eligibility_reason="fixture",
    )


def test_non_scored_statuses_cannot_carry_a_hidden_zero_or_other_score() -> None:
    for status in (
        MetricStatus.INELIGIBLE,
        MetricStatus.UNIMPLEMENTED,
        MetricStatus.MISSING,
        MetricStatus.MALFORMED,
        MetricStatus.EVALUATOR_ERROR,
        MetricStatus.FAILED,
    ):
        findings = validate_metric_result(_result(status, eligible=True, normalized=0.0))
        assert any("carries a score" in finding for finding in findings)
        assert any("marked eligible" in finding for finding in findings)


def test_scored_metrics_require_eligible_finite_normalized_values() -> None:
    missing = validate_metric_result(_result(MetricStatus.SCORED, eligible=True))
    assert any("no normalized_value" in finding for finding in missing)

    ineligible = validate_metric_result(_result(MetricStatus.SCORED, eligible=False, normalized=0.5))
    assert any("marked ineligible" in finding for finding in ineligible)

    non_finite = validate_metric_result(
        _result(MetricStatus.SCORED, eligible=True, normalized=0.5).model_copy(
            update={"raw_value": math.inf}
        )
    )
    assert any("raw_value is not finite" in finding for finding in non_finite)


def test_a_valid_scored_metric_has_no_universal_invariant_findings() -> None:
    result = _result(MetricStatus.SCORED, eligible=True, normalized=0.75)

    assert validate_metric_result(result) == []
