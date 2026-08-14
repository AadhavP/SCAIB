"""Human calibration and metric validation study tests."""

from __future__ import annotations

import pytest

from agent_evals.research.calibration import (
    CalibrationProtocol,
    DecisionPrediction,
    ExpertRating,
    build_calibration_report,
)
from agent_evals.research.sensitivity import (
    MetricCorrelationVector,
    MetricSensitivityObservation,
    build_metric_validation_study,
)


def _ratings() -> list[ExpertRating]:
    return [
        ExpertRating(case_id=f"case-{index}", rater_id="r1", score=score, rubric_version="1")
        for index, score in enumerate((0.2, 0.4, 0.6, 0.8), start=1)
    ] + [
        ExpertRating(case_id=f"case-{index}", rater_id="r2", score=score, rubric_version="1")
        for index, score in enumerate((0.2, 0.5, 0.6, 0.8), start=1)
    ]


def test_calibration_reports_agreement_and_prediction_fit() -> None:
    report = build_calibration_report(
        _ratings(),
        [
            DecisionPrediction(case_id=f"case-{index}", score=score, score_version="2")
            for index, score in enumerate((0.2, 0.45, 0.6, 0.8), start=1)
        ],
        protocol=CalibrationProtocol(
            minimum_cases=4,
            minimum_raters=2,
            minimum_agreement_fraction=0.75,
        ),
    )

    assert report.research_ready is True
    assert report.agreement_fraction == 1.0
    assert report.expert_prediction_correlation is not None
    assert report.mean_absolute_error is not None


def test_calibration_does_not_call_one_rater_or_one_case_research_ready() -> None:
    report = build_calibration_report(
        _ratings()[:2],
        [DecisionPrediction(case_id="case-1", score=0.2, score_version="2")],
        protocol=CalibrationProtocol(minimum_cases=2, minimum_raters=2),
    )

    assert report.research_ready is False
    assert any("complete cases" in limitation for limitation in report.limitations)


def test_metric_validation_study_exposes_range_and_redundant_metrics() -> None:
    observations = [
        MetricSensitivityObservation(metric_id="m1", configuration_id="a", value=0.2),
        MetricSensitivityObservation(metric_id="m1", configuration_id="b", value=0.8),
    ]
    report = build_metric_validation_study(
        observations,
        [
            MetricCorrelationVector(metric_id="m1", values={"1": 0.2, "2": 0.8}),
            MetricCorrelationVector(metric_id="m2", values={"1": 0.3, "2": 0.9}),
        ],
    )

    assert report.sensitivity[0].range == pytest.approx(0.6)
    assert report.correlations[0].pearson == pytest.approx(1.0)
    assert report.correlations[0].high_correlation is True
    assert report.passed is False
    assert report.sensitivity[0].limitations
