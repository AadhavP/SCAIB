"""Human calibration of observable decision-quality scores.

Decision quality cannot be made research-grade by choosing a rubric in code. It
needs independent expert ratings, an explicit agreement protocol, and evidence
that the automated score tracks those ratings. This module computes the
recordable, deterministic part of that study without pretending that a model
judge is an expert.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExpertRating(BaseModel):
    """One independent expert's rating of one observable decision case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    rater_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    rubric_version: str = Field(min_length=1)
    adjudicated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        """Reject NaN and infinity even though they fit the numeric range."""
        if not math.isfinite(value):
            raise ValueError("expert rating must be finite")
        return value


class DecisionPrediction(BaseModel):
    """Automated decision score to be calibrated against expert ratings."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    score_version: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float) -> float:
        """Reject non-finite predictions before fitting calibration."""
        if not math.isfinite(value):
            raise ValueError("decision prediction must be finite")
        return value


class CalibrationProtocol(BaseModel):
    """Frozen acceptance criteria for an expert calibration study."""

    model_config = ConfigDict(extra="forbid")

    protocol_version: str = "1.0.0"
    minimum_cases: int = Field(default=30, ge=1)
    minimum_raters: int = Field(default=2, ge=2)
    agreement_tolerance: float = Field(default=0.1, ge=0, le=1)
    minimum_agreement_fraction: float = Field(default=0.8, ge=0, le=1)
    minimum_icc: float = Field(default=0.75, ge=-1, le=1)
    minimum_prediction_correlation: float = Field(default=0.7, ge=-1, le=1)
    require_same_rubric: bool = True


class CalibrationReport(BaseModel):
    """Human-calibration result and limitations."""

    model_config = ConfigDict(extra="forbid")

    calibration_version: str = "1.0.0"
    protocol: CalibrationProtocol
    cases_rated: int = Field(ge=0)
    complete_cases: int = Field(ge=0)
    raters: int = Field(ge=0)
    ratings: int = Field(ge=0)
    mean_rater_disagreement: float | None = None
    agreement_fraction: float | None = None
    intraclass_correlation: float | None = None
    expert_prediction_correlation: float | None = None
    mean_absolute_error: float | None = None
    mean_bias: float | None = None
    calibration_slope: float | None = None
    calibration_intercept: float | None = None
    limitations: list[str] = Field(default_factory=list)

    @property
    def research_ready(self) -> bool:
        """Whether the declared calibration protocol was actually met."""
        return (
            self.complete_cases >= self.protocol.minimum_cases
            and self.raters >= self.protocol.minimum_raters
            and self.agreement_fraction is not None
            and self.agreement_fraction >= self.protocol.minimum_agreement_fraction
            and self.intraclass_correlation is not None
            and self.intraclass_correlation >= self.protocol.minimum_icc
            and self.expert_prediction_correlation is not None
            and self.expert_prediction_correlation >= self.protocol.minimum_prediction_correlation
            and not self.limitations
        )


def build_calibration_report(  # noqa: C901
    ratings: Sequence[ExpertRating],
    predictions: Sequence[DecisionPrediction],
    *,
    protocol: CalibrationProtocol | None = None,
) -> CalibrationReport:
    """Compute agreement and prediction calibration against independent ratings."""
    resolved_protocol = protocol or CalibrationProtocol()
    grouped: dict[str, list[ExpertRating]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for rating in ratings:
        key = (rating.case_id, rating.rater_id)
        if key in seen:
            raise ValueError(
                f"duplicate expert rating for case '{rating.case_id}' and rater '{rating.rater_id}'"
            )
        seen.add(key)
        grouped[rating.case_id].append(rating)
    prediction_map: dict[str, DecisionPrediction] = {}
    for prediction in predictions:
        if prediction.case_id in prediction_map:
            raise ValueError(f"duplicate decision prediction for case '{prediction.case_id}'")
        prediction_map[prediction.case_id] = prediction
    raters = {rating.rater_id for rating in ratings}
    complete = {
        case_id: case_ratings
        for case_id, case_ratings in grouped.items()
        if len({rating.rater_id for rating in case_ratings}) >= resolved_protocol.minimum_raters
    }
    limitations: list[str] = []
    rubric_versions = {rating.rubric_version for rating in ratings}
    if resolved_protocol.require_same_rubric and len(rubric_versions) > 1:
        limitations.append("expert ratings used more than one rubric version")
    missing_predictions = sorted(set(complete) - set(prediction_map))
    if missing_predictions:
        limitations.append(
            "automated predictions are missing for complete cases: "
            + ", ".join(missing_predictions)
        )
    pairwise_differences = [
        abs(left.score - right.score)
        for case_ratings in complete.values()
        for left, right in itertools.combinations(case_ratings, 2)
    ]
    agreement_fraction = (
        sum(difference <= resolved_protocol.agreement_tolerance for difference in pairwise_differences)
        / len(pairwise_differences)
        if pairwise_differences
        else None
    )
    expert_means = {
        case_id: sum(rating.score for rating in case_ratings) / len(case_ratings)
        for case_id, case_ratings in complete.items()
    }
    paired = [
        (prediction_map[case_id].score, expert_means[case_id])
        for case_id in sorted(expert_means)
        if case_id in prediction_map
    ]
    predictions_only = [item[0] for item in paired]
    expert_only = [item[1] for item in paired]
    correlation = _pearson(predictions_only, expert_only)
    icc = _intraclass_correlation(complete)
    mae = (
        sum(abs(predicted - expert) for predicted, expert in paired) / len(paired)
        if paired
        else None
    )
    bias = (
        sum(predicted - expert for predicted, expert in paired) / len(paired)
        if paired
        else None
    )
    slope, intercept = _linear_fit(predictions_only, expert_only)
    if len(complete) < resolved_protocol.minimum_cases:
        limitations.append(
            f"only {len(complete)} complete cases; requires {resolved_protocol.minimum_cases}"
        )
    if len(raters) < resolved_protocol.minimum_raters:
        limitations.append(
            f"only {len(raters)} independent rater(s); requires {resolved_protocol.minimum_raters}"
        )
    if agreement_fraction is None:
        limitations.append("inter-rater agreement was unmeasured")
    elif agreement_fraction < resolved_protocol.minimum_agreement_fraction:
        limitations.append(
            f"agreement fraction {agreement_fraction:.3f} is below the protocol "
            f"threshold {resolved_protocol.minimum_agreement_fraction:.3f}"
        )
    if icc is None:
        limitations.append("inter-rater ICC was unmeasured")
    elif icc < resolved_protocol.minimum_icc:
        limitations.append(
            f"inter-rater ICC {icc:.3f} is below the protocol threshold "
            f"{resolved_protocol.minimum_icc:.3f}"
        )
    if correlation is None:
        limitations.append("expert-to-automated score correlation was unmeasured")
    elif correlation < resolved_protocol.minimum_prediction_correlation:
        limitations.append(
            f"expert-to-automated correlation {correlation:.3f} is below the "
            f"protocol threshold {resolved_protocol.minimum_prediction_correlation:.3f}"
        )
    return CalibrationReport(
        protocol=resolved_protocol,
        cases_rated=len(grouped),
        complete_cases=len(complete),
        raters=len(raters),
        ratings=len(ratings),
        mean_rater_disagreement=(
            sum(pairwise_differences) / len(pairwise_differences)
            if pairwise_differences
            else None
        ),
        agreement_fraction=agreement_fraction,
        intraclass_correlation=icc,
        expert_prediction_correlation=correlation,
        mean_absolute_error=mae,
        mean_bias=bias,
        calibration_slope=slope,
        calibration_intercept=intercept,
        limitations=list(dict.fromkeys(limitations)),
    )


def _intraclass_correlation(
    complete: Mapping[str, Sequence[ExpertRating]],
) -> float | None:
    """Calculate ICC(2,1), absolute agreement, for balanced complete cases.

    ICC is reported alongside the transparent tolerance agreement fraction because
    pairwise agreement alone can hide systematic rater offsets. The implementation
    uses the two-way random-effects absolute-agreement form and returns ``None``
    when a balanced variance estimate cannot be formed.
    """
    rows = [list(values) for values in complete.values()]
    if len(rows) < 2:
        return None
    rater_ids = sorted({rating.rater_id for values in rows for rating in values})
    if len(rater_ids) < 2 or any(
        {rating.rater_id for rating in values} != set(rater_ids) for values in rows
    ):
        return None
    matrix = [
        [next(rating.score for rating in values if rating.rater_id == rater_id) for rater_id in rater_ids]
        for values in rows
    ]
    n = len(matrix)
    k = len(rater_ids)
    grand = sum(value for row in matrix for value in row) / (n * k)
    row_means = [sum(row) / k for row in matrix]
    column_means = [sum(matrix[row][column] for row in range(n)) / n for column in range(k)]
    ss_subject = k * sum((mean - grand) ** 2 for mean in row_means)
    ss_rater = n * sum((mean - grand) ** 2 for mean in column_means)
    ss_total = sum((value - grand) ** 2 for row in matrix for value in row)
    ss_error = max(0.0, ss_total - ss_subject - ss_rater)
    ms_subject = ss_subject / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_subject + (k - 1) * ms_error + k * (ms_rater - ms_error) / n
    if denominator == 0:
        return None
    return (ms_subject - ms_error) / denominator


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Calculate Pearson correlation, returning None for undefined variance."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_norm = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return numerator / (left_norm * right_norm)


def _linear_fit(
    predicted: Sequence[float], expert: Sequence[float]
) -> tuple[float | None, float | None]:
    """Fit expert score = slope * automated score + intercept."""
    if len(predicted) != len(expert) or len(predicted) < 2:
        return None, None
    mean_predicted = sum(predicted) / len(predicted)
    mean_expert = sum(expert) / len(expert)
    denominator = sum((value - mean_predicted) ** 2 for value in predicted)
    if denominator == 0:
        return None, None
    slope = sum(
        (left - mean_predicted) * (right - mean_expert)
        for left, right in zip(predicted, expert, strict=True)
    ) / denominator
    return slope, mean_expert - slope * mean_predicted


__all__ = [
    "CalibrationProtocol",
    "CalibrationReport",
    "DecisionPrediction",
    "ExpertRating",
    "build_calibration_report",
]
