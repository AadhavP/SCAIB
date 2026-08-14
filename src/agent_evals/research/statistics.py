"""Reproducible statistics for agent, baseline, and ablation studies.

The benchmark's primary observations are repeated runs under fixed seeds and
paired conditions. This module intentionally uses the Python standard library so
statistics do not depend on an unpinned notebook environment. Every stochastic
operation takes an explicit seed and the resulting report records the protocol.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ReplicateStatus(StrEnum):
    """Interpretation of one replicate's result."""

    COMPLETED = "completed"
    FAILED = "failed"
    INELIGIBLE = "ineligible"


class ReplicateScore(BaseModel):
    """One score-bearing replicate with the identity needed for pairing."""

    model_config = ConfigDict(extra="forbid")

    replicate_id: str = Field(min_length=1)
    seed: int
    score: float | None = Field(default=None, ge=0, le=1)
    dimensions: dict[str, float] = Field(default_factory=dict)
    status: ReplicateStatus = ReplicateStatus.COMPLETED
    run_id: str | None = None
    cost_usd: float | None = Field(default=None, ge=0)
    wall_time_seconds: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def score_must_be_finite(cls, value: float | None) -> float | None:
        """Reject NaN because it would bypass meaningful aggregation."""
        if value is not None and not math.isfinite(value):
            raise ValueError("replicate score must be finite")
        return value

    @field_validator("dimensions")
    @classmethod
    def dimensions_must_be_finite(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject non-finite dimension values before statistical reporting."""
        for name, number in value.items():
            if not math.isfinite(number):
                raise ValueError(f"dimension '{name}' must be finite")
        return value

    @model_validator(mode="after")
    def validate_status_score(self) -> ReplicateScore:
        """Prevent failed runs from being silently treated as score zeros."""
        if self.status is ReplicateStatus.COMPLETED and self.score is None:
            raise ValueError("completed replicate must provide a score")
        if self.status is not ReplicateStatus.COMPLETED and self.score is not None:
            raise ValueError(
                "failed or ineligible replicates must omit score; record the reason "
                "in metadata instead"
            )
        return self


class ConfidenceInterval(BaseModel):
    """Bootstrap interval and the exact resampling protocol used."""

    model_config = ConfigDict(extra="forbid")

    low: float | None = None
    high: float | None = None
    confidence: float = Field(default=0.95, gt=0, lt=1)
    iterations: int = Field(default=0, ge=0)
    seed: int | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> ConfidenceInterval:
        """Reject non-finite or inverted intervals in persisted reports."""
        for name, value in (("low", self.low), ("high", self.high)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"confidence interval {name} must be finite")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("confidence interval low cannot exceed high")
        return self


class DescriptiveStatistics(BaseModel):
    """Descriptive summary of one arm or metric dimension."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    confidence_interval: ConfidenceInterval
    excluded_replicates: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_values(self) -> DescriptiveStatistics:
        """Reject non-finite descriptive values before archive serialization."""
        for name in ("mean", "median", "standard_deviation", "minimum", "maximum"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"descriptive statistic {name} must be finite")
        return self


class PairedComparison(BaseModel):
    """Paired candidate-minus-baseline comparison with uncertainty."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    baseline_id: str
    dimension: str = "score"
    n_pairs: int = Field(ge=0)
    mean_delta: float | None = None
    median_delta: float | None = None
    wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    losses: int = Field(ge=0)
    confidence_interval: ConfidenceInterval
    bootstrap_seed: int | None = None
    #: Number of sign assignments actually evaluated. For small paired samples
    #: the implementation enumerates all assignments, so this may be smaller than
    #: the requested Monte Carlo budget and must not be reported as if it were the
    #: latter.
    permutation_iterations: int = Field(default=0, ge=0)
    permutation_requested_iterations: int = Field(default=0, ge=0)
    permutation_method: str | None = None
    permutation_seed: int | None = None
    p_value: float | None = Field(default=None, ge=0, le=1)
    adjusted_p_value: float | None = Field(default=None, ge=0, le=1)
    standardized_effect: float | None = None
    missing_replicates: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pair_counts(self) -> PairedComparison:
        """Keep wins, ties, and losses consistent with the paired sample."""
        if self.wins + self.ties + self.losses != self.n_pairs:
            raise ValueError("wins + ties + losses must equal n_pairs")
        for name in (
            "mean_delta",
            "median_delta",
            "standardized_effect",
            "p_value",
            "adjusted_p_value",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"paired comparison {name} must be finite")
        return self


class StudyStatisticsReport(BaseModel):
    """Complete deterministic statistics payload for a study."""

    model_config = ConfigDict(extra="forbid")

    statistics_version: str = "1.0.0"
    study_id: str
    bootstrap_iterations: int = Field(default=2000, ge=0)
    permutation_iterations: int = Field(default=5000, ge=0)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    seed: int
    arms: dict[str, DescriptiveStatistics] = Field(default_factory=dict)
    comparisons: list[PairedComparison] = Field(default_factory=list)
    multiple_comparison_method: str = "benjamini_hochberg"
    limitations: list[str] = Field(default_factory=list)


def summarize_values(
    values: Sequence[float],
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> DescriptiveStatistics:
    """Summarize finite values and compute a seeded percentile bootstrap CI."""
    clean = [float(value) for value in values]
    if any(not math.isfinite(value) for value in clean):
        raise ValueError("statistics input contains a non-finite value")
    if not clean:
        return DescriptiveStatistics(
            n=0,
            confidence_interval=ConfidenceInterval(
                confidence=confidence,
                iterations=0,
                seed=None,
            ),
        )
    ordered = sorted(clean)
    mean = sum(clean) / len(clean)
    median = _percentile(ordered, 0.5)
    deviation = _sample_standard_deviation(clean)
    interval = bootstrap_mean_ci(
        clean,
        iterations=bootstrap_iterations,
        seed=seed,
        confidence=confidence,
    )
    return DescriptiveStatistics(
        n=len(clean),
        mean=mean,
        median=median,
        standard_deviation=deviation,
        minimum=ordered[0],
        maximum=ordered[-1],
        confidence_interval=interval,
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Return a reproducible percentile bootstrap interval for a sample mean."""
    _validate_interval_inputs(values, iterations, confidence)
    if not values:
        return ConfidenceInterval(confidence=confidence, iterations=0, seed=None)
    if iterations == 0 or len(values) == 1:
        mean = sum(values) / len(values)
        return ConfidenceInterval(
            low=mean,
            high=mean,
            confidence=confidence,
            iterations=iterations,
            seed=seed if iterations else None,
        )
    rng = random.Random(seed)
    samples = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(iterations)
    ]
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        low=_percentile(sorted(samples), alpha),
        high=_percentile(sorted(samples), 1.0 - alpha),
        confidence=confidence,
        iterations=iterations,
        seed=seed,
    )


def compare_paired(
    candidate_id: str,
    candidate: Sequence[ReplicateScore],
    baseline_id: str,
    baseline: Sequence[ReplicateScore],
    *,
    dimension: str = "score",
    bootstrap_iterations: int = 2000,
    permutation_iterations: int = 5000,
    seed: int = 0,
    confidence: float = 0.95,
) -> PairedComparison:
    """Compare arms on shared replicate IDs, never on independently ordered rows."""
    candidate_by_id = _replicate_map(candidate, candidate_id)
    baseline_by_id = _replicate_map(baseline, baseline_id)
    common = sorted(set(candidate_by_id) & set(baseline_by_id))
    missing = sorted(
        [
            f"{candidate_id}:{replicate_id}"
            for replicate_id in set(baseline_by_id) - set(candidate_by_id)
        ]
        + [
            f"{baseline_id}:{replicate_id}"
            for replicate_id in set(candidate_by_id) - set(baseline_by_id)
        ]
    )
    usable: list[str] = []
    for replicate_id in common:
        candidate_replicate = candidate_by_id[replicate_id]
        baseline_replicate = baseline_by_id[replicate_id]
        if candidate_replicate.status is not ReplicateStatus.COMPLETED:
            missing.append(
                f"{candidate_id}:{replicate_id}:{candidate_replicate.status.value}"
            )
        elif baseline_replicate.status is not ReplicateStatus.COMPLETED:
            missing.append(
                f"{baseline_id}:{replicate_id}:{baseline_replicate.status.value}"
            )
        elif candidate_replicate.seed != baseline_replicate.seed:
            missing.append(
                f"{replicate_id}:seed_mismatch:{candidate_replicate.seed}!="
                f"{baseline_replicate.seed}"
            )
        else:
            usable.append(replicate_id)
    deltas = [
        _dimension(candidate_by_id[replicate_id], dimension)
        - _dimension(baseline_by_id[replicate_id], dimension)
        for replicate_id in usable
    ]
    wins = sum(delta > 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    interval = bootstrap_mean_ci(
        deltas,
        iterations=bootstrap_iterations,
        seed=seed,
        confidence=confidence,
    )
    p_value, actual_permutations, permutation_method = _paired_sign_permutation_pvalue(
        deltas,
        iterations=permutation_iterations,
        seed=seed + 1,
    )
    return PairedComparison(
        candidate_id=candidate_id,
        baseline_id=baseline_id,
        dimension=dimension,
        n_pairs=len(deltas),
        mean_delta=(sum(deltas) / len(deltas)) if deltas else None,
        median_delta=_percentile(sorted(deltas), 0.5) if deltas else None,
        wins=wins,
        ties=ties,
        losses=losses,
        confidence_interval=interval,
        bootstrap_seed=seed,
        permutation_iterations=actual_permutations,
        permutation_requested_iterations=permutation_iterations,
        permutation_method=permutation_method,
        permutation_seed=seed + 1 if actual_permutations else None,
        p_value=p_value,
        standardized_effect=_standardized_effect(deltas),
        missing_replicates=missing,
    )


def build_statistics_report(
    study_id: str,
    arms: Mapping[str, Sequence[ReplicateScore]],
    *,
    dimensions: Sequence[str] = ("score",),
    bootstrap_iterations: int = 2000,
    permutation_iterations: int = 5000,
    seed: int = 0,
    confidence: float = 0.95,
) -> StudyStatisticsReport:
    """Build arm summaries and all pairwise paired comparisons."""
    if not dimensions:
        raise ValueError("statistics requires at least one measured dimension")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("statistics dimensions must be unique")
    if any(not dimension.strip() for dimension in dimensions):
        raise ValueError("statistics dimension names cannot be empty")
    arm_summaries: dict[str, DescriptiveStatistics] = {}
    limitations: list[str] = []
    for index, (arm_id, replicates) in enumerate(sorted(arms.items())):
        completed = [
            replicate
            for replicate in replicates
            if replicate.status is ReplicateStatus.COMPLETED
        ]
        excluded_count = len(replicates) - len(completed)
        if excluded_count:
            limitations.append(
                f"arm '{arm_id}' excluded {excluded_count} failed/ineligible replicate(s) "
                "from score summaries"
            )
        summary = summarize_values(
            [_dimension(replicate, "score") for replicate in completed],
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + index,
            confidence=confidence,
        )
        arm_summaries[arm_id] = summary.model_copy(
            update={"excluded_replicates": excluded_count}
        )
    comparisons: list[PairedComparison] = []
    for candidate_id, baseline_id in itertools.combinations(sorted(arms), 2):
        for index, dimension in enumerate(dimensions):
            comparisons.append(
                compare_paired(
                    candidate_id,
                    arms[candidate_id],
                    baseline_id,
                    arms[baseline_id],
                    dimension=dimension,
                    bootstrap_iterations=bootstrap_iterations,
                    permutation_iterations=permutation_iterations,
                    seed=seed + index,
                    confidence=confidence,
                )
            )
    adjusted = benjamini_hochberg(
        [comparison.p_value for comparison in comparisons]
    )
    comparisons = [
        comparison.model_copy(update={"adjusted_p_value": adjusted[index]})
        for index, comparison in enumerate(comparisons)
    ]
    if any(comparison.n_pairs == 0 for comparison in comparisons):
        limitations.append("one or more arm comparisons had no shared replicate IDs")
    if any(summary.n < 2 for summary in arm_summaries.values()):
        limitations.append("one or more arms had fewer than two replicates")
    return StudyStatisticsReport(
        study_id=study_id,
        bootstrap_iterations=bootstrap_iterations,
        permutation_iterations=permutation_iterations,
        confidence=confidence,
        seed=seed,
        arms=arm_summaries,
        comparisons=comparisons,
        limitations=list(dict.fromkeys(limitations)),
    )


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    """Adjust p-values while preserving input order and missingness.

    Invalid p-values are rejected instead of clipped. Clipping would turn a
    malformed evaluator result into apparently valid inferential evidence.
    """
    for value in p_values:
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Benjamini-Hochberg p-values must be finite and in [0, 1]")
    indexed = sorted(
        ((index, value) for index, value in enumerate(p_values) if value is not None),
        key=lambda item: item[1],
    )
    adjusted: list[float | None] = [None] * len(p_values)
    running = 1.0
    count = len(indexed)
    for rank in range(count, 0, -1):
        index, value = indexed[rank - 1]
        running = min(running, float(value) * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def _replicate_map(
    replicates: Sequence[ReplicateScore], arm_id: str
) -> dict[str, ReplicateScore]:
    """Index replicates and reject duplicate pairing identities."""
    result: dict[str, ReplicateScore] = {}
    for replicate in replicates:
        if replicate.replicate_id in result:
            raise ValueError(
                f"arm '{arm_id}' contains duplicate replicate_id "
                f"'{replicate.replicate_id}'"
            )
        result[replicate.replicate_id] = replicate
    return result


def _dimension(replicate: ReplicateScore, dimension: str) -> float:
    """Read a score or named dimension from a completed replicate only."""
    if replicate.status is not ReplicateStatus.COMPLETED:
        raise ValueError(
            f"replicate '{replicate.replicate_id}' is {replicate.status.value}, "
            "not score-bearing"
        )
    if dimension == "score":
        if replicate.score is None:
            raise ValueError(f"replicate '{replicate.replicate_id}' has no score")
        return replicate.score
    try:
        return replicate.dimensions[dimension]
    except KeyError as error:
        raise ValueError(
            f"replicate '{replicate.replicate_id}' has no dimension '{dimension}'"
        ) from error


def _paired_sign_permutation_pvalue(
    deltas: Sequence[float], *, iterations: int, seed: int
) -> tuple[float | None, int, str | None]:
    """Return a two-sided sign-flip p-value and its actual protocol.

    Small samples use exact enumeration. Larger samples use the requested seeded
    Monte Carlo count. Returning the method and actual count prevents a report
    from claiming a 5,000-draw test when it actually evaluated all 2**n signs.
    """
    if not deltas:
        return None, 0, "unmeasured"
    observed = abs(sum(deltas) / len(deltas))
    if observed == 0:
        return 1.0, 0, "degenerate_zero_mean"
    if iterations <= 0:
        return None, 0, "disabled"
    if len(deltas) <= 14:
        sign_vectors = itertools.product((-1.0, 1.0), repeat=len(deltas))
        values = [
            abs(
                sum(delta * sign for delta, sign in zip(deltas, sign_vector, strict=True))
                / len(deltas)
            )
            for sign_vector in sign_vectors
        ]
        method = "exact_sign_flip"
    else:
        rng = random.Random(seed)
        values = [
            abs(
                sum(
                    delta * (1.0 if rng.random() >= 0.5 else -1.0)
                    for delta in deltas
                )
                / len(deltas)
            )
            for _ in range(iterations)
        ]
        method = "monte_carlo_sign_flip"
    exceed = sum(value >= observed for value in values)
    return (exceed + 1) / (len(values) + 1), len(values), method


def _standardized_effect(deltas: Sequence[float]) -> float | None:
    """Return paired Cohen's dz, or None when variation is undefined."""
    if not deltas:
        return None
    deviation = _sample_standard_deviation(deltas)
    if deviation is None or deviation == 0:
        return None
    return (sum(deltas) / len(deltas)) / deviation


def _sample_standard_deviation(values: Sequence[float]) -> float | None:
    """Calculate sample standard deviation without a statistics-version drift."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile for a sorted non-empty sequence."""
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = max(0.0, min(1.0, fraction)) * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] + (values[upper] - values[lower]) * weight)


def _validate_interval_inputs(
    values: Sequence[float], iterations: int, confidence: float
) -> None:
    """Validate common bootstrap arguments."""
    if iterations < 0:
        raise ValueError("bootstrap iterations cannot be negative")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("bootstrap input contains a non-finite value")


__all__ = [
    "ConfidenceInterval",
    "DescriptiveStatistics",
    "PairedComparison",
    "ReplicateScore",
    "ReplicateStatus",
    "StudyStatisticsReport",
    "benjamini_hochberg",
    "bootstrap_mean_ci",
    "build_statistics_report",
    "compare_paired",
    "summarize_values",
]
