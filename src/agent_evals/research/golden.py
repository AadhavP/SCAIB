"""Golden fixtures and invariant checks for scientific metric implementations.

A metric implementation is not research-grade because it returns a number once.
This module provides a small, dependency-free harness for expected-value fixtures,
status semantics, bounded normalization, deterministic reruns, and monotonicity
or equality invariants. The fixture inputs remain opaque dictionaries so the
same harness can validate sklearn, scIB, and project-specific backends.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_evals.metrics.models import MetricDirection, MetricRole
from agent_evals.metrics.results import MetricResult, MetricStatus

MetricComputer = Callable[[Mapping[str, Any]], MetricResult | float | None]


class InvariantRelation(StrEnum):
    """Supported pairwise properties of a metric fixture."""

    EQUAL = "equal"
    NON_DECREASING = "non_decreasing"
    NON_INCREASING = "non_increasing"


class GoldenMetricCase(BaseModel):
    """One input and expected output for a metric computer."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected_normalized: float | None = Field(default=None, ge=0, le=1)
    expected_raw: float | None = None
    expected_status: MetricStatus = MetricStatus.SCORED
    tolerance: float = Field(default=1e-8, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_raw")
    @classmethod
    def expected_raw_must_be_finite(cls, value: float | None) -> float | None:
        """Reject a golden value that could never be compared reproducibly."""
        if value is not None and not math.isfinite(value):
            raise ValueError("expected_raw must be finite")
        return value


class MetricInvariantCase(BaseModel):
    """Pairwise invariant over two named golden cases."""

    model_config = ConfigDict(extra="forbid")

    invariant_id: str = Field(min_length=1)
    left_case_id: str = Field(min_length=1)
    right_case_id: str = Field(min_length=1)
    relation: InvariantRelation
    tolerance: float = Field(default=1e-8, ge=0)
    description: str = ""


class GoldenCaseResult(BaseModel):
    """Auditable result of one golden fixture execution."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    passed: bool
    observed_normalized: float | None = None
    observed_raw: float | None = None
    observed_status: MetricStatus | None = None
    deterministic: bool
    findings: list[str] = Field(default_factory=list)


class InvariantResult(BaseModel):
    """Auditable result of one invariant comparison."""

    model_config = ConfigDict(extra="forbid")

    invariant_id: str
    passed: bool
    left_value: float | None = None
    right_value: float | None = None
    findings: list[str] = Field(default_factory=list)


class MetricValidationReport(BaseModel):
    """Complete validation report for one metric implementation."""

    model_config = ConfigDict(extra="forbid")

    validation_version: str = "1.0.0"
    metric_id: str
    cases: list[GoldenCaseResult] = Field(default_factory=list)
    invariants: list[InvariantResult] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether a non-empty fixture suite and every invariant passed."""
        # An empty suite must never certify an implementation: ``all([])`` is
        # mathematically true but is not evidence that the metric was measured.
        return bool(self.cases) and all(item.passed for item in self.cases) and all(
            item.passed for item in self.invariants
        ) and not self.limitations


def validate_metric_result(result: MetricResult) -> list[str]:  # noqa: C901
    """Check universal metric result invariants independent of scientific domain."""
    findings: list[str] = []
    if result.status is MetricStatus.SCORED:
        if result.normalized_value is None:
            findings.append("scored result has no normalized_value")
        elif not math.isfinite(result.normalized_value):
            findings.append("normalized_value is not finite")
        elif not 0 <= result.normalized_value <= 1:
            findings.append("normalized_value is outside [0, 1]")
        if not result.eligible:
            findings.append("scored result is marked ineligible")
    elif result.status in {
        MetricStatus.INELIGIBLE,
        MetricStatus.UNIMPLEMENTED,
        MetricStatus.MISSING,
        MetricStatus.MALFORMED,
        MetricStatus.EVALUATOR_ERROR,
        MetricStatus.FAILED,
    }:
        if result.normalized_value is not None:
            findings.append(
                f"{result.status.value} result carries a score and could be mistaken for evidence"
            )
        if result.eligible:
            findings.append(f"{result.status.value} result is marked eligible")
    for name, value in (("raw_value", result.raw_value), ("normalized_value", result.normalized_value)):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
            findings.append(f"{name} is not finite")
    return findings


def run_golden_metric_suite(  # noqa: C901
    metric_id: str,
    cases: Sequence[GoldenMetricCase],
    computer: MetricComputer,
    *,
    invariants: Sequence[MetricInvariantCase] = (),
) -> MetricValidationReport:
    """Execute fixtures twice, compare expected values, then check invariants."""
    case_results: list[GoldenCaseResult] = []
    observed: dict[str, float | None] = {}
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise ValueError(f"duplicate golden case id '{case.case_id}'")
        seen.add(case.case_id)
        findings: list[str] = []
        try:
            first = _coerce_output(computer(case.inputs), metric_id)
            second = _coerce_output(computer(case.inputs), metric_id)
            findings.extend(validate_metric_result(first))
            findings.extend(validate_metric_result(second))
            if first.metric_id != metric_id or second.metric_id != metric_id:
                findings.append(
                    f"metric computer returned unexpected metric_id(s): "
                    f"{first.metric_id!r}, {second.metric_id!r}"
                )
            deterministic = (
                first.status is second.status
                and _close(first.normalized_value, second.normalized_value, case.tolerance)
                and _close_any(first.raw_value, second.raw_value, case.tolerance)
            )
            if not deterministic:
                findings.append("repeated computation was not deterministic")
            if first.status is not case.expected_status:
                findings.append(
                    f"expected status {case.expected_status.value}, got {first.status.value}"
                )
            if case.expected_normalized is not None and not _close(
                first.normalized_value, case.expected_normalized, case.tolerance
            ):
                findings.append(
                    f"expected normalized value {case.expected_normalized}, "
                    f"got {first.normalized_value}"
                )
            if case.expected_raw is not None and not _close(
                first.raw_value, case.expected_raw, case.tolerance
            ):
                findings.append(f"expected raw value {case.expected_raw}, got {first.raw_value}")
            observed[case.case_id] = first.normalized_value
            case_results.append(
                GoldenCaseResult(
                    case_id=case.case_id,
                    passed=not findings,
                    observed_normalized=first.normalized_value,
                    observed_raw=_numeric_or_none(first.raw_value),
                    observed_status=first.status,
                    deterministic=deterministic,
                    findings=findings,
                )
            )
        except Exception as error:
            case_results.append(
                GoldenCaseResult(
                    case_id=case.case_id,
                    passed=False,
                    deterministic=False,
                    findings=[f"metric computer raised {type(error).__name__}: {error}"],
                )
            )
            observed[case.case_id] = None
    invariant_ids = [invariant.invariant_id for invariant in invariants]
    if len(invariant_ids) != len(set(invariant_ids)):
        raise ValueError("metric invariant IDs must be unique")
    invariant_results: list[InvariantResult] = []
    for invariant in invariants:
        invariant_findings: list[str] = []
        if invariant.left_case_id not in observed or invariant.right_case_id not in observed:
            invariant_findings.append("invariant references an unknown golden case")
            left = right = None
        else:
            left = observed[invariant.left_case_id]
            right = observed[invariant.right_case_id]
            if left is None or right is None:
                invariant_findings.append("invariant input did not produce a numeric normalized value")
            elif invariant.relation is InvariantRelation.EQUAL and not math.isclose(
                left, right, abs_tol=invariant.tolerance, rel_tol=0
            ):
                invariant_findings.append("values were expected to be equal")
            elif invariant.relation is InvariantRelation.NON_DECREASING and left > right + invariant.tolerance:
                invariant_findings.append("left value exceeded right value for a non-decreasing invariant")
            elif invariant.relation is InvariantRelation.NON_INCREASING and left + invariant.tolerance < right:
                invariant_findings.append("left value was below right value for a non-increasing invariant")
        invariant_results.append(
            InvariantResult(
                invariant_id=invariant.invariant_id,
                passed=not invariant_findings,
                left_value=left,
                right_value=right,
                findings=invariant_findings,
            )
        )
    limitations: list[str] = []
    if not cases:
        limitations.append("no golden metric cases were supplied")
    if not invariants:
        limitations.append("no metric invariants were supplied")
    return MetricValidationReport(
        metric_id=metric_id,
        cases=case_results,
        invariants=invariant_results,
        limitations=limitations,
    )


def _coerce_output(
    output: MetricResult | float | None,
    metric_id: str,
) -> MetricResult:
    """Normalize simple test doubles to the canonical metric result shape."""
    if isinstance(output, MetricResult):
        return output
    if output is None:
        return MetricResult(
            metric_id=metric_id,
            version="fixture",
            metric_name="fixture",
            role=MetricRole.DIAGNOSTIC,
            direction=MetricDirection.HIGHER_IS_BETTER,
            eligible=False,
            status=MetricStatus.MISSING,
            eligibility_reason="fixture returned no value",
        )
    return MetricResult(
        metric_id=metric_id,
        version="fixture",
        metric_name="fixture",
        role=MetricRole.DIAGNOSTIC,
        direction=MetricDirection.HIGHER_IS_BETTER,
        raw_value=float(output),
        normalized_value=float(output),
        eligible=True,
        status=MetricStatus.SCORED,
        eligibility_reason="fixture returned a scalar",
    )


def _numeric_or_none(value: Any) -> float | None:
    """Keep the report schema numeric without coercing arbitrary raw payloads."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _close(left: float | None, right: float | None, tolerance: float) -> bool:
    """Compare optional numeric values with explicit missingness semantics."""
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, abs_tol=tolerance, rel_tol=0)


def _close_any(left: Any, right: Any, tolerance: float) -> bool:
    """Compare numeric raw values while preserving equality for non-numeric values."""
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=0)
    return bool(left == right)


__all__ = [
    "GoldenCaseResult",
    "GoldenMetricCase",
    "InvariantRelation",
    "InvariantResult",
    "MetricInvariantCase",
    "MetricValidationReport",
    "run_golden_metric_suite",
    "validate_metric_result",
]
