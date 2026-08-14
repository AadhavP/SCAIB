"""Study plans and evidence records for baseline and ablation experiments.

A benchmark result is not a research result until it is situated against fixed
baselines and controlled ablations. These models define that protocol without
forcing the benchmark runner to use one execution framework: runners contribute
replicate records, while this module validates identities, seeds, and paired
comparisons.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_evals.baselines.base import BaselineResult
from agent_evals.research.statistics import (
    PairedComparison,
    ReplicateScore,
    ReplicateStatus,
    StudyStatisticsReport,
    build_statistics_report,
    compare_paired,
)


class StudyArmKind(StrEnum):
    """Role of an arm in a controlled study."""

    AGENT = "agent"
    DETERMINISTIC_BASELINE = "deterministic_baseline"
    WEAK_BASELINE = "weak_baseline"
    ORACLE = "oracle"
    ABLATION = "ablation"


class StudyArm(BaseModel):
    """One named arm and its repeated score-bearing runs."""

    model_config = ConfigDict(extra="forbid")

    arm_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: StudyArmKind
    replicates: list[ReplicateScore] = Field(default_factory=list)
    implementation_digest: str | None = None
    environment_digest: str | None = None
    benchmark_digest: str | None = None
    dataset_digest: str | None = None
    configuration_digest: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def _validate_digest(value: str | None, field_name: str) -> str | None:
        """Require immutable identifiers when an arm declares one."""
        if value is not None and re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError(f"{field_name} must be a 64-character hexadecimal digest")
        return value.lower() if value is not None else None

    @model_validator(mode="after")
    def validate_digests(self) -> StudyArm:
        """Normalize and validate reproducibility digests."""
        for field_name in (
            "implementation_digest",
            "environment_digest",
            "benchmark_digest",
            "dataset_digest",
            "configuration_digest",
        ):
            value = self._validate_digest(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
        return self


class AblationSpec(BaseModel):
    """Predeclared removal or substitution used to test a causal component."""

    model_config = ConfigDict(extra="forbid")

    ablation_id: str = Field(min_length=1)
    full_arm_id: str = Field(min_length=1)
    ablated_arm_id: str = Field(min_length=1)
    removed_components: list[str] = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    expected_direction: str = Field(min_length=1)
    required: bool = True


class StudyPlan(BaseModel):
    """Frozen protocol for baseline, ablation, and replicate collection."""

    model_config = ConfigDict(extra="forbid")

    study_id: str = Field(min_length=1)
    benchmark_id: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    protocol_version: str = "1.0.0"
    required_replicates: int = Field(default=3, ge=1)
    seed_schedule: list[int] = Field(min_length=1)
    arm_ids: list[str] = Field(min_length=1)
    required_arm_kinds: list[StudyArmKind] = Field(
        default_factory=lambda: [
            StudyArmKind.DETERMINISTIC_BASELINE,
            StudyArmKind.WEAK_BASELINE,
            StudyArmKind.ORACLE,
        ]
    )
    ablations: list[AblationSpec] = Field(default_factory=list)
    multiple_comparison_method: str = "benjamini_hochberg"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_protocol(self) -> StudyPlan:  # noqa: C901
        """Reject ambiguous arms and ablations before any run consumes budget."""
        if self.multiple_comparison_method != "benjamini_hochberg":
            raise ValueError(
                "study multiple_comparison_method must be 'benjamini_hochberg'"
            )
        if len(self.seed_schedule) != len(set(self.seed_schedule)):
            raise ValueError("study seed_schedule must contain unique seeds")
        if len(self.seed_schedule) < self.required_replicates:
            raise ValueError(
                "study seed_schedule must contain at least required_replicates "
                "unique seeds"
            )
        if len(self.arm_ids) != len(set(self.arm_ids)):
            raise ValueError("study arm_ids must be unique")
        if len(self.required_arm_kinds) != len(set(self.required_arm_kinds)):
            raise ValueError("study required_arm_kinds must be unique")
        ablation_ids = [ablation.ablation_id for ablation in self.ablations]
        if len(ablation_ids) != len(set(ablation_ids)):
            raise ValueError("study ablation IDs must be unique")
        for ablation in self.ablations:
            if ablation.full_arm_id == ablation.ablated_arm_id:
                raise ValueError(
                    f"ablation '{ablation.ablation_id}' must compare two different arms"
                )
            if len(ablation.removed_components) != len(set(ablation.removed_components)):
                raise ValueError(
                    f"ablation '{ablation.ablation_id}' has duplicate removed components"
                )
            if ablation.full_arm_id not in self.arm_ids:
                raise ValueError(
                    f"ablation '{ablation.ablation_id}' references unknown full arm "
                    f"'{ablation.full_arm_id}'"
                )
            if ablation.ablated_arm_id not in self.arm_ids:
                raise ValueError(
                    f"ablation '{ablation.ablation_id}' references unknown ablated arm "
                    f"'{ablation.ablated_arm_id}'"
                )
        return self

    def canonical_payload(self) -> dict[str, Any]:
        """Return the frozen, timestamp-independent protocol declaration."""
        return self.model_dump(mode="json")

    def canonical_digest(self) -> str:
        """Hash the study protocol used to generate every replicate."""
        encoded = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class AblationResult(BaseModel):
    """Observed effect of one predeclared ablation."""

    model_config = ConfigDict(extra="forbid")

    ablation_id: str
    full_arm_id: str
    ablated_arm_id: str
    comparison: PairedComparison
    hypothesis: str
    expected_direction: str
    interpretation: str


class StudyReport(BaseModel):
    """Complete baseline/ablation study artifact."""

    model_config = ConfigDict(extra="forbid")

    study_version: str = "1.0.0"
    plan: StudyPlan
    protocol_digest: str | None = None
    arms: dict[str, StudyArm]
    statistics: StudyStatisticsReport
    baseline_results: dict[str, BaselineResult] = Field(default_factory=dict)
    ablation_results: list[AblationResult] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @property
    def research_ready(self) -> bool:
        """Whether the study satisfies its declared protocol, not just its count."""
        observed_kinds = {arm.kind for arm in self.arms.values()}
        required_kinds_present = all(
            kind in observed_kinds for kind in self.plan.required_arm_kinds
        )
        expected_seeds = set(self.plan.seed_schedule[: self.plan.required_replicates])
        enough_replicates = all(
            expected_seeds.issubset(
                {
                    replicate.seed
                    for replicate in arm.replicates
                    if replicate.status is ReplicateStatus.COMPLETED
                }
            )
            for arm in self.arms.values()
        )
        return (
            not self.limitations
            and required_kinds_present
            and set(self.arms) == set(self.plan.arm_ids)
            and enough_replicates
        )


def build_study_report(  # noqa: C901
    plan: StudyPlan,
    arms: list[StudyArm],
    *,
    baseline_results: dict[str, BaselineResult] | None = None,
    dimensions: tuple[str, ...] = ("score",),
    bootstrap_iterations: int = 2000,
    permutation_iterations: int = 5000,
    seed: int = 0,
) -> StudyReport:
    """Validate arms, calculate statistics, and materialize ablation comparisons.

    The first ``required_replicates`` entries of ``seed_schedule`` are the frozen
    primary replicate set. Extra scheduled seeds may be collected for exploratory
    analysis, but they cannot silently replace a missing primary seed.
    """
    duplicate_arm_ids = sorted(
        arm_id
        for arm_id in {arm.arm_id for arm in arms}
        if sum(item.arm_id == arm_id for item in arms) > 1
    )
    if duplicate_arm_ids:
        raise ValueError(
            "study contains duplicate arm identifier(s): "
            + ", ".join(duplicate_arm_ids)
        )
    arm_map = {arm.arm_id: arm for arm in arms}
    missing = sorted(set(plan.arm_ids) - set(arm_map))
    unexpected = sorted(set(arm_map) - set(plan.arm_ids))
    if missing or unexpected:
        detail = []
        if missing:
            detail.append("missing arms: " + ", ".join(missing))
        if unexpected:
            detail.append("unexpected arms: " + ", ".join(unexpected))
        raise ValueError("study arm declarations do not match plan (" + "; ".join(detail) + ")")
    statistics = build_statistics_report(
        plan.study_id,
        {arm_id: arm.replicates for arm_id, arm in arm_map.items()},
        dimensions=dimensions,
        bootstrap_iterations=bootstrap_iterations,
        permutation_iterations=permutation_iterations,
        seed=seed,
    )
    ablation_results: list[AblationResult] = []
    limitations = list(statistics.limitations)
    observed_kinds = {arm.kind for arm in arm_map.values()}
    for required_kind in plan.required_arm_kinds:
        if required_kind not in observed_kinds:
            limitations.append(
                f"study is missing required arm kind '{required_kind.value}'"
            )
    for arm_id, arm in arm_map.items():
        seeds = [replicate.seed for replicate in arm.replicates]
        if len(seeds) != len(set(seeds)):
            limitations.append(f"arm '{arm_id}' repeats a seed across replicates")
        outside_schedule = sorted(set(seeds) - set(plan.seed_schedule))
        if outside_schedule:
            limitations.append(
                f"arm '{arm_id}' uses seeds outside the frozen schedule: "
                + ", ".join(str(seed) for seed in outside_schedule)
            )
        expected_seeds = set(plan.seed_schedule[: plan.required_replicates])
        missing_schedule_seeds = sorted(expected_seeds - set(seeds))
        if missing_schedule_seeds:
            limitations.append(
                f"arm '{arm_id}' is missing required frozen seed(s): "
                + ", ".join(str(seed) for seed in missing_schedule_seeds)
            )
        incomplete_schedule_seeds = sorted(
            replicate.seed
            for replicate in arm.replicates
            if replicate.seed in expected_seeds
            and replicate.status is not ReplicateStatus.COMPLETED
        )
        if incomplete_schedule_seeds:
            limitations.append(
                f"arm '{arm_id}' has failed/ineligible required seed(s): "
                + ", ".join(str(seed) for seed in incomplete_schedule_seeds)
            )
        for field_name in (
            "implementation_digest",
            "benchmark_digest",
            "dataset_digest",
            "configuration_digest",
            "environment_digest",
        ):
            if getattr(arm, field_name) is None:
                limitations.append(
                    f"arm '{arm_id}' has no {field_name} for reproducibility"
                )
    for field_name in (
        "benchmark_digest",
        "dataset_digest",
        "configuration_digest",
        "environment_digest",
    ):
        values = {
            getattr(arm, field_name)
            for arm in arm_map.values()
            if getattr(arm, field_name) is not None
        }
        if len(values) > 1:
            limitations.append(
                f"study arms disagree on shared {field_name}"
            )
    for ablation in plan.ablations:
        comparison = compare_paired(
            ablation.full_arm_id,
            arm_map[ablation.full_arm_id].replicates,
            ablation.ablated_arm_id,
            arm_map[ablation.ablated_arm_id].replicates,
            dimension="score",
            bootstrap_iterations=bootstrap_iterations,
            permutation_iterations=permutation_iterations,
            seed=seed,
        )
        if comparison.n_pairs < plan.required_replicates:
            message = (
                f"ablation '{ablation.ablation_id}' has {comparison.n_pairs} paired "
                f"replicate(s), requires {plan.required_replicates}"
            )
            if ablation.required:
                limitations.append(message)
        ablation_results.append(
            AblationResult(
                ablation_id=ablation.ablation_id,
                full_arm_id=ablation.full_arm_id,
                ablated_arm_id=ablation.ablated_arm_id,
                comparison=comparison,
                hypothesis=ablation.hypothesis,
                expected_direction=ablation.expected_direction,
                interpretation=_interpret_ablation(comparison, ablation.expected_direction),
            )
        )
    for arm_id, arm in arm_map.items():
        if len(arm.replicates) < plan.required_replicates:
            limitations.append(
                f"arm '{arm_id}' has {len(arm.replicates)} replicate(s), "
                f"requires {plan.required_replicates}"
            )
        if arm.kind is StudyArmKind.DETERMINISTIC_BASELINE and not arm.implementation_digest:
            limitations.append(f"deterministic baseline '{arm_id}' has no implementation digest")
    return StudyReport(
        plan=plan,
        protocol_digest=plan.canonical_digest(),
        arms=arm_map,
        statistics=statistics,
        baseline_results=baseline_results or {},
        ablation_results=ablation_results,
        limitations=list(dict.fromkeys(limitations)),
    )


def _interpret_ablation(comparison: PairedComparison, expected_direction: str) -> str:
    """Write a cautious interpretation without turning a p-value into causality."""
    if comparison.mean_delta is None:
        return "unmeasured: no paired replicates were available"
    direction = "improved" if comparison.mean_delta > 0 else "declined" if comparison.mean_delta < 0 else "did not change"
    return (
        f"the full arm {direction} the score by {abs(comparison.mean_delta):.4g} "
        f"on average; expected direction was {expected_direction}"
    )


__all__ = [
    "AblationResult",
    "AblationSpec",
    "StudyArm",
    "StudyArmKind",
    "StudyPlan",
    "StudyReport",
    "build_study_report",
]
