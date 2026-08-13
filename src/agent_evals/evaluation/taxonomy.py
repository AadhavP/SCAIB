"""Benchmark-independent scientific decision ontology and profiles."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.agents.trajectory import DecisionCategory, ScientificDecision


class DecisionProfile(BaseModel):
    """Observable evaluation contract for one scientific decision category."""

    model_config = ConfigDict(extra="forbid")

    category: DecisionCategory
    allowed_methods: list[str] = Field(default_factory=list)
    expected_inputs: list[str] = Field(default_factory=list)
    possible_alternatives: list[str] = Field(default_factory=list)
    evaluator_metrics: list[str] = Field(default_factory=list)
    parameter_ranges: dict[str, dict[str, Any]] = Field(default_factory=dict)


class DecisionOntology:
    """Registry of reusable scientific decision profiles."""

    def __init__(self) -> None:
        self._profiles: dict[DecisionCategory, DecisionProfile] = {}

    def register(self, profile: DecisionProfile, *, replace: bool = False) -> None:
        """Register a profile by stable category."""
        if profile.category in self._profiles and not replace:
            raise ValueError(f"decision profile '{profile.category.value}' is already registered")
        self._profiles[profile.category] = profile

    def get(self, category: DecisionCategory) -> DecisionProfile:
        """Resolve a profile, falling back to an empty category contract."""
        return self._profiles.get(
            category,
            DecisionProfile(category=category),
        )

    def list(self) -> list[DecisionProfile]:
        """Return profiles in stable ontology order."""
        return [self._profiles[key] for key in sorted(self._profiles, key=lambda item: item.value)]

    def evaluate_method_allowed(self, decision: ScientificDecision) -> bool | None:
        """Return method validity when the category declares allowed methods."""
        profile = self.get(decision.decision_category)
        if not profile.allowed_methods or decision.chosen_method is None:
            return None
        return decision.chosen_method in profile.allowed_methods


def default_decision_ontology() -> DecisionOntology:
    """Build the deterministic default ontology used by all benchmarks."""
    ontology = DecisionOntology()
    definitions = {
        DecisionCategory.QC_STRATEGY: (
            ["fixed_threshold", "mitochondrial_filter", "adaptive_quantile", "mad_outlier"],
            ["current-anndata", "qc-statistics"],
            # Spelled to match ``LocalRewardEvaluator._WEIGHTS`` and the benchmark
            # YAML, which both say ``rare_population_preservation``. This field
            # named a fourth spelling that nothing scored.
            ["artifact_removal", "biological_retention", "rare_population_preservation"],
        ),
        DecisionCategory.NORMALIZATION: (
            ["library_size_log1p", "median_counts_log1p"],
            ["current-anndata", "qc-statistics"],
            ["library_size_stability", "downstream_separation"],
        ),
        DecisionCategory.INTEGRATION: (
            ["harmony", "scanorama", "scvi"],
            ["normalized-anndata", "batch-labels"],
            ["batch_removal", "biology_preservation"],
        ),
        DecisionCategory.CLUSTERING: (
            ["leiden", "louvain"],
            ["embedding"],
            ["ari", "rare_cell_recovery", "stability"],
        ),
        DecisionCategory.ANNOTATION: (
            ["marker_based"],
            ["normalized-anndata", "marker-table"],
            ["macro_f1", "rare_recall", "calibration"],
        ),
        DecisionCategory.DIFFERENTIAL_EXPRESSION: (
            ["marker-genes", "wilcoxon", "logreg"],
            ["normalized-anndata"],
            ["marker_precision", "marker_recall", "effect_size_correlation"],
        ),
    }
    for category, (methods, inputs, metrics) in definitions.items():
        ontology.register(
            DecisionProfile(
                category=category,
                allowed_methods=methods,
                expected_inputs=inputs,
                possible_alternatives=methods,
                evaluator_metrics=metrics,
            )
        )
    return ontology


decision_ontology = default_decision_ontology()


__all__ = ["DecisionOntology", "DecisionProfile", "decision_ontology", "default_decision_ontology"]
