"""Adapters from the existing open-source-backed metric implementations."""

from __future__ import annotations

from agent_evals.evaluation.metrics.base import (
    ArtifactBundle,
    EvaluationContext,
    MetricApplicability,
    MetricRole,
    RawMetricResult,
    ReferenceBundle,
    ScientificMetric,
    ScoreAnchors,
)
from agent_evals.metrics.applicability import (
    ApplicabilityContext as LegacyApplicabilityContext,
)
from agent_evals.metrics.applicability import (
    MetricApplicabilityEngine as LegacyApplicabilityEngine,
)
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.models import MetricDefinition
from agent_evals.metrics.models import MetricDirection as LegacyDirection
from agent_evals.metrics.normalization import (
    NormalizationEngine as LegacyNormalizationEngine,
)
from agent_evals.metrics.registry import MetricRegistry as LegacyRegistry
from agent_evals.metrics.registry import metric_registry as legacy_registry


class LegacyMetricAdapter(ScientificMetric):
    """Expose a versioned legacy metric through the generic Stage 8 contract."""

    def __init__(self, definition: MetricDefinition, registry: LegacyRegistry = legacy_registry) -> None:
        self.definition = definition
        self.registry = registry
        self.name = definition.metric_id
        self.category = definition.category.value
        self.direction = (
            "minimize"
            if definition.direction == LegacyDirection.LOWER_IS_BETTER
            else "maximize"
        )
        self.role = MetricRole(definition.role.value)
        self.implementation_version = f"{definition.computation_backend}@{definition.version}"

    def applicability(self, context: EvaluationContext) -> MetricApplicability:
        """Delegate structural/candidate applicability to the existing engine."""
        legacy_context = _legacy_applicability_context(context)
        decision = LegacyApplicabilityEngine().evaluate(self.definition, legacy_context)
        return MetricApplicability(
            applicable=decision.eligible,
            structurally_ineligible=decision.structurally_ineligible,
            reason=decision.reason,
            missing_artifacts=decision.missing_candidate_artifacts,
            missing_metadata=decision.missing_candidate_metadata,
        )

    def compute(
        self,
        prediction: ArtifactBundle,
        reference: ReferenceBundle,
        context: EvaluationContext,
    ) -> RawMetricResult:
        """Call the existing sklearn/scipy/scIB-bound computation."""
        del prediction, reference
        scientific_context = _scientific_context(context)
        computation = self.registry.get_computer(self.definition.metric_id)(scientific_context)
        return RawMetricResult(value=computation.raw_value, metadata=computation.metadata or {})

    def normalize(self, value: float, anchors: ScoreAnchors) -> float:
        """Use the existing direction-aware normalization implementation."""
        del anchors
        normalized = LegacyNormalizationEngine().normalize(value, self.definition)
        if normalized is None:
            raise ValueError(f"metric '{self.name}' has no valid normalization anchors")
        return normalized


def _scientific_context(context: EvaluationContext) -> ScientificMetricContext:
    """Resolve the legacy context carried by the generic payload."""
    if isinstance(context.payload, ScientificMetricContext):
        return context.payload
    if isinstance(context.payload, dict):
        payload = context.payload
        # Every field is carried across, including the two that gate scoring:
        # dropping ``agent_produced_columns`` would let a dataset's own ``louvain``
        # be read back as this agent's clustering, and dropping
        # ``reference_join_gap`` would present a reference the evaluator cannot
        # join and score the run against a candidate it never had. Both failures
        # are silent, and both are in the direction of a wrong number rather than
        # a missing one.
        return ScientificMetricContext(
            adata=payload.get("adata"),
            candidate_artifacts=payload.get("candidate_artifacts", {}),
            reference_artifacts=payload.get("reference_artifacts", {}),
            metadata=payload.get("metadata", {}),
            trajectory=payload.get("trajectory"),
            agent_produced_columns=frozenset(
                payload.get("agent_produced_columns") or ()
            ),
            reference_join_gap=payload.get("reference_join_gap"),
        )
    raise TypeError("generic metric context payload must contain ScientificMetricContext")


def _legacy_applicability_context(context: EvaluationContext) -> LegacyApplicabilityContext:
    """Build legacy applicability evidence from the generic context."""
    scientific = _scientific_context(context)
    adata = scientific.adata
    columns = {str(column) for column in adata.obs.columns} if adata is not None else set()
    representations = {str(key) for key in adata.obsm.keys()} if adata is not None else set()
    return LegacyApplicabilityContext(
        structural_artifacts=set(scientific.reference_artifacts),
        structural_metadata=set(scientific.metadata),
        candidate_artifacts=set(scientific.candidate_artifacts),
        candidate_metadata=set(scientific.metadata),
        observation_columns=columns,
        representations=representations,
        reference_labels_available=scientific.has_reference_labels,
        reference_gap_reason=scientific.reference_join_gap,
        predictions_available="prediction" in scientific.candidate_artifacts
        or "cluster_labels" in scientific.candidate_artifacts,
        payload=scientific,
    )


__all__ = ["LegacyMetricAdapter"]
