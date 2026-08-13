"""Metric computation and scientific outcome aggregation engine."""

from __future__ import annotations

from importlib import metadata
from typing import Any

from agent_evals.metrics import (
    ApplicabilityContext,
    MetricApplicabilityEngine,
    MetricDefinition,
    MetricGroup,
    MetricRegistry,
    NormalizationEngine,
    aggregate_group,
    metric_registry,
)
from agent_evals.metrics.applicability import ApplicabilityResult
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.results import MetricResult, MetricStatus


class ScientificMetricEngine:
    """Evaluate versioned metrics with frozen applicability decisions."""

    def __init__(
        self,
        registry: MetricRegistry = metric_registry,
        applicability: MetricApplicabilityEngine | None = None,
        normalization: NormalizationEngine | None = None,
    ) -> None:
        self.registry = registry
        self.applicability = applicability or MetricApplicabilityEngine()
        self.normalization = normalization or NormalizationEngine()

    def evaluate(
        self,
        metric_ids: list[str],
        context: ScientificMetricContext,
        *,
        applicability_context: ApplicabilityContext | None = None,
        groups: list[MetricGroup] | None = None,
    ) -> tuple[list[MetricResult], list[ApplicabilityResult], list[Any], float | None]:
        """Compute metrics and groups without candidate-specific weight changes."""
        applicability_context = applicability_context or self._context(context)
        results: list[MetricResult] = []
        decisions: list[ApplicabilityResult] = []
        for metric_id in metric_ids:
            definition = self.registry.get(metric_id)
            decision = self.applicability.evaluate(definition, applicability_context)
            decisions.append(decision)
            if decision.structurally_ineligible:
                results.append(
                    MetricResult(
                        metric_id=definition.metric_id,
                        version=definition.version,
                        metric_name=definition.name,
                        role=definition.role,
                        direction=definition.direction,
                        eligible=False,
                        status=MetricStatus.INELIGIBLE,
                        eligibility_reason=decision.reason,
                        metadata={"implementation": self._implementation_metadata(definition)},
                    )
                )
                continue
            if decision.missing_candidate_artifacts or decision.missing_candidate_metadata:
                results.append(
                    MetricResult(
                        metric_id=definition.metric_id,
                        version=definition.version,
                        metric_name=definition.name,
                        role=definition.role,
                        direction=definition.direction,
                        raw_value=None,
                        normalized_value=definition.failure_score,
                        eligible=True,
                        # The agent could have produced this input and did not, so
                        # the failure score stands. Naming the cause only makes the
                        # record readable; it does not excuse the gap.
                        status=MetricStatus.MISSING,
                        eligibility_reason=decision.reason,
                        missing_artifacts=decision.missing_candidate_artifacts,
                        metadata={
                            "missing_metadata": decision.missing_candidate_metadata,
                            "implementation": self._implementation_metadata(definition),
                        },
                    )
                )
                continue
            try:
                computation = self.registry.get_computer(metric_id)(context)
                if computation.unavailable:
                    results.append(
                        MetricResult(
                            metric_id=definition.metric_id,
                            version=definition.version,
                            metric_name=definition.name,
                            role=definition.role,
                            direction=definition.direction,
                            # No number, and deliberately not the failure score.
                            # This outcome is identical for a perfect run and a
                            # worthless one, so it carries no information about the
                            # agent and must not be spent on the agent's score.
                            normalized_value=None,
                            eligible=False,
                            status=MetricStatus.UNIMPLEMENTED,
                            eligibility_reason=decision.reason,
                            metadata={
                                **(computation.metadata or {}),
                                "implementation": self._implementation_metadata(definition),
                            },
                        )
                    )
                    continue
                if computation.raw_value is None:
                    results.append(
                        MetricResult(
                            metric_id=definition.metric_id,
                            version=definition.version,
                            metric_name=definition.name,
                            role=definition.role,
                            direction=definition.direction,
                            normalized_value=definition.failure_score,
                            eligible=True,
                            # Every required input was present and the computer
                            # still read no number out of them.
                            status=MetricStatus.MALFORMED,
                            eligibility_reason=decision.reason,
                            metadata={
                                **(computation.metadata or {}),
                                "implementation": self._implementation_metadata(definition),
                            },
                        )
                    )
                    continue
                normalized = self.normalization.normalize(computation.raw_value, definition)
                results.append(
                    MetricResult(
                        metric_id=definition.metric_id,
                        version=definition.version,
                        metric_name=definition.name,
                        role=definition.role,
                        direction=definition.direction,
                        raw_value=computation.raw_value,
                        normalized_value=normalized,
                        eligible=True,
                        status=MetricStatus.SCORED,
                        eligibility_reason=decision.reason,
                        metadata={
                            **(computation.metadata or {}),
                            "evidence": list(computation.evidence),
                            "implementation": self._implementation_metadata(definition),
                        },
                    )
                )
            except Exception as error:
                results.append(
                    MetricResult(
                        metric_id=definition.metric_id,
                        version=definition.version,
                        metric_name=definition.name,
                        role=definition.role,
                        direction=definition.direction,
                        raw_value=None,
                        # Still the failure score, deliberately. The agent chose
                        # the arrays this computer ran on, so excluding an
                        # evaluator crash from the aggregate would let it delete a
                        # metric it was about to fail by handing the computer
                        # something that raises.
                        normalized_value=definition.failure_score,
                        eligible=True,
                        status=MetricStatus.EVALUATOR_ERROR,
                        eligibility_reason=decision.reason,
                        metadata={
                            "failure_reason": f"{type(error).__name__}: {error}",
                            "implementation": self._implementation_metadata(definition),
                        },
                    )
                )
        group_results = [aggregate_group(group, results) for group in groups or []]
        primary_group_ids = {
            group.group_id for group in groups or [] if group.contributes_to_primary
        }
        primary = [
            group.value
            for group in group_results
            if group.value is not None
            and (not groups or group.group_id in primary_group_ids)
        ]
        scientific_score = sum(primary) / len(primary) if primary else None
        return results, decisions, group_results, scientific_score

    @staticmethod
    def _context(context: ScientificMetricContext) -> ApplicabilityContext:
        adata = context.adata
        observation_columns = (
            {str(column) for column in adata.obs.columns}
            if adata is not None
            else set()
        )
        representations = {str(key) for key in adata.obsm.keys()} if adata is not None else set()
        return ApplicabilityContext(
            structural_artifacts=set(context.reference_artifacts),
            structural_metadata=set(context.metadata),
            candidate_artifacts=set(context.candidate_artifacts),
            candidate_metadata=set(context.metadata),
            observation_columns=observation_columns,
            representations=representations,
            reference_labels_available=context.has_reference_labels,
            reference_gap_reason=context.reference_join_gap,
            predictions_available="prediction" in context.candidate_artifacts
            or "cluster_labels" in context.candidate_artifacts,
            payload=context,
        )

    @staticmethod
    def _implementation_metadata(definition: MetricDefinition) -> dict[str, str]:
        """Record metric package versions for reproducible reports."""
        packages = ["agent-evals"]
        if definition.computation_backend.startswith("sklearn"):
            packages.append("scikit-learn")
        if "scipy" in definition.computation_backend:
            packages.append("scipy")
        values: dict[str, str] = {}
        for package in packages:
            try:
                values[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                values[package] = "workspace"
        values["backend"] = definition.computation_backend
        values["metric_version"] = definition.version
        return values


__all__ = ["ScientificMetricEngine"]
