"""Batch integration and biological-conservation metric adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_evals.metrics.backends import scib_metrics
from agent_evals.metrics.backends.sklearn import adjusted_rand, silhouette
from agent_evals.metrics.builtin._helpers import (
    embedding,
    failed,
    labels,
    reference_labels,
    unavailable,
)
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.models import (
    MetricApplicability,
    MetricCategory,
    MetricDefinition,
    MetricDirection,
    MetricRole,
    NormalizationSpec,
)
from agent_evals.metrics.registry import MetricComputation


def _definition(
    metric_id: str,
    name: str,
    role: MetricRole,
    *,
    category: MetricCategory = MetricCategory.BATCH_INTEGRATION,
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
) -> MetricDefinition:
    return MetricDefinition(
        metric_id=metric_id,
        name=name,
        version="1.0",
        description=name,
        category=category,
        role=role,
        direction=direction,
        native_min=0,
        native_max=1,
        applicability=MetricApplicability(
            required_artifacts=["embedding"],
            structural_metadata=["batch"],
        ),
        computation_backend="sklearn/scipy",
        normalization=NormalizationSpec(policy="bounded"),
    )


def _batch_and_embedding(context: ScientificMetricContext) -> tuple[Any, Any] | None:
    """Resolve a candidate representation and batch vector from either tier."""
    representation = embedding(context)
    if representation is None:
        return None
    adata = context.adata
    if adata is not None:
        key = next(
            (item for item in ("batch", "batch_id", "batch_labels") if item in adata.obs),
            None,
        )
        if key is not None:
            return representation, adata.obs[key].astype(str).to_numpy()
    # Workspace evaluators do not have an in-memory AnnData object. They can still
    # provide a verified batch vector alongside the embedding artifact; keeping
    # this fallback in the metric context makes the typed and black-box tiers use
    # the same backend rather than silently making the free tier unmeasurable.
    for key in ("batch", "batch_labels", "batch_id"):
        values = context.candidate_artifacts.get(key)
        if values is None:
            values = context.metadata.get(f"{key}_values")
        if values is not None and not isinstance(values, str):
            return representation, np.asarray(values).astype(str)
    return None


def _scib_missing() -> MetricComputation:
    """Keep an optional authoritative backend gap out of the agent's score."""
    return unavailable(
        "scib-metrics is not installed; install the pinned science extra to "
        "compute the authoritative integration metric"
    )


def _scib_parameters(context: ScientificMetricContext) -> dict[str, int]:
    """Read deterministic neighbor settings from evaluator configuration."""
    return {
        "n_neighbors": max(2, int(context.metadata.get("integration_n_neighbors", 30))),
        "random_state": int(context.metadata.get("integration_random_state", 0)),
        "n_jobs": int(context.metadata.get("integration_n_jobs", 1)),
    }


def _ilis(context: ScientificMetricContext) -> MetricComputation:
    if not scib_metrics.available():
        return _scib_missing()
    values = _batch_and_embedding(context)
    if values is None:
        return failed("batch labels and embedding unavailable")
    representation, batches = values
    parameters = _scib_parameters(context)
    k = min(parameters["n_neighbors"], len(batches) - 1)
    if k < 2 or len(set(batches)) < 2:
        return failed("at least two batches and two neighbors are required")
    try:
        value, metadata = scib_metrics.ilisi_knn(
            representation,
            batches,
            n_neighbors=k,
            random_state=parameters["random_state"],
            n_jobs=parameters["n_jobs"],
        )
    except ImportError as error:
        return unavailable(f"scib-metrics neighbor backend is unavailable: {error}")
    return MetricComputation(value, metadata=metadata)


def _kbet(context: ScientificMetricContext) -> MetricComputation:
    """Compute kBET per label through the pinned scib-metrics backend."""
    if not scib_metrics.available():
        return _scib_missing()
    values = _batch_and_embedding(context)
    biological = reference_labels(context)
    if values is None or biological is None:
        return failed("batch labels, biological labels, and embedding unavailable")
    representation, batches = values
    parameters = _scib_parameters(context)
    k = min(parameters["n_neighbors"], len(batches) - 1)
    if k < 2 or len(set(batches)) < 2:
        return failed("at least two batches and two neighbors are required")
    try:
        value, metadata = scib_metrics.kbet_per_label(
            representation,
            batches,
            biological,
            n_neighbors=k,
            random_state=parameters["random_state"],
            n_jobs=parameters["n_jobs"],
        )
    except ImportError as error:
        return unavailable(f"scib-metrics neighbor backend is unavailable: {error}")
    return MetricComputation(value, metadata=metadata)


def _bras(context: ScientificMetricContext) -> MetricComputation:
    """Compute BRAS through the pinned scib-metrics backend."""
    if not scib_metrics.available():
        return _scib_missing()
    values = _batch_and_embedding(context)
    biological = reference_labels(context)
    if values is None or biological is None:
        return failed("batch labels, biological labels, and embedding unavailable")
    representation, batches = values
    try:
        value, metadata = scib_metrics.bras(representation, biological, batches)
    except ImportError as error:
        return unavailable(f"scib-metrics is unavailable: {error}")
    return MetricComputation(value, metadata=metadata)


def cell_type_asw(context: ScientificMetricContext) -> MetricComputation:
    biological = reference_labels(context)
    representation = embedding(context)
    return (
        failed("biological labels and embedding unavailable")
        if biological is None or representation is None
        else MetricComputation((silhouette(representation, biological) + 1) / 2)
    )


def label_conservation_ari(context: ScientificMetricContext) -> MetricComputation:
    values = labels(context)
    return failed("reference and conserved labels unavailable") if values is None else MetricComputation((adjusted_rand(values[0], values[1]) + 1) / 2)


def pcr_batch_variance(context: ScientificMetricContext) -> MetricComputation:
    values = _batch_and_embedding(context)
    if values is None:
        return failed("batch labels and embedding unavailable")
    representation, batches = values
    overall = float(np.var(representation))
    within = float(np.mean([np.var(representation[batches == batch]) for batch in set(batches)]))
    if overall == 0:
        return MetricComputation(1.0)
    return MetricComputation(max(0.0, min(1.0, 1 - within / overall)))


def graph_connectivity(context: ScientificMetricContext) -> MetricComputation:
    if not scib_metrics.available():
        return _scib_missing()
    values = _batch_and_embedding(context)
    biological = reference_labels(context)
    if values is None or biological is None:
        return failed("embedding and biological labels unavailable")
    representation, _ = values
    parameters = _scib_parameters(context)
    k = min(parameters["n_neighbors"], len(biological) - 1)
    if k < 2:
        return failed("at least two neighbors are required")
    try:
        value, metadata = scib_metrics.graph_connectivity(
            representation,
            biological,
            n_neighbors=k,
            random_state=parameters["random_state"],
            n_jobs=parameters["n_jobs"],
        )
    except ImportError as error:
        return unavailable(f"scib-metrics neighbor backend is unavailable: {error}")
    return MetricComputation(value, metadata=metadata)


def batch_asw(context: ScientificMetricContext) -> MetricComputation:
    values = _batch_and_embedding(context)
    return (
        failed("batch labels and embedding unavailable")
        if values is None
        else MetricComputation((silhouette(*values) + 1) / 2, metadata={"raw_silhouette": silhouette(*values)})
    )


def integration_definitions() -> list[tuple[MetricDefinition, Any]]:
    """Return integration metric definitions."""
    return [
        (_definition("batch_integration.iLISI", "iLISI", MetricRole.PRIMARY), _ilis),
        (_definition("batch_integration.kBET", "kBET", MetricRole.PRIMARY), _kbet),
        (_definition("batch_integration.BRAS", "BRAS", MetricRole.PRIMARY), _bras),
        (_definition("biological_conservation.cell_type_asw", "Cell-type ASW", MetricRole.PRIMARY, category=MetricCategory.BIOLOGICAL_CONSERVATION), cell_type_asw),
        (_definition("biological_conservation.ari", "Label conservation ARI", MetricRole.PRIMARY, category=MetricCategory.BIOLOGICAL_CONSERVATION), label_conservation_ari),
        (_definition("batch_integration.pcr", "PCR batch variance reduction", MetricRole.SECONDARY), pcr_batch_variance),
        (_definition("biological_conservation.graph_connectivity", "Graph connectivity", MetricRole.PRIMARY, category=MetricCategory.BIOLOGICAL_CONSERVATION), graph_connectivity),
        (_definition("batch_integration.batch_asw", "Batch ASW", MetricRole.DIAGNOSTIC), batch_asw),
    ]


__all__ = ["integration_definitions"]
