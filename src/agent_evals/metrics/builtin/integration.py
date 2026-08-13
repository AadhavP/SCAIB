"""Batch integration and biological-conservation metric adapters."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_evals.metrics.backends.sklearn import adjusted_rand, silhouette
from agent_evals.metrics.builtin._helpers import (
    embedding,
    failed,
    labels,
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
    adata = context.adata
    representation = embedding(context)
    if adata is None or representation is None:
        return None
    key = next((item for item in ("batch", "batch_id", "batch_labels") if item in adata.obs), None)
    if key is None:
        return None
    return representation, adata.obs[key].astype(str).to_numpy()


def _ilis(context: ScientificMetricContext) -> MetricComputation:
    values = _batch_and_embedding(context)
    if values is None:
        return failed("batch labels and embedding unavailable")
    representation, batches = values
    from sklearn.neighbors import NearestNeighbors

    k = min(30, len(batches) - 1)
    if k < 2 or len(set(batches)) < 2:
        return failed("at least two batches and two neighbors are required")
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(representation).kneighbors(return_distance=False)[:, 1:]
    scores = []
    batch_count = len(set(batches))
    for row in neighbors:
        counts = np.bincount(
            np.asarray([sorted(set(batches)).index(batches[index]) for index in row]),
            minlength=batch_count,
        )
        probabilities = counts / counts.sum()
        scores.append(1 / np.square(probabilities).sum())
    return MetricComputation(float(np.mean(scores) / batch_count), metadata={"k": k, "implementation": "local_inverse_simpson_knn"})


def _kbet(context: ScientificMetricContext) -> MetricComputation:
    """Report the missing backend rather than scoring the agent for it.

    Both branches were ``failed(...)``, which carries the metric's failure score
    of 0.0 -- so an agent with flawless batch integration was charged a zero for
    a metric SCAIB has never implemented. Neither branch depends on anything the
    agent did, which is the tell.
    """
    from agent_evals.metrics.backends.scib_metrics import available

    if not available():
        return unavailable("scib-metrics is not installed")
    return unavailable("no kBET adapter is wired to the installed scib-metrics")


def _bras(context: ScientificMetricContext) -> MetricComputation:
    """See :func:`_kbet`: a harness gap, reported as one."""
    from agent_evals.metrics.backends.scib_metrics import available

    if not available():
        return unavailable("scib-metrics is not installed")
    return unavailable("no BRAS adapter is wired to the installed scib-metrics")


def cell_type_asw(context: ScientificMetricContext) -> MetricComputation:
    values = labels(context)
    representation = embedding(context)
    return (
        failed("cell-type labels and embedding unavailable")
        if values is None or representation is None
        else MetricComputation((silhouette(representation, values[0]) + 1) / 2)
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
    adata = context.adata
    values = labels(context)
    if adata is None or values is None or "connectivities" not in adata.obsp:
        return failed("connectivity graph and labels unavailable")
    graph = adata.obsp["connectivities"]
    scores = []
    for label in sorted(set(values[0])):
        indices = np.flatnonzero(values[0] == label)
        if len(indices) < 2:
            continue
        subgraph = graph[indices][:, indices]
        scores.append(float((subgraph.sum(axis=1) > 0).mean()))
    return MetricComputation(float(np.mean(scores)) if scores else None)


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
        (_definition("batch_integration.graph_connectivity", "Graph connectivity", MetricRole.SECONDARY), graph_connectivity),
        (_definition("batch_integration.batch_asw", "Batch ASW", MetricRole.DIAGNOSTIC), batch_asw),
    ]


__all__ = ["integration_definitions"]
