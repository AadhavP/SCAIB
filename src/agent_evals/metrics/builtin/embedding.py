"""Embedding fidelity metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_evals.metrics.builtin._helpers import failed
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


def _definition(metric_id: str, name: str, direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER) -> MetricDefinition:
    return MetricDefinition(
        metric_id=metric_id,
        name=name,
        version="1.0",
        description=name,
        category=MetricCategory.EMBEDDING,
        role=MetricRole.PRIMARY,
        direction=direction,
        native_min=0,
        native_max=1 if direction == MetricDirection.HIGHER_IS_BETTER else None,
        applicability=MetricApplicability(required_artifacts=["embedding"], required_representations=["X_pca"]),
        computation_backend="sklearn/scipy",
        normalization=NormalizationSpec(policy="bounded") if direction == MetricDirection.HIGHER_IS_BETTER else NormalizationSpec(policy="anchor", bad_anchor=1, target_anchor=0),
    )


def _representations(context: ScientificMetricContext) -> tuple[Any, Any] | None:
    adata = context.adata
    if adata is None or "X_pca" not in adata.obsm or "X_umap" not in adata.obsm:
        return None
    return adata.obsm["X_pca"], adata.obsm["X_umap"]


def trustworthiness_k(context: ScientificMetricContext) -> MetricComputation:
    values = _representations(context)
    if values is None:
        return failed("reference and candidate representations unavailable")
    from sklearn.manifold import trustworthiness

    k = int(context.metadata.get("k", 15))
    return MetricComputation(trustworthiness(values[0], values[1], n_neighbors=min(k, len(values[0]) - 1)), metadata={"k": k, "n_cells": len(values[0]), "representation": "X_umap"})


def knn_overlap(context: ScientificMetricContext) -> MetricComputation:
    values = _representations(context)
    if values is None:
        return failed("reference and candidate representations unavailable")
    from sklearn.neighbors import NearestNeighbors

    k = min(int(context.metadata.get("k", 15)), len(values[0]) - 1)
    ref = NearestNeighbors(n_neighbors=k + 1).fit(values[0]).kneighbors(return_distance=False)[:, 1:]
    candidate = NearestNeighbors(n_neighbors=k + 1).fit(values[1]).kneighbors(return_distance=False)[:, 1:]
    score = np.mean([len(set(ref[index]) & set(candidate[index])) / k for index in range(len(ref))])
    return MetricComputation(float(score), metadata={"k": k, "n_cells": len(ref)})


def continuity(context: ScientificMetricContext) -> MetricComputation:
    return knn_overlap(context)


def normalized_stress(context: ScientificMetricContext) -> MetricComputation:
    values = _representations(context)
    if values is None:
        return failed("reference and candidate representations unavailable")
    from sklearn.metrics import pairwise_distances

    reference = pairwise_distances(values[0])
    candidate = pairwise_distances(values[1])
    denominator = float(np.square(reference).sum())
    if denominator == 0:
        return failed("reference distances are all zero")
    stress = float(np.sqrt(np.square(reference - candidate).sum() / denominator))
    return MetricComputation(stress, metadata={"n_cells": len(values[0]), "representation": "X_umap"})


def embedding_definitions() -> list[tuple[MetricDefinition, Any]]:
    return [
        (_definition("embedding.trustworthiness", "Trustworthiness@K"), trustworthiness_k),
        (_definition("embedding.continuity", "Continuity@K"), continuity),
        (_definition("embedding.knn_overlap", "kNN overlap / recall@K"), knn_overlap),
        (_definition("embedding.normalized_stress", "Normalized stress", MetricDirection.LOWER_IS_BETTER), normalized_stress),
    ]


__all__ = ["embedding_definitions"]
