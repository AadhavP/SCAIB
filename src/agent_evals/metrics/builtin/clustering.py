"""Clustering metric definitions and benchmark-owned label matching."""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_evals.metrics.backends.sklearn import (
    adjusted_rand,
    fowlkes_mallows,
    normalized_mutual_information,
    silhouette,
)
from agent_evals.metrics.builtin._helpers import embedding, failed, labels
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


def _definition(metric_id: str, name: str, role: MetricRole) -> MetricDefinition:
    return MetricDefinition(
        metric_id=metric_id,
        name=name,
        version="1.0",
        description=name,
        category=MetricCategory.CLUSTERING,
        role=role,
        direction=MetricDirection.HIGHER_IS_BETTER,
        native_min=-1 if metric_id.endswith((".ari", ".stability_ari")) else 0,
        native_max=1,
        applicability=MetricApplicability(
            required_artifacts=["cluster_labels"],
            requires_reference_labels=True,
            requires_predictions=True,
        ),
        computation_backend="sklearn",
        normalization=NormalizationSpec(
            policy="bounded"
        ),
    )


def _inputs(context: ScientificMetricContext) -> tuple[Any, Any] | None:
    return labels(context)


def ari(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    return failed("cluster and reference labels unavailable") if values is None else MetricComputation(adjusted_rand(*values))


def ami(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    return failed("cluster and reference labels unavailable") if values is None else MetricComputation(normalized_mutual_information(*values))


def hungarian_macro_f1(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("cluster and reference labels unavailable")
    reference, predicted = values
    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
    from sklearn.metrics import f1_score

    reference_labels = sorted(set(reference))
    predicted_labels = sorted(set(predicted))
    matrix = np.zeros((len(reference_labels), len(predicted_labels)), dtype=int)
    reference_index = {value: index for index, value in enumerate(reference_labels)}
    predicted_index = {value: index for index, value in enumerate(predicted_labels)}
    for reference_value, predicted_value in zip(reference, predicted, strict=True):
        matrix[reference_index[reference_value], predicted_index[predicted_value]] += 1
    rows, columns = linear_sum_assignment(-matrix)
    mapping = {
        predicted_labels[column]: reference_labels[row]
        for row, column in zip(rows, columns, strict=True)
    }
    mapped = np.asarray([mapping.get(value, "__unmatched__") for value in predicted])
    return MetricComputation(float(f1_score(reference, mapped, average="macro", zero_division=0)), metadata={"mapping": mapping})


def rare_label_recall(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    if values is None:
        return failed("cluster and reference labels unavailable")
    reference, predicted = values
    from sklearn.metrics import recall_score

    unique, counts = np.unique(reference, return_counts=True)
    rare = set(unique[counts <= max(2, int(np.percentile(counts, 25)))])
    return MetricComputation(
        float(np.mean(recall_score(reference, predicted, labels=sorted(rare), average=None, zero_division=0))),
        metadata={"rare_labels": sorted(rare)},
    )


def fms(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    return failed("cluster and reference labels unavailable") if values is None else MetricComputation(fowlkes_mallows(*values))


def stability(context: ScientificMetricContext) -> MetricComputation:
    clusterings = context.metadata.get("clusterings")
    if not clusterings or len(clusterings) < 2:
        return failed("at least two deterministic clusterings are required")
    values = [adjusted_rand(clusterings[0], candidate) for candidate in clusterings[1:]]
    return MetricComputation(float(np.mean(values)))


def cluster_silhouette(context: ScientificMetricContext) -> MetricComputation:
    values = _inputs(context)
    representation = embedding(context)
    return (
        failed("cluster labels and embedding unavailable")
        if values is None or representation is None
        else MetricComputation(silhouette(representation, values[1]), metadata={"representation": "best_available"})
    )


def clustering_definitions() -> list[tuple[MetricDefinition, Any]]:
    """Return the clustering catalog."""
    return [
        (_definition("clustering.ari", "Adjusted Rand index", MetricRole.PRIMARY), ari),
        (_definition("clustering.ami", "Adjusted/normalized mutual information", MetricRole.PRIMARY), ami),
        (_definition("clustering.hungarian_macro_f1", "Hungarian-matched macro F1", MetricRole.PRIMARY), hungarian_macro_f1),
        (_definition("clustering.rare_recall", "Rare-label recall", MetricRole.PRIMARY), rare_label_recall),
        (_definition("clustering.fowlkes_mallows", "Fowlkes-Mallows", MetricRole.SECONDARY), fms),
        (_definition("clustering.stability_ari", "Stability ARI", MetricRole.SECONDARY), stability),
        (_definition("clustering.silhouette", "Silhouette", MetricRole.DIAGNOSTIC), cluster_silhouette),
    ]


__all__ = ["clustering_definitions"]
