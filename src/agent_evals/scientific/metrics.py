"""Objective metrics for the PBMC scientific benchmark vertical slice."""

from __future__ import annotations

from typing import Any

from agent_evals.benchmarks.schema import Direction
from agent_evals.evaluators.models import EvaluationLevel, MetricResult, MetricStatus


def _column(adata: Any, candidates: tuple[str, ...]) -> str | None:
    """Find the first available observation label column."""
    return next((name for name in candidates if name in adata.obs.columns), None)


def _embedding(adata: Any, candidates: tuple[str, ...] = ("X_integrated", "X_pca", "X_umap")) -> Any | None:
    """Find the best available low-dimensional representation."""
    return next((adata.obsm[name] for name in candidates if name in adata.obsm), None)


def _unavailable(metric_id: str, name: str, reason: str) -> MetricResult:
    return MetricResult(
        metric_id=metric_id,
        metric_name=name,
        level=EvaluationLevel.ARTIFACT,
        direction=Direction.HIGHER_IS_BETTER,
        status=MetricStatus.UNAVAILABLE,
        error=reason,
    )


def _success(
    metric_id: str,
    name: str,
    value: float,
    *,
    evidence: list[str],
    metadata: dict[str, Any] | None = None,
) -> MetricResult:
    return MetricResult(
        metric_id=metric_id,
        metric_name=name,
        level=EvaluationLevel.ARTIFACT,
        direction=Direction.HIGHER_IS_BETTER,
        raw_value=value,
        value=value,
        normalized_score=max(0.0, min(1.0, value)),
        status=MetricStatus.SUCCEEDED,
        evidence=evidence,
        metadata=metadata or {},
    )


def annotation_metrics(
    adata: Any,
    agent_produced_columns: set[str] | None = None,
) -> list[MetricResult]:
    """Compute ARI, NMI, and embedding silhouette against reference labels.

    ``agent_produced_columns`` restricts prediction candidates to columns the
    agent actually wrote. Without it, a dataset shipping its own ``louvain`` or
    ``bulk_labels`` column would be scored as if the agent had predicted it.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    reference_key = _column(adata, ("cell_type", "cell_type_ref", "known_labels", "bulk_labels"))
    prediction_candidates = ("predicted_labels", "predicted_cell_type", "leiden", "louvain", "cluster")
    if agent_produced_columns is None:
        predicted_key = None
    else:
        predicted_key = next(
            (
                name
                for name in prediction_candidates
                if name in agent_produced_columns and name in adata.obs.columns
            ),
            None,
        )
    if reference_key is None or predicted_key is None:
        reason = (
            "an agent-produced prediction column and reference labels are required"
            if reference_key is not None
            else "reference observation labels are required"
        )
        return [
            _unavailable(metric, name, reason)
            for metric, name in (
                ("annotation_ari", "Adjusted Rand index"),
                ("annotation_nmi", "Normalized mutual information"),
                ("annotation_silhouette", "Annotation silhouette"),
            )
        ]
    reference = adata.obs[reference_key].astype(str).to_numpy()
    predicted = adata.obs[predicted_key].astype(str).to_numpy()
    ari = float(adjusted_rand_score(reference, predicted))
    nmi = float(normalized_mutual_info_score(reference, predicted))
    embedding = _embedding(adata)
    silhouette = None if embedding is None or len(set(predicted)) < 2 else float(silhouette_score(embedding, predicted))
    results = [
        _success("annotation_ari", "Adjusted Rand index", ari, evidence=[f"obs.{reference_key}", f"obs.{predicted_key}"]),
        _success("annotation_nmi", "Normalized mutual information", nmi, evidence=[f"obs.{reference_key}", f"obs.{predicted_key}"]),
    ]
    if silhouette is None:
        results.append(_unavailable("annotation_silhouette", "Annotation silhouette", "at least two predicted groups and an embedding are required"))
    else:
        results.append(_success("annotation_silhouette", "Annotation silhouette", (silhouette + 1) / 2, evidence=["obsm embedding", f"obs.{predicted_key}"], metadata={"raw_silhouette": silhouette}))
    return results


def batch_metrics(adata: Any) -> list[MetricResult]:
    """Measure batch separation and biological conservation in an embedding."""
    from sklearn.metrics import silhouette_score

    batch_key = _column(adata, ("batch", "batch_id", "batch_labels"))
    biology_key = _column(adata, ("cell_type", "cell_type_ref", "known_labels", "bulk_labels"))
    embedding = _embedding(adata)
    if batch_key is None:
        return [
            _unavailable("batch_silhouette", "Batch silhouette", "a batch observation column is not present"),
            _unavailable("cell_type_silhouette", "Cell-type silhouette", "a batch observation column is not present"),
        ]
    if embedding is None:
        return [
            _unavailable("batch_silhouette", "Batch silhouette", "an embedding is required"),
            _unavailable("cell_type_silhouette", "Cell-type silhouette", "an embedding is required"),
        ]
    batch = adata.obs[batch_key].astype(str).to_numpy()
    if len(set(batch)) < 2:
        return [
            _unavailable("batch_silhouette", "Batch silhouette", "at least two batch groups are required"),
            _unavailable("cell_type_silhouette", "Cell-type silhouette", "at least two batch groups are required"),
        ]
    batch_raw = float(silhouette_score(embedding, batch))
    results = [_success("batch_silhouette", "Batch silhouette", 1 - ((batch_raw + 1) / 2), evidence=["obsm embedding", f"obs.{batch_key}"], metadata={"raw_silhouette": batch_raw})]
    if biology_key is None or len(set(adata.obs[biology_key].astype(str))) < 2:
        results.append(_unavailable("cell_type_silhouette", "Cell-type silhouette", "at least two biological label groups are required"))
    else:
        labels = adata.obs[biology_key].astype(str).to_numpy()
        raw = float(silhouette_score(embedding, labels))
        results.append(_success("cell_type_silhouette", "Cell-type silhouette", (raw + 1) / 2, evidence=["obsm embedding", f"obs.{biology_key}"], metadata={"raw_silhouette": raw}))
    return results


def differential_expression_metrics(adata: Any, parameters: dict[str, Any]) -> list[MetricResult]:
    """Compare ranked DE genes with a declared reference marker set."""
    reference_markers = {str(gene) for gene in parameters.get("reference_markers", [])}
    ranked: list[str] = []
    if "rank_genes_groups" in adata.uns:
        names = adata.uns["rank_genes_groups"].get("names")
        if names is not None:
            group = names.dtype.names[0] if getattr(names.dtype, "names", None) else None
            ranked = [str(row[group] if group else row) for row in names[: int(parameters.get("top_k", 50))]]
    if not reference_markers or not ranked:
        reason = "ranked DE genes and pipeline parameter reference_markers are required"
        return [
            _unavailable(metric, name, reason)
            for metric, name in (
                ("marker_overlap", "Marker overlap"),
                ("precision_at_k", "Marker precision@K"),
                ("auroc", "Marker AUROC"),
            )
        ]
    from sklearn.metrics import roc_auc_score

    k = min(int(parameters.get("top_k", 50)), len(ranked))
    top = ranked[:k]
    overlap = len(set(top) & reference_markers) / max(1, len(reference_markers))
    precision = len(set(top) & reference_markers) / max(1, k)
    truth = [int(gene in reference_markers) for gene in ranked]
    scores = list(range(len(ranked), 0, -1))
    auroc = float(roc_auc_score(truth, scores)) if len(set(truth)) == 2 else None
    results = [
        _success("marker_overlap", "Marker overlap", float(overlap), evidence=["rank_genes_groups", "reference_markers"]),
        _success("precision_at_k", "Marker precision@K", float(precision), evidence=[f"top_k={k}", "reference_markers"]),
    ]
    results.append(
        _unavailable("auroc", "Marker AUROC", "reference markers must include positive and negative ranked genes")
        if auroc is None
        else _success("auroc", "Marker AUROC", auroc, evidence=["rank_genes_groups", "reference_markers"])
    )
    return results


def compute_objective_metrics(
    benchmark_id: str,
    adata: Any,
    pipeline_parameters: dict[str, Any] | None = None,
    agent_produced_columns: set[str] | None = None,
) -> list[MetricResult]:
    """Dispatch the benchmark's objective metric family."""
    if "cell-annotation" in benchmark_id:
        return annotation_metrics(adata, agent_produced_columns)
    if "batch-correction" in benchmark_id:
        return batch_metrics(adata)
    if "differential-expression" in benchmark_id:
        return differential_expression_metrics(adata, pipeline_parameters or {})
    return []


def aggregate_objective_score(
    benchmark_id: str,
    metrics: list[MetricResult],
) -> float | None:
    """Aggregate only complete objective metric families into a final score."""
    values = {
        metric.metric_id: float(metric.normalized_score)
        for metric in metrics
        if metric.normalized_score is not None and metric.status == MetricStatus.SUCCEEDED
    }
    if "cell-annotation" in benchmark_id:
        required: tuple[str, ...] = (
            "annotation_ari",
            "annotation_nmi",
            "annotation_silhouette",
        )
        weights: tuple[float, ...] = (0.4, 0.4, 0.2)
    elif "batch-correction" in benchmark_id:
        required = ("batch_silhouette", "cell_type_silhouette")
        weights = (0.5, 0.5)
    elif "differential-expression" in benchmark_id:
        required = ("marker_overlap", "precision_at_k", "auroc")
        weights = (0.4, 0.3, 0.3)
    else:
        return None
    if not all(metric_id in values for metric_id in required):
        return None
    return sum(values[metric_id] * weight for metric_id, weight in zip(required, weights, strict=True))
