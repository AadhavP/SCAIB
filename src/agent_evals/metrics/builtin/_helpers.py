"""Shared evaluator-side extraction helpers."""

from __future__ import annotations

from typing import Any

from agent_evals.core.reference_columns import (
    AGENT_PREDICTION_COLUMNS,
    REFERENCE_LABEL_COLUMNS,
)
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.registry import MetricComputation

#: Observation columns that may carry an agent's own labelling, in preference
#: order. A column here is only usable once the agent is known to have written
#: it; see :func:`labels`.
_CANDIDATE_COLUMNS = (*AGENT_PREDICTION_COLUMNS, "leiden", "louvain", "cluster")


def labels(context: ScientificMetricContext) -> tuple[Any, Any] | None:
    """Return reference and candidate labels from standardized evidence."""
    table = context.candidate_artifacts.get("prediction")
    if table is not None and context.reference_artifacts.get("labels") is not None:
        reference = context.reference_artifacts["labels"]
        if hasattr(reference, "columns") and "reference_label" in reference.columns:
            reference = reference["reference_label"].to_numpy()
        elif hasattr(reference, "to_numpy"):
            reference = reference.to_numpy()
        return reference, table["predicted_label"].to_numpy()
    adata = context.adata
    if adata is None:
        return None
    reference_key = next(
        (key for key in REFERENCE_LABEL_COLUMNS if key in adata.obs),
        None,
    )
    # Only a column this run's agent actually wrote may stand in as its
    # prediction. Without that check a dataset shipping its own `louvain` is
    # scored as an agent result, and under free execution the agent can simply
    # write the answer key into a column named `cluster`.
    predicted_key = next(
        (
            key
            for key in _CANDIDATE_COLUMNS
            if key in adata.obs and key in context.agent_produced_columns
        ),
        None,
    )
    if reference_key is None or predicted_key is None:
        return None
    return (
        adata.obs[reference_key].astype(str).to_numpy(),
        adata.obs[predicted_key].astype(str).to_numpy(),
    )


def embedding(context: ScientificMetricContext, name: str | None = None) -> Any | None:
    """Extract a candidate/reference representation."""
    if name is not None:
        if name in context.candidate_artifacts:
            return context.candidate_artifacts[name]
        if context.adata is not None and name in context.adata.obsm:
            return context.adata.obsm[name]
    if context.adata is not None:
        for key in ("X_integrated", "X_pca", "X_umap"):
            if key in context.adata.obsm:
                return context.adata.obsm[key]
    return None


def failed(reason: str) -> MetricComputation:
    """Return a computation failure with machine-readable reason."""
    return MetricComputation(raw_value=None, metadata={"failure_reason": reason})


def unavailable(reason: str) -> MetricComputation:
    """Return "this deployment cannot compute this metric at all".

    Distinct from :func:`failed`, which means an implementation ran on the
    agent's inputs and got nothing out of them. This one is a statement about
    SCAIB, so the metric is dropped from scoring rather than scored zero.
    """
    return MetricComputation(
        raw_value=None,
        metadata={"unavailable_reason": reason},
        unavailable=True,
    )


def de_ranked(context: ScientificMetricContext) -> tuple[list[str], set[str]] | None:
    """Extract ranked genes and evaluator-owned reference markers."""
    table = context.candidate_artifacts.get("de_table")
    reference = context.metadata.get("reference_markers")
    if table is not None:
        column = "gene" if "gene" in table.columns else "names"
        ranked = [str(value) for value in table[column].tolist()]
    elif context.adata is not None and "rank_genes_groups" in context.adata.uns:
        names = context.adata.uns["rank_genes_groups"]["names"]
        group = names.dtype.names[0] if getattr(names.dtype, "names", None) else None
        ranked = [str(row[group] if group else row) for row in names]
    else:
        return None
    if not reference:
        return None
    return ranked, {str(value) for value in reference}


__all__ = ["de_ranked", "embedding", "failed", "labels", "unavailable"]
