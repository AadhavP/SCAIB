"""Shared evaluator-side extraction helpers."""

from __future__ import annotations

import math
from typing import Any

from agent_evals.core.de_evidence import (
    DE_SCORED_GROUP,
    DE_TABLE_ARTIFACT,
    DE_TABLE_EFFECT_COLUMN,
    DE_TABLE_GENE_COLUMNS,
    DE_TABLE_GROUP_COLUMN,
    REFERENCE_EFFECT_SIZES,
    REFERENCE_MARKERS,
)
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


def reference_labels(context: ScientificMetricContext) -> Any | None:
    """Return evaluator-held biological labels for conservation metrics.

    Integration metrics compare an embedding against the biological structure
    supplied by the benchmark; they do not require the agent to submit a second
    prediction just to score whether the representation preserved that structure.
    This helper intentionally never reads an agent-produced label as the
    reference side.
    """
    table = context.reference_artifacts.get("labels")
    if table is not None:
        if hasattr(table, "columns") and "reference_label" in table.columns:
            return table["reference_label"].astype(str).to_numpy()
        if hasattr(table, "to_numpy"):
            values = table.to_numpy()
            return values[:, 0] if getattr(values, "ndim", 1) > 1 else values
    adata = context.adata
    if adata is None:
        return None
    reference_key = next(
        (key for key in REFERENCE_LABEL_COLUMNS if key in adata.obs),
        None,
    )
    return (
        adata.obs[reference_key].astype(str).to_numpy()
        if reference_key is not None
        else None
    )


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
    """Extract a candidate/reference representation from either execution tier."""
    if name is not None:
        if name in context.candidate_artifacts:
            return context.candidate_artifacts[name]
        if context.adata is not None and name in context.adata.obsm:
            return context.adata.obsm[name]
    for key in ("embedding", "integrated_embedding", "X_integrated", "X_pca", "X_umap"):
        if key in context.candidate_artifacts:
            return context.candidate_artifacts[key]
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
    """Extract the agent's ranked genes and the evaluator's reference markers.

    The candidate ranking comes from the agent's own DE table and from nowhere
    else. An earlier version fell back to ``adata.uns["rank_genes_groups"]`` when
    no table was present, which cannot distinguish a block the agent computed from
    one the dataset shipped -- and pbmc68k ships that key precomputed with
    ``groupby="bulk_labels"``, so on the benchmark's own fixture the fallback read
    the answer key and scored it as the agent's answer. Same family as the ``uns``
    redaction finding in Stage 7.
    """
    table = context.candidate_artifacts.get(DE_TABLE_ARTIFACT)
    reference = context.metadata.get(REFERENCE_MARKERS)
    if table is None or not reference:
        return None
    ranked = _ranked_genes(table, context.metadata.get(DE_SCORED_GROUP))
    if ranked is None:
        return None
    return ranked, {str(value) for value in reference}


def _ranked_genes(table: Any, scored_group: Any) -> list[str] | None:
    """Read one ranking out of a candidate DE table, narrowing to one group.

    ``rank_genes_groups_df(adata, group=None)`` stacks every tested group into one
    frame, so a table carrying a group column holds several rankings concatenated.
    Reading it flat would build a mixed ranking whose top-K is whichever group
    sorted first, and precision@K against a single population's markers would then
    measure row order rather than biology.

    A table with no group column is one ranking already and is read whole. A table
    that has the column but not the requested group yields an empty ranking, which
    is the honest reading: the agent characterized populations, and none of them
    was the one being scored.
    """
    columns = getattr(table, "columns", ())
    column = next((name for name in DE_TABLE_GENE_COLUMNS if name in columns), None)
    if column is None:
        return None
    if scored_group is not None and DE_TABLE_GROUP_COLUMN in columns:
        table = table[table[DE_TABLE_GROUP_COLUMN].astype(str) == str(scored_group)]
    return [str(value) for value in table[column].tolist()]


def de_effect_sizes(
    context: ScientificMetricContext,
) -> tuple[dict[str, float], dict[str, float]] | None:
    """Pair candidate and reference effect sizes over the genes both carry.

    Narrowed to the scored group for the same reason :func:`_ranked_genes` is, and
    the consequence here is sharper: the previous per-gene lookup was
    ``table.loc[table["gene"] == gene].iloc[0]``, which on a multi-group table
    silently answered with whichever group happened to be listed first. So the
    correlation was computed against an arbitrary population's fold changes.

    Returns ``None`` when either side is absent, and drops any gene whose
    candidate value is not finite -- one NaN poisons Pearson's r for the whole run.
    """
    table = context.candidate_artifacts.get(DE_TABLE_ARTIFACT)
    reference = context.metadata.get(REFERENCE_EFFECT_SIZES)
    if table is None or not reference:
        return None
    columns = getattr(table, "columns", ())
    gene_column = next((name for name in DE_TABLE_GENE_COLUMNS if name in columns), None)
    if gene_column is None or DE_TABLE_EFFECT_COLUMN not in columns:
        return None
    scored_group = context.metadata.get(DE_SCORED_GROUP)
    if scored_group is not None and DE_TABLE_GROUP_COLUMN in columns:
        table = table[table[DE_TABLE_GROUP_COLUMN].astype(str) == str(scored_group)]
    reference_values = {str(gene): float(value) for gene, value in reference.items()}
    candidate: dict[str, float] = {}
    genes = table[gene_column].tolist()
    effects = table[DE_TABLE_EFFECT_COLUMN].tolist()
    for gene, effect in zip(genes, effects, strict=False):
        name = str(gene)
        if name not in reference_values or name in candidate or effect is None:
            continue
        value = float(effect)
        if math.isfinite(value):
            candidate[name] = value
    return candidate, reference_values


__all__ = [
    "de_effect_sizes",
    "de_ranked",
    "embedding",
    "failed",
    "labels",
    "reference_labels",
    "unavailable",
]
