"""Assign cell-type labels from marker evidence, never from reference labels.

This operation is the only way an agent can produce the prediction column that
annotation scoring consumes.  It deliberately refuses to read held-out
reference biology: the label for each cluster is chosen from the agent-supplied
marker vocabulary by scoring mean expression of each label's marker genes.
That keeps the benchmark honest -- a high score requires the agent to have
picked sensible markers, not to have copied the answer key.
"""

from typing import Any

from agent_evals.core.reference_columns import RESERVED_REFERENCE_COLUMNS
from agent_evals.scientific.context import OperationOutput, ScientificContext
from agent_evals.scientific.operations.cluster import CLUSTER_COLUMN

#: Observation column holding the agent's per-cell prediction.
PREDICTION_COLUMN = "predicted_labels"

#: Label assigned when no supplied marker gene is expressed in a cluster.
UNASSIGNED_LABEL = "unassigned"


def _grouping_column(context: ScientificContext, parameters: dict[str, Any]) -> str:
    """Resolve the cluster column, refusing reference-label shortcuts."""
    requested = parameters.get("group_key", parameters.get("groupby"))
    if requested is not None:
        name = str(requested)
        if name in RESERVED_REFERENCE_COLUMNS:
            raise ValueError(
                f"annotation may not group on reference column '{name}'; "
                "cluster the data first and annotate the resulting groups"
            )
        if name not in context.adata.obs:
            raise ValueError(f"grouping column '{name}' is not present")
        if name not in context.agent_produced_columns:
            raise ValueError(
                f"grouping column '{name}' was not produced by this run; "
                "annotation must group on an agent-produced clustering"
            )
        return name
    if CLUSTER_COLUMN in context.adata.obs:
        return CLUSTER_COLUMN
    raise ValueError(
        "no agent-produced cluster column is available; run a clustering "
        "action before annotate"
    )


def _marker_map(parameters: dict[str, Any]) -> dict[str, list[str]]:
    """Read the label -> marker-gene mapping the agent must supply."""
    markers = parameters.get("markers", parameters.get("marker_genes"))
    if not isinstance(markers, dict) or not markers:
        raise ValueError(
            "annotate requires a 'markers' mapping of cell-type label to marker "
            "gene list; annotation cannot be scored without stated evidence"
        )
    resolved: dict[str, list[str]] = {}
    for label, genes in markers.items():
        if isinstance(genes, str):
            resolved[str(label)] = [genes]
        else:
            resolved[str(label)] = [str(gene) for gene in genes]
    return resolved


def annotate(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:
    """Label each cluster with the cell type whose markers it most expresses."""
    import numpy as np
    import pandas as pd

    grouping = _grouping_column(context, parameters)
    markers = _marker_map(parameters)
    vocabulary = parameters.get("label_vocabulary")
    if vocabulary:
        allowed = {str(label) for label in vocabulary}
        unknown = set(markers) - allowed
        if unknown:
            raise ValueError(
                f"marker labels {sorted(unknown)} are outside the declared "
                f"label_vocabulary {sorted(allowed)}"
            )

    var_names = {str(name).upper(): str(name) for name in context.adata.var_names}
    groups = context.adata.obs[grouping].astype(str)
    assignments: dict[str, str] = {}
    scores: dict[str, dict[str, float]] = {}
    for group in sorted(groups.unique()):
        mask = (groups == group).to_numpy()
        group_scores: dict[str, float] = {}
        for label, genes in markers.items():
            present = [var_names[gene.upper()] for gene in genes if gene.upper() in var_names]
            if not present:
                continue
            values = context.adata[mask, present].X
            dense = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
            group_scores[label] = float(np.nanmean(dense)) if dense.size else 0.0
        scores[group] = group_scores
        positive = {label: score for label, score in group_scores.items() if score > 0}
        assignments[group] = (
            max(positive, key=lambda label: positive[label]) if positive else UNASSIGNED_LABEL
        )

    predicted = groups.map(assignments).astype(str)
    context.adata.obs[PREDICTION_COLUMN] = pd.Categorical(predicted)
    table = pd.DataFrame(
        {
            "cell_id": [str(index) for index in context.adata.obs_names],
            "cluster": groups.to_numpy(),
            "predicted_label": predicted.to_numpy(),
        }
    )
    artifact = context.artifact_store.save_table(
        "cell_annotations",
        table,
        metadata={
            "column": PREDICTION_COLUMN,
            "grouping_column": grouping,
            "labels": sorted(set(predicted)),
            "cluster_assignments": assignments,
            "marker_scores": scores,
            "unassigned_cells": int((predicted == UNASSIGNED_LABEL).sum()),
        },
    )
    return OperationOutput(
        artifacts=[artifact],
        outputs={
            "column": PREDICTION_COLUMN,
            "grouping_column": grouping,
            "n_labels": len(set(predicted)),
            "unassigned_cells": int((predicted == UNASSIGNED_LABEL).sum()),
        },
    )


__all__ = ["PREDICTION_COLUMN", "UNASSIGNED_LABEL", "annotate"]
