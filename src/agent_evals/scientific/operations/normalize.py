"""Library-size normalization and log transformation."""

from typing import Any

from agent_evals.scientific.context import OperationOutput, ScientificContext


def normalize(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:
    """Normalize each cell and apply log1p using Scanpy."""
    import numpy as np
    import scanpy as sc

    target_sum = float(parameters.get("target_sum", 10_000))
    inplace = bool(parameters.get("inplace", True))
    if not inplace:
        raise ValueError("normalize requires inplace=true so the normalized AnnData remains available to the next action")
    values = context.adata.X
    minimum = float(np.nanmin(values.toarray() if hasattr(values, "toarray") else values))
    already_preprocessed = minimum < 0
    if not already_preprocessed:
        sc.pp.normalize_total(context.adata, target_sum=target_sum)
        sc.pp.log1p(context.adata)
    artifact = context.artifact_store.save_adata(
        "normalized_anndata",
        context.adata,
        metadata={
            "target_sum": target_sum,
            "log1p": not already_preprocessed,
            "input_already_preprocessed": already_preprocessed,
            "inplace": inplace,
        },
    )
    return OperationOutput(
        artifacts=[artifact],
        outputs={
            "target_sum": target_sum,
            "input_already_preprocessed": already_preprocessed,
            "inplace": inplace,
        },
    )
