"""Quality-control filtering for AnnData objects."""

from typing import Any

from agent_evals.scientific.context import OperationOutput, ScientificContext


def qc_filter(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:  # noqa: C901
    """Calculate QC statistics and apply declared cell-level filters."""
    import numpy as np
    import scanpy as sc

    adata = context.adata
    before = int(adata.n_obs)
    values = adata.X
    minimum = float(np.nanmin(values.toarray() if hasattr(values, "toarray") else values))
    already_preprocessed = minimum < 0
    adata.var["mt"] = [str(name).upper().startswith("MT-") for name in adata.var_names]
    if already_preprocessed:
        if "n_genes_by_counts" not in adata.obs:
            if "n_genes" in adata.obs:
                adata.obs["n_genes_by_counts"] = adata.obs["n_genes"]
            else:
                adata.obs["n_genes_by_counts"] = (values != 0).sum(axis=1)
        if "total_counts" not in adata.obs:
            adata.obs["total_counts"] = adata.obs.get("n_counts", 0)
        if "pct_counts_mt" not in adata.obs:
            adata.obs["pct_counts_mt"] = adata.obs.get("percent_mito", 0) * 100
    elif bool(adata.var["mt"].any()):
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    else:
        sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
        adata.obs["pct_counts_mt"] = 0.0
    min_genes = int(parameters.get("min_genes", 0))
    max_genes = parameters.get("max_genes")
    max_pct_mt = parameters.get("max_pct_mt", parameters.get("max_mito_fraction", 1.0))
    if max_pct_mt is not None and float(max_pct_mt) <= 1:
        max_pct_mt = float(max_pct_mt) * 100
    keep = adata.obs["n_genes_by_counts"] >= min_genes
    if max_genes is not None:
        keep &= adata.obs["n_genes_by_counts"] <= int(max_genes)
    if max_pct_mt is not None:
        keep &= adata.obs["pct_counts_mt"] <= float(max_pct_mt)
    adata._inplace_subset_obs(keep.to_numpy())
    if adata.n_obs == 0:
        raise ValueError("QC filters removed every cell")
    table = adata.obs.reset_index(names="cell_id")
    artifact = context.artifact_store.save_table(
        "qc_statistics",
        table,
        metadata={
            "cells_before": before,
            "cells_after": int(adata.n_obs),
            "input_already_preprocessed": already_preprocessed,
        },
    )
    summary = {
        "cells_before": before,
        "cells_after": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "median_counts": float(adata.obs["total_counts"].median()),
        "median_genes": float(adata.obs["n_genes_by_counts"].median()),
        "median_pct_mt": float(adata.obs["pct_counts_mt"].median()),
    }
    return OperationOutput(artifacts=[artifact], outputs=summary)
