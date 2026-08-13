"""Quality-control strategies for AnnData objects.

QC is intentionally a method choice rather than one opaque filter. The
benchmark can therefore measure whether an agent selected fixed thresholds,
data-adaptive quantiles, robust MAD outlier detection, or a focused
mitochondrial strategy, and the resulting thresholds are preserved in the
artifact metadata for downstream audit.
"""

from typing import Any

from agent_evals.scientific.context import OperationOutput, ScientificContext

QC_METHODS = (
    "fixed_threshold",
    "mitochondrial_filter",
    "adaptive_quantile",
    "mad_outlier",
)


def qc_filter(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:  # noqa: C901
    """Calculate QC statistics and apply one declared filtering strategy.

    Existing callers that omit ``method`` retain fixed-threshold behavior. The
    QC table contains both retained and rejected cells, while the current
    AnnData object contains only retained cells.
    """
    import numpy as np
    import pandas as pd
    import scanpy as sc

    adata = context.adata
    cells_before = int(adata.n_obs)
    genes_before = int(adata.n_vars)
    values = adata.X
    dense_values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    minimum = float(np.nanmin(dense_values))
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
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"],
            inplace=True,
            percent_top=None,
            log1p=False,
        )
    else:
        sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=None, log1p=False)
        adata.obs["pct_counts_mt"] = 0.0

    method = str(parameters.get("method", "fixed_threshold")).lower()
    if method not in QC_METHODS:
        raise ValueError(f"unsupported QC method '{method}'; choose from {QC_METHODS}")

    cell_genes = adata.obs["n_genes_by_counts"].astype(float)
    cell_mito = adata.obs["pct_counts_mt"].astype(float)
    gene_keep = _gene_nonzero_counts(values) >= int(parameters.get("min_cells", 0))
    thresholds: dict[str, Any] = {"method": method}

    if method == "fixed_threshold":
        min_genes = int(parameters.get("min_genes", 0))
        max_genes = parameters.get("max_genes")
        max_pct_mt = parameters.get("max_pct_mt", parameters.get("max_mito_fraction"))
        max_pct_mt = _as_percent(max_pct_mt)
        cell_keep = cell_genes >= min_genes
        if max_genes is not None:
            cell_keep &= cell_genes <= int(max_genes)
        if max_pct_mt is not None:
            cell_keep &= cell_mito <= float(max_pct_mt)
        thresholds.update(
            {
                "min_genes": min_genes,
                "max_genes": max_genes,
                "max_pct_mt": max_pct_mt,
                "min_cells": int(parameters.get("min_cells", 0)),
            }
        )
    elif method == "mitochondrial_filter":
        max_pct_mt = _as_percent(
            parameters.get("max_pct_mt", parameters.get("max_mito_fraction", 20.0))
        )
        cell_keep = cell_mito <= float(max_pct_mt)
        if "min_genes" in parameters:
            cell_keep &= cell_genes >= int(parameters["min_genes"])
        thresholds.update(
            {
                "max_pct_mt": max_pct_mt,
                "min_genes": parameters.get("min_genes"),
                "min_cells": int(parameters.get("min_cells", 0)),
            }
        )
    elif method == "adaptive_quantile":
        lower_genes_q = float(parameters.get("min_genes_quantile", 0.05))
        upper_mito_q = float(parameters.get("max_mito_quantile", 0.95))
        lower_genes = float(cell_genes.quantile(lower_genes_q))
        upper_mito = float(cell_mito.quantile(upper_mito_q))
        cell_keep = (cell_genes >= lower_genes) & (cell_mito <= upper_mito)
        thresholds.update(
            {
                "min_genes_quantile": lower_genes_q,
                "max_mito_quantile": upper_mito_q,
                "derived_min_genes": lower_genes,
                "derived_max_pct_mt": upper_mito,
                "min_cells": int(parameters.get("min_cells", 0)),
            }
        )
    else:  # mad_outlier
        multiplier = float(parameters.get("mad_multiplier", 3.0))
        genes_median = float(cell_genes.median())
        genes_mad = _mad(cell_genes)
        mito_median = float(cell_mito.median())
        mito_mad = _mad(cell_mito)
        lower_genes = genes_median - multiplier * genes_mad
        upper_genes = genes_median + multiplier * genes_mad
        upper_mito = mito_median + multiplier * mito_mad
        cell_keep = (
            (cell_genes >= lower_genes)
            & (cell_genes <= upper_genes)
            & (cell_mito <= upper_mito)
        )
        thresholds.update(
            {
                "mad_multiplier": multiplier,
                "derived_min_genes": lower_genes,
                "derived_max_genes": upper_genes,
                "derived_max_pct_mt": upper_mito,
                "min_cells": int(parameters.get("min_cells", 0)),
            }
        )

    # Gene filtering changes the expression space for every retained cell, but
    # it is not a cell-quality failure. Keep the two decisions independent so a
    # sensible ``min_cells`` value cannot remove every cell by accident.
    cell_keep = cell_keep.astype(bool)
    fail_reasons = pd.Series("", index=adata.obs.index, dtype="object")
    if method in {"fixed_threshold", "mitochondrial_filter"}:
        if thresholds.get("min_genes") is not None:
            fail_reasons.loc[cell_genes < float(thresholds["min_genes"])] = "below_min_genes"
        if thresholds.get("max_genes") is not None:
            fail_reasons.loc[cell_genes > float(thresholds["max_genes"])] = "above_max_genes"
        if thresholds.get("max_pct_mt") is not None:
            fail_reasons.loc[cell_mito > float(thresholds["max_pct_mt"])] = "above_max_pct_mt"
    else:
        fail_reasons.loc[~cell_keep] = "outlier_by_" + method

    table = adata.obs.reset_index(names="cell_id")
    table["qc_pass"] = cell_keep.to_numpy()
    table["qc_fail_reason"] = fail_reasons.to_numpy()
    adata._inplace_subset_var(gene_keep)
    adata._inplace_subset_obs(cell_keep.to_numpy())
    if adata.n_obs == 0:
        raise ValueError("QC filters removed every cell")

    artifact = context.artifact_store.save_table(
        "qc_statistics",
        table,
        metadata={
            "cells_before": cells_before,
            "cells_after": int(adata.n_obs),
            "genes_before": genes_before,
            "genes_after": int(adata.n_vars),
            "method": method,
            "thresholds": thresholds,
            "input_already_preprocessed": already_preprocessed,
        },
    )
    summary = {
        "method": method,
        "cells_before": cells_before,
        "cells_after": int(adata.n_obs),
        "genes_before": genes_before,
        "genes_after": int(adata.n_vars),
        "cells_removed": cells_before - int(adata.n_obs),
        "genes_removed": genes_before - int(adata.n_vars),
        "thresholds": thresholds,
        "median_counts": float(adata.obs["total_counts"].median()),
        "median_genes": float(adata.obs["n_genes_by_counts"].median()),
        "median_pct_mt": float(adata.obs["pct_counts_mt"].median()),
    }
    return OperationOutput(artifacts=[artifact], outputs=summary)


def _as_percent(value: Any) -> float | None:
    """Accept either a fraction in [0, 1] or a percentage in [0, 100]."""
    if value is None:
        return None
    number = float(value)
    return number * 100 if number <= 1 else number


def _gene_nonzero_counts(values: Any) -> Any:
    """Return detected-cell counts per gene for dense and sparse matrices."""
    if hasattr(values, "getnnz"):
        return values.getnnz(axis=0).A1.astype(int)
    import numpy as np

    return (np.asarray(values) != 0).sum(axis=0).astype(int)


def _mad(values: Any) -> float:
    """Return a stable median absolute deviation, avoiding a zero cutoff."""
    import numpy as np

    median = float(values.median())
    mad = float(np.median(np.abs(values.to_numpy(dtype=float) - median)))
    return mad if mad > 0 else 1.0


__all__ = ["QC_METHODS", "qc_filter"]
