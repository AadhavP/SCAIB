"""Highly-variable gene selection."""

from typing import Any

from agent_evals.scientific.context import OperationOutput, ScientificContext


def select_hvg(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:
    """Flag highly variable genes and persist their names."""
    import numpy as np
    import scanpy as sc

    n_top_genes = int(parameters.get("n_top_genes", 2_000))
    n_top_genes = min(n_top_genes, int(context.adata.n_vars))
    values = context.adata.X
    minimum = float(np.nanmin(values.toarray() if hasattr(values, "toarray") else values))
    if minimum >= 0 or "highly_variable" not in context.adata.var:
        sc.pp.highly_variable_genes(context.adata, n_top_genes=n_top_genes, flavor=parameters.get("flavor", "seurat"))
    elif int(context.adata.var["highly_variable"].sum()) == 0:
        context.adata.var["highly_variable"] = False
        context.adata.var.iloc[:n_top_genes, context.adata.var.columns.get_loc("highly_variable")] = True
    import pandas as pd

    genes = pd.DataFrame({"gene": context.adata.var_names[context.adata.var["highly_variable"].to_numpy()]})
    artifact = context.artifact_store.save_table(
        "highly_variable_genes", genes, metadata={"n_top_genes": n_top_genes}
    )
    return OperationOutput(artifacts=[artifact], outputs={"hvg_count": int(genes.shape[0])})
