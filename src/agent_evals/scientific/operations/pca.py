"""Principal-component representation."""

from typing import Any

from agent_evals.scientific.context import OperationOutput, ScientificContext


def pca(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:
    """Compute a finite PCA representation with Scanpy."""
    import scanpy as sc

    requested = int(parameters.get("n_comps", parameters.get("n_components", 50)))
    n_comps = max(1, min(requested, int(context.adata.n_obs) - 1, int(context.adata.n_vars) - 1))
    sc.tl.pca(context.adata, n_comps=n_comps, use_highly_variable=parameters.get("use_highly_variable", None), svd_solver="arpack")
    artifact = context.artifact_store.save_adata(
        "pca_anndata", context.adata, metadata={"embedding": "X_pca", "n_comps": n_comps}
    )
    return OperationOutput(artifacts=[artifact], outputs={"embedding": "X_pca", "n_comps": n_comps})
