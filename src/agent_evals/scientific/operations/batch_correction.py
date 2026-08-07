"""Batch correction operations implemented with Scanpy integrations."""

from typing import Any

from agent_evals.scientific.context import OperationOutput, ScientificContext


def batch_correct(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:
    """Run Harmony on an existing PCA representation."""
    method = str(parameters.get("method", "harmony")).lower()
    if method in {"none", "identity"}:
        context.adata.obsm["X_integrated"] = context.adata.obsm["X_pca"].copy()
    elif method == "harmony":
        import scanpy.external as sce

        batch_key = str(parameters.get("batch_key", "batch"))
        if batch_key not in context.adata.obs:
            raise ValueError(f"batch metadata column '{batch_key}' is not present")
        sce.pp.harmony_integrate(context.adata, key=batch_key, basis=str(parameters.get("basis", "X_pca")))
        context.adata.obsm["X_integrated"] = context.adata.obsm["X_pca_harmony"]
    else:
        raise ValueError(f"unsupported batch correction method '{method}'")
    artifact = context.artifact_store.save_adata(
        "batch_corrected_anndata",
        context.adata,
        metadata={"method": method, "embedding": "X_integrated"},
    )
    return OperationOutput(artifacts=[artifact], outputs={"embedding": "X_integrated", "method": method})
