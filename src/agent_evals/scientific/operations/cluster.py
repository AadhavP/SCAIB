"""Unsupervised clustering that writes an agent-owned grouping column.

Clustering is the step that lets an agent group cells *without* consulting the
held-out reference biology.  The resulting column is written under an
agent-owned name so that downstream scoring can prove the grouping came from
the agent rather than from a label the dataset already shipped.
"""

from typing import Any

from agent_evals.scientific.context import OperationOutput, ScientificContext

#: Observation column this operation writes. Never a reserved reference name.
CLUSTER_COLUMN = "predicted_clusters"

#: Embeddings to cluster, most-integrated first.
EMBEDDING_CANDIDATES = ("X_integrated", "X_pca")


def _resolve_embedding(context: ScientificContext) -> str:
    """Return the embedding key to cluster, computing PCA when none exists."""
    for key in EMBEDDING_CANDIDATES:
        if key in context.adata.obsm:
            return key
    import scanpy as sc

    n_comps = max(
        1,
        min(20, int(context.adata.n_obs) - 1, int(context.adata.n_vars) - 1),
    )
    sc.tl.pca(context.adata, n_comps=n_comps, svd_solver="arpack")
    return "X_pca"


def _leiden_available() -> bool:
    """Report whether a graph-clustering backend is importable."""
    from importlib.util import find_spec

    return find_spec("leidenalg") is not None and find_spec("igraph") is not None


def cluster(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:
    """Group cells into unsupervised clusters using only expression data."""
    import numpy as np
    import pandas as pd

    embedding_key = _resolve_embedding(context)
    resolution = float(parameters.get("resolution", 1.0))
    requested = parameters.get("n_clusters")
    method = str(parameters.get("method", "leiden")).lower()
    seed = int(parameters.get("random_state", 0))

    if method in {"leiden", "louvain"} and _leiden_available():
        import scanpy as sc

        sc.pp.neighbors(
            context.adata,
            n_neighbors=int(parameters.get("n_neighbors", 15)),
            use_rep=embedding_key,
            random_state=seed,
        )
        sc.tl.leiden(
            context.adata,
            resolution=resolution,
            key_added=CLUSTER_COLUMN,
            random_state=seed,
        )
        backend = "leiden"
    else:
        # KMeans keeps the benchmark's `deterministic` constraint satisfiable on
        # installations without a graph-partitioning backend. It is a weaker
        # method than Leiden, so the artifact records which one actually ran.
        from sklearn.cluster import KMeans

        embedding = np.asarray(context.adata.obsm[embedding_key])
        n_clusters = int(requested) if requested is not None else max(2, round(8 * resolution))
        n_clusters = max(2, min(n_clusters, int(context.adata.n_obs)))
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(embedding)
        context.adata.obs[CLUSTER_COLUMN] = pd.Categorical([str(label) for label in labels])
        backend = "kmeans"

    counts = context.adata.obs[CLUSTER_COLUMN].astype(str).value_counts()
    table = pd.DataFrame(
        {
            "cell_id": [str(index) for index in context.adata.obs_names],
            "cluster": context.adata.obs[CLUSTER_COLUMN].astype(str).to_numpy(),
        }
    )
    artifact = context.artifact_store.save_table(
        "cluster_labels",
        table,
        metadata={
            "column": CLUSTER_COLUMN,
            "backend": backend,
            "embedding": embedding_key,
            "n_clusters": int(counts.shape[0]),
            "resolution": resolution,
        },
    )
    return OperationOutput(
        artifacts=[artifact],
        outputs={
            "column": CLUSTER_COLUMN,
            "backend": backend,
            "n_clusters": int(counts.shape[0]),
            "embedding": embedding_key,
        },
    )


__all__ = ["CLUSTER_COLUMN", "cluster"]
