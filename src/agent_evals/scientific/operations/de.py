"""Differential-expression operation backed by Scanpy."""

from typing import Any

from agent_evals.scientific.context import OperationOutput, ScientificContext


def differential_expression(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:
    """Rank genes for a declared observation-group contrast."""
    import scanpy as sc

    groupby = str(parameters.get("groupby", parameters.get("group_key", "cell_type")))
    if groupby not in context.adata.obs:
        raise ValueError(f"DE group column '{groupby}' is not present")
    groups = parameters.get("groups")
    reference = parameters.get("reference")
    kwargs: dict[str, Any] = {
        "groupby": groupby,
        "method": str(parameters.get("method", "wilcoxon")),
    }
    if groups is not None:
        kwargs["groups"] = groups if isinstance(groups, list) else [str(groups)]
    if reference is not None:
        kwargs["reference"] = str(reference)
    sc.tl.rank_genes_groups(context.adata, **kwargs)
    group = (groups[0] if isinstance(groups, list) and groups else groups) or str(context.adata.obs[groupby].iloc[0])
    table = sc.get.rank_genes_groups_df(context.adata, group=group)
    table = table.rename(columns={"names": "gene", "logfoldchanges": "effect_size", "pvals": "p_value", "pvals_adj": "q_value"})
    artifact = context.artifact_store.save_table(
        "differential_expression", table, metadata={"groupby": groupby, "group": group}
    )
    return OperationOutput(artifacts=[artifact], outputs={"groupby": groupby, "group": group, "rows": int(table.shape[0])})
