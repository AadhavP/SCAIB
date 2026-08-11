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
    # Scanpy raises when any tested group holds a single cell, which turns a
    # legitimate clustering into a hard tool failure. Restrict the contrast to
    # groups that can actually support a rank test.
    counts = context.adata.obs[groupby].astype(str).value_counts()
    testable = [str(name) for name, count in counts.items() if int(count) >= 2]
    if not testable:
        raise ValueError(
            f"no group in '{groupby}' contains at least 2 cells; "
            "differential expression requires replicated groups"
        )
    requested = (
        [str(group) for group in groups]
        if isinstance(groups, list)
        else ([str(groups)] if groups is not None else None)
    )
    selected = [group for group in requested if group in testable] if requested else testable
    if requested and not selected:
        raise ValueError(
            f"requested groups {requested} contain fewer than 2 cells in '{groupby}'"
        )
    if len(selected) < len(counts):
        kwargs["groups"] = selected
    if reference is not None:
        kwargs["reference"] = str(reference)
    sc.tl.rank_genes_groups(context.adata, **kwargs)
    # Report a group that was actually tested; the first observation's group may
    # have been excluded as a singleton.
    group = selected[0]
    table = sc.get.rank_genes_groups_df(context.adata, group=None)
    table = table.rename(
        columns={
            "names": "gene",
            "logfoldchanges": "effect_size",
            "pvals": "p_value",
            "pvals_adj": "q_value",
        }
    )
    excluded = [str(name) for name in counts.index if str(name) not in selected]
    artifact = context.artifact_store.save_table(
        "differential_expression",
        table,
        metadata={
            "groupby": groupby,
            "groups_tested": selected,
            "groups_excluded": excluded,
            "method": kwargs["method"],
        },
    )
    return OperationOutput(
        artifacts=[artifact],
        outputs={
            "groupby": groupby,
            "group": group,
            "groups_tested": selected,
            "groups_excluded": excluded,
            "rows": int(table.shape[0]),
        },
    )
