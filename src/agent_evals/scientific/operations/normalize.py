"""Library-size normalization and log transformation."""

from typing import Any

from agent_evals.scientific.context import OperationOutput, ScientificContext

#: Rescale every cell to one fixed library size before ``log1p``.
LIBRARY_SIZE_LOG1P = "library_size_log1p"

#: Rescale to the dataset's own median library size before ``log1p`` -- scanpy's
#: default, which moves the observed depths less far than a fixed target does.
MEDIAN_COUNTS_LOG1P = "median_counts_log1p"

#: Library-size target used when the caller names none. Retained as the default
#: so a caller that sends only ``target_sum`` -- which is every caller written
#: before ``method`` existed -- gets exactly the behaviour it always got.
DEFAULT_TARGET_SUM = 10_000

#: The methods this operation actually implements. A benchmark that offers a
#: choice absent from here would fail an agent for the harness's gap, so the
#: declared ``choices`` in each catalog must be a subset of this.
NORMALIZATION_METHODS = (LIBRARY_SIZE_LOG1P, MEDIAN_COUNTS_LOG1P)


def normalize(context: ScientificContext, parameters: dict[str, Any]) -> OperationOutput:
    """Normalize each cell by the declared method and apply log1p using Scanpy."""
    import numpy as np
    import scanpy as sc

    method = str(parameters.get("method", LIBRARY_SIZE_LOG1P))
    if method not in NORMALIZATION_METHODS:
        raise ValueError(
            f"normalize does not implement method '{method}'; "
            f"expected one of {list(NORMALIZATION_METHODS)}"
        )
    # A target library size means nothing under median normalization, and
    # honouring it would silently deliver the other method. Refused rather than
    # ignored, because a parameter accepted and discarded is recorded as a
    # decision the agent made and the run then never took.
    if method == MEDIAN_COUNTS_LOG1P and "target_sum" in parameters:
        raise ValueError(
            f"method '{MEDIAN_COUNTS_LOG1P}' normalizes to the dataset's own median "
            "library size, so it accepts no target_sum"
        )
    target_sum = (
        float(parameters.get("target_sum", DEFAULT_TARGET_SUM))
        if method == LIBRARY_SIZE_LOG1P
        else None
    )
    inplace = bool(parameters.get("inplace", True))
    if not inplace:
        raise ValueError("normalize requires inplace=true so the normalized AnnData remains available to the next action")
    values = context.adata.X
    minimum = float(np.nanmin(values.toarray() if hasattr(values, "toarray") else values))
    already_preprocessed = minimum < 0
    if not already_preprocessed:
        # ``target_sum=None`` is how scanpy spells "normalize to the median", so
        # the two methods differ only in this argument.
        sc.pp.normalize_total(context.adata, target_sum=target_sum)
        sc.pp.log1p(context.adata)
    artifact = context.artifact_store.save_adata(
        "normalized_anndata",
        context.adata,
        metadata={
            "method": method,
            "target_sum": target_sum,
            "log1p": not already_preprocessed,
            "input_already_preprocessed": already_preprocessed,
            "inplace": inplace,
        },
    )
    return OperationOutput(
        artifacts=[artifact],
        outputs={
            "method": method,
            "target_sum": target_sum,
            "input_already_preprocessed": already_preprocessed,
            "inplace": inplace,
        },
    )
