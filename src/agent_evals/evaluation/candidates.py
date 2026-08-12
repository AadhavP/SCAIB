"""Assembly of the candidate and reference inputs a metric engine consumes.

Extracted so that the per-step scientific state and the final outcome are built
from the same code. If they were assembled separately, ``dS_t`` could drift from
``S_final`` for reasons that have nothing to do with the agent -- a column
resolved one way mid-run and another way at the end -- and the trajectory score
would be measuring the harness rather than the science.

Nothing here reads a reference column into the candidate side. The prediction and
grouping column names arrive already resolved by the caller, which is what keeps
the "only what this agent wrote" rule in one place.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent_evals.core.reference_columns import REFERENCE_LABEL_COLUMNS

#: Placeholder written when the agent produced no prediction column. A row per
#: cell is still emitted so the metric engine sees a well-formed artifact and
#: reports a real score of zero, rather than a missing artifact that would be
#: indistinguishable from a benchmark wiring fault.
UNASSIGNED_LABEL = "__unassigned__"

#: Embedding the evaluator scores when the agent produced one.
CANDIDATE_EMBEDDING_KEY = "X_pca"


@runtime_checkable
class ScientificStateProvider(Protocol):
    """Read-only view of live scientific state, without a package dependency.

    ``ScientificContext`` satisfies this structurally. Declaring the shape here
    rather than importing it keeps ``evaluation`` and ``scientific`` disjoint --
    they have no import edge in either direction today, and the free-execution
    tier will need to supply its own implementation over a file on disk rather
    than an in-memory object.
    """

    @property
    def adata(self) -> Any:
        """The dataset in its current state."""

    def agent_prediction_column(self) -> str | None:
        """Name the column this agent wrote its predictions to, if any."""

    def agent_cluster_column(self) -> str | None:
        """Name the column this agent wrote its grouping to, if any."""


def build_prediction_frame(adata: Any, prediction_column: str | None) -> Any:
    """Return a cell-indexed prediction table for the metric engine."""
    import pandas as pd

    cell_ids = [str(value) for value in adata.obs_names]
    labels = (
        [str(value) for value in adata.obs[prediction_column]]
        if prediction_column is not None
        else [UNASSIGNED_LABEL] * len(cell_ids)
    )
    return pd.DataFrame({"cell_id": cell_ids, "predicted_label": labels})


def build_candidate_artifacts(
    adata: Any,
    *,
    prediction_column: str | None,
    cluster_column: str | None,
    prediction: Any | None = None,
) -> dict[str, Any]:
    """Collect everything the agent produced that a metric may be scored on.

    A key is present only when the agent actually produced its input. Supplying
    an empty stand-in instead would turn "this was never attempted" into "this
    was attempted and scored zero", which are different claims about the run.
    """
    frame = (
        build_prediction_frame(adata, prediction_column)
        if prediction is None
        else prediction
    )
    artifacts: dict[str, Any] = {"prediction": frame}
    if cluster_column is not None:
        artifacts["cluster_labels"] = adata.obs[cluster_column].astype(str).to_numpy()
    if CANDIDATE_EMBEDDING_KEY in adata.obsm:
        artifacts["embedding"] = adata.obsm[CANDIDATE_EMBEDDING_KEY]
    return artifacts


def build_reference_artifacts(adata: Any) -> dict[str, Any]:
    """Collect the held-out reference labels, when this dataset carries them.

    Returns an empty mapping for a redacted dataset, which makes every
    reference-consuming metric structurally ineligible rather than scoring it
    against absent truth.
    """
    import pandas as pd

    column = next(
        (name for name in REFERENCE_LABEL_COLUMNS if name in adata.obs),
        None,
    )
    if column is None:
        return {}
    return {
        "labels": pd.DataFrame(
            {"reference_label": adata.obs[column].astype(str).to_numpy()}
        )
    }


__all__ = [
    "CANDIDATE_EMBEDDING_KEY",
    "UNASSIGNED_LABEL",
    "ScientificStateProvider",
    "build_candidate_artifacts",
    "build_prediction_frame",
    "build_reference_artifacts",
]
