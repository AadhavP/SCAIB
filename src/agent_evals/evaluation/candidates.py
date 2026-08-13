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

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_evals.core.reference_columns import REFERENCE_LABEL_COLUMNS

#: Placeholder written when SCAIB ran the workflow and the agent still left no
#: prediction column behind. A row per cell is emitted so the metric engine sees
#: a well-formed artifact and reports a real score of zero: the agent had the
#: means to annotate and did not, which is a fact about the agent and belongs on
#: its score. This is *only* sound while the evaluator would have seen the column
#: had one been written -- see :func:`build_metric_inputs` for the tier where it
#: would not, and where using this placeholder manufactures a zero instead.
UNASSIGNED_LABEL = "__unassigned__"

#: Embedding the evaluator scores when the agent produced one.
CANDIDATE_EMBEDDING_KEY = "X_pca"

#: Recorded on every metric the missing join excludes, and on the evaluation
#: itself. One constant rather than a string at each call site because it is the
#: published explanation for an unmeasured outcome, and two wordings of the same
#: gap read as two different gaps.
UNJOINABLE_CANDIDATE_GAP = (
    "the agent implemented this task itself, so its outputs are workspace files "
    "the evaluator has not joined back onto the reference; reference-consuming "
    "metrics are structurally ineligible and the scientific outcome is unmeasured "
    "rather than zero"
)


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
    return {"prediction": frame, **_candidate_artifacts(adata, cluster_column)}


def _candidate_artifacts(adata: Any, cluster_column: str | None) -> dict[str, Any]:
    """Collect the candidate evidence that does not depend on a prediction."""
    artifacts: dict[str, Any] = {}
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


@dataclass(frozen=True)
class MetricInputs:
    """Both sides of a metric computation, assembled so they cannot disagree.

    The candidate and the reference are returned together because withholding one
    without the other is worse than withholding neither. Omitting an
    unobservable candidate on its own leaves the metric *eligible* with a missing
    input, which the engine scores at its failure value -- the same manufactured
    zero, reached by a longer route. Only an unavailable reference makes the
    metric structurally ineligible, so the two decisions are one decision.
    """

    candidate_artifacts: dict[str, Any]
    reference_artifacts: dict[str, Any]
    #: The prediction table, or ``None`` when there was no candidate to build one
    #: from. Callers that persist it as evidence must not persist a placeholder.
    prediction: Any | None
    #: Why the reference was withheld, threaded onto the metric context so the
    #: exclusion is recorded against its real cause.
    reference_join_gap: str | None = None

    @property
    def limitations(self) -> tuple[str, ...]:
        """The join gap in the list form the publishing fields want.

        Derived rather than stored, so it cannot disagree with
        :attr:`reference_join_gap` -- the same reason
        :data:`UNJOINABLE_CANDIDATE_GAP` is one constant.

        It carries *only* unmeasurability. An agent that could have written a
        prediction and did not is measured, at zero, and saying so here would put
        an explained score into fields whose whole job is to mean "this was not
        measured". That fact is recorded where it belongs instead: on the archived
        prediction's ``agent_produced_prediction`` metadata and in the metric
        result's own status.
        """
        return () if self.reference_join_gap is None else (self.reference_join_gap,)


def build_metric_inputs(
    adata: Any,
    *,
    prediction_column: str | None,
    cluster_column: str | None,
    evaluator_observes_predictions: bool,
) -> MetricInputs:
    """Assemble metric inputs, distinguishing the agent's gaps from the harness's.

    ``evaluator_observes_predictions`` answers one question: if this agent had
    produced a prediction, would it appear in the object scored here? On the typed
    tier SCAIB runs the science into that very object, so the answer is yes and an
    absent column is the agent's omission -- scored, via the
    :data:`UNASSIGNED_LABEL` placeholder. When the agent implements the workflow
    itself its results land in workspace files instead, the answer is no, and an
    absent column says nothing whatever about the agent.

    Scoring the second case measured a free-execution run's biology domain at
    exactly 0.0 on every run, for a join the harness has not built yet -- while
    the environment record shipped alongside it promised the outcome would be
    "reported as unmeasured rather than as zero". A benchmark that contradicts its
    own disclosure in the direction of a worse score is not conservative, it is
    wrong, and it is wrong identically for a brilliant agent and a broken one.
    """
    if prediction_column is None and not evaluator_observes_predictions:
        return MetricInputs(
            candidate_artifacts=_candidate_artifacts(adata, cluster_column),
            reference_artifacts={},
            prediction=None,
            reference_join_gap=UNJOINABLE_CANDIDATE_GAP,
        )
    prediction = build_prediction_frame(adata, prediction_column)
    return MetricInputs(
        candidate_artifacts=build_candidate_artifacts(
            adata,
            prediction_column=prediction_column,
            cluster_column=cluster_column,
            prediction=prediction,
        ),
        reference_artifacts=build_reference_artifacts(adata),
        prediction=prediction,
    )


__all__ = [
    "CANDIDATE_EMBEDDING_KEY",
    "UNASSIGNED_LABEL",
    "UNJOINABLE_CANDIDATE_GAP",
    "MetricInputs",
    "ScientificStateProvider",
    "build_candidate_artifacts",
    "build_metric_inputs",
    "build_prediction_frame",
    "build_reference_artifacts",
]
