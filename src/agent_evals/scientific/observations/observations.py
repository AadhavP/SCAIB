"""Structured, AnnData-free observations for scientific agents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.core.reference_columns import RESERVED_REFERENCE_COLUMNS
from agent_evals.environment.models import EpisodeSnapshot, Observation
from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.executor.scanpy import ScanpyExecutor


def visible_metadata_columns(obs: Any) -> list[str]:
    """List the observation columns an agent is allowed to know exist.

    A reference column is withheld by *name*, not merely by value. The column
    name is itself a disclosure: ``bulk_labels`` tells an agent which key holds
    the answer, and under free execution it names something the sanitized
    workspace copy does not contain -- so publishing it would be both a leak and
    a false statement about the dataset the agent actually has.

    This is the observation-builder half of the boundary. The filesystem half
    physically strips the columns; neither is a prompt instruction, because an
    instruction is not a boundary.
    """
    return [
        str(column)
        for column in obs.columns
        if str(column) not in RESERVED_REFERENCE_COLUMNS
    ]


class ScientificObservation(BaseModel):
    """The complete state representation exposed to an agent policy."""

    model_config = ConfigDict(extra="forbid")

    dataset_summary: dict[str, Any] = Field(default_factory=dict)
    quality_metrics: dict[str, Any] = Field(default_factory=dict)
    batch_information: dict[str, Any] = Field(default_factory=dict)
    biological_information: dict[str, Any] = Field(default_factory=dict)
    pipeline_state: dict[str, bool] = Field(default_factory=dict)
    available_actions: list[str] = Field(default_factory=list)
    #: Why each declared action is or is not currently selectable. An agent that
    #: can read the precondition does not have to infer it from a failure.
    action_preconditions: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ScientificObservationBuilder:
    """Convert scientific runtime state into typed observations."""

    def __init__(self, context: ScientificContext) -> None:
        self.context = context

    async def build(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
    ) -> list[Observation]:
        """Build only serializable summaries; never expose the AnnData object."""
        scientific = self.build_scientific(specification, task, snapshot)
        values: dict[str, Any] = {
            "scientific-observation": scientific.model_dump(mode="json"),
            "current-anndata": scientific.dataset_summary | {
                "quality_metrics": scientific.quality_metrics,
                "pipeline_state": scientific.pipeline_state,
            },
            "qc-statistics": scientific.quality_metrics,
            "batch-labels": scientific.batch_information.get("labels", []),
            "biological-labels": {
                "available": scientific.biological_information.get("reference_available", False),
                "hidden": True,
            },
            "pipeline-history": [
                operation.model_dump(mode="json") for operation in self.context.operations
            ],
            "available-tools": scientific.available_actions,
            # Accurate on the free tier too, even though ``context.adata`` is the
            # evaluator's unredacted object there: ``visible_metadata_columns``
            # withholds exactly the set ``datasets.redaction`` strips, both
            # deriving from ``core.reference_columns``, and redaction drops
            # columns rather than cells so the shape is identical either way.
            "dataset-summary": scientific.dataset_summary,
        }
        return [
            Observation(
                observation_id=observation_id,
                value=values[observation_id],
                source="scientific-observation-builder",
                step=snapshot.state.current_step,
                metadata={"ann_data_hidden": True},
            )
            # Serve what this builder has, and nothing for what it does not.
            # A ``values.get(observation_id, {})`` default published an empty
            # payload under every id a task declared, which is worse than absence
            # in two distinct ways: the agent worked blind while the harness
            # recorded the observation as delivered, and -- because observations
            # are stored by id -- the placeholder *overwrote* the real value
            # whenever another producer had already served it.
            for observation_id in dict.fromkeys(
                [*task.observations, "scientific-observation"]
            )
            if observation_id in values
        ]

    def build_scientific(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
    ) -> ScientificObservation:
        """Build a typed observation from AnnData metadata and operation history."""
        adata = self.context.adata
        obs = adata.obs
        dataset_metadata = dict(self.context.dataset_metadata)
        dataset_summary = {
            "cells": int(adata.n_obs),
            "genes": int(adata.n_vars),
            "organism": dataset_metadata.get("organism", "unknown"),
            "technology": dataset_metadata.get("technology", "unknown"),
            "assay": dataset_metadata.get("assay", "scRNA-seq"),
            "source": dataset_metadata.get("source", "unknown"),
            "metadata_columns": visible_metadata_columns(obs),
        }
        quality_metrics = self._quality_metrics(obs)
        batch_information = self._categorical_information(
            obs,
            ("batch", "batch_id", "batch_labels"),
            "batch",
        )
        biological_information = self._hidden_biological_information(
            obs,
            ("cell_type", "cell_type_ref", "known_labels", "bulk_labels"),
        )
        operation_names = {
            operation.operation
            for operation in self.context.operations
            if operation.status == "succeeded"
        }
        pipeline_state = {
            "qc_complete": bool({"qc", "qc_filter"} & operation_names),
            "normalized": "normalize" in operation_names,
            "hvg_selected": bool({"select_hvg", "hvg"} & operation_names),
            "pca_complete": "pca" in operation_names,
            "batch_corrected": bool({"batch_correct", "harmony"} & operation_names),
            "clustered": bool(
                {"cluster", "clustering", "leiden", "neighborhood-graph"} & operation_names
            ),
            "annotated": bool({"annotate", "annotation"} & operation_names),
            "differential_expression_complete": bool(
                {"differential_expression", "differential-expression", "marker-genes"}
                & operation_names
            ),
        }
        supported = set(ScanpyExecutor._operations)
        preconditions = {
            action.id: self._action_precondition(
                action.id,
                supported=supported,
                allowed=action.id in task.allowed_actions,
                pipeline_state=pipeline_state,
                batch_information=batch_information,
            )
            for action in specification.actions
        }
        available_actions = [
            action_id
            for action_id, precondition in preconditions.items()
            if precondition["available"]
        ]
        return ScientificObservation(
            dataset_summary=dataset_summary,
            quality_metrics=quality_metrics,
            batch_information=batch_information,
            biological_information=biological_information,
            pipeline_state=pipeline_state,
            available_actions=available_actions,
            action_preconditions=preconditions,
        )

    @staticmethod
    def _action_precondition(
        action_id: str,
        *,
        supported: set[str],
        allowed: bool,
        pipeline_state: dict[str, bool],
        batch_information: dict[str, Any],
    ) -> dict[str, Any]:
        """State whether an action can run now, and why not when it cannot."""
        if not allowed:
            return {"available": False, "reason": "not permitted by the task definition"}
        if action_id not in supported:
            return {"available": False, "reason": "no executor implements this operation"}
        if ScientificObservationBuilder._action_completed(action_id, pipeline_state):
            return {"available": False, "reason": "this pipeline stage already succeeded"}
        # Batch correction needs a real covariate with at least two levels;
        # advertising it against single-batch data invites a guaranteed failure.
        if action_id in {"harmony", "batch_correct"}:
            batches = int(batch_information.get("num_batches", 0) or 0)
            if batch_information.get("label_key") is None:
                return {
                    "available": False,
                    "reason": "no batch metadata column exists in this dataset",
                }
            if batches < 2:
                return {
                    "available": False,
                    "reason": (
                        f"batch column '{batch_information.get('label_key')}' has "
                        f"{batches} distinct value(s); at least 2 are required"
                    ),
                }
        # Annotation labels agent-produced groups, so a clustering must exist
        # first; otherwise the only groups available are reference labels.
        if action_id in {"annotate", "annotation"} and not pipeline_state.get("clustered", False):
            return {
                "available": False,
                "reason": "a clustering action must produce cell groups before annotation",
            }
        # Normalization must precede representation learning; scoring a PCA of
        # raw counts as though it were a considered choice is not meaningful.
        if action_id == "pca" and not pipeline_state.get("normalized", False):
            return {
                "available": False,
                "reason": "normalization must precede dimensionality reduction",
            }
        if action_id in {"cluster", "clustering", "leiden"} and not pipeline_state.get(
            "normalized", False
        ):
            return {
                "available": False,
                "reason": "normalization must precede clustering",
            }
        return {"available": True, "reason": "preconditions satisfied"}

    @staticmethod
    def _quality_metrics(obs: Any) -> dict[str, Any]:
        """Extract QC summaries while tolerating missing metadata columns."""
        counts = obs.get("total_counts", obs.get("n_counts"))
        genes = obs.get("n_genes_by_counts", obs.get("n_genes"))
        pct_mt = obs.get("pct_counts_mt", obs.get("percent_mito"))
        if pct_mt is not None:
            values = pct_mt.astype(float)
            mean_pct_mt = float(values.mean() * (100 if float(values.mean()) <= 1 else 1))
            low_quality_fraction = float((values > (0.15 if float(values.mean()) <= 1 else 15)).mean())
        else:
            mean_pct_mt = None
            low_quality_fraction = None
        return {
            "median_counts": float(counts.median()) if counts is not None else None,
            "median_genes": float(genes.median()) if genes is not None else None,
            "mean_pct_mt": mean_pct_mt,
            "low_quality_fraction": low_quality_fraction,
        }

    @staticmethod
    def _categorical_information(
        obs: Any,
        candidates: tuple[str, ...],
        label: str,
    ) -> dict[str, Any]:
        """Summarize the first available categorical column."""
        column = next((candidate for candidate in candidates if candidate in obs), None)
        if column is None:
            return {"label_key": None, "num_groups": 0, "labels": [], "balance": "unknown"}
        counts = obs[column].astype(str).value_counts()
        values = [str(value) for value in counts.index]
        if not values:
            balance = "unknown"
        elif float(counts.max()) / float(counts.min()) > 3:
            balance = "imbalanced"
        else:
            balance = "balanced"
        return {
            "label": label,
            "label_key": column,
            "num_batches" if label == "batch" else "num_groups": len(values),
            "labels": values,
            "balance": balance,
            "counts": {str(key): int(value) for key, value in counts.items()},
        }

    @staticmethod
    def _hidden_biological_information(obs: Any, candidates: tuple[str, ...]) -> dict[str, Any]:
        """Signal reference availability without leaking labels or counts."""
        column = next((candidate for candidate in candidates if candidate in obs), None)
        return {
            "reference_available": column is not None,
            "label_key": None,
            "num_groups": None,
            "labels": [],
            "counts": {},
            "hidden": True,
        }

    @staticmethod
    def _action_completed(action_id: str, state: dict[str, bool]) -> bool:
        """Prevent the baseline from repeating idempotent pipeline stages."""
        aliases = {
            "qc": "qc_complete",
            "normalize": "normalized",
            "select_hvg": "hvg_selected",
            "pca": "pca_complete",
            "harmony": "batch_corrected",
            "batch_correct": "batch_corrected",
            "cluster": "clustered",
            "clustering": "clustered",
            "leiden": "clustered",
            "neighborhood-graph": "clustered",
            "annotate": "annotated",
            "annotation": "annotated",
            "differential-expression": "differential_expression_complete",
            "marker-genes": "differential_expression_complete",
        }
        return state.get(aliases.get(action_id, ""), False)


__all__ = ["ScientificObservation", "ScientificObservationBuilder"]
