"""Benchmark dataset contract validation performed before any agent runs.

A benchmark declares the scientific shape of the data it needs
(``DatasetSpecification.expected_observations``) and the observations a task
must be able to see.  Historically the runtime loaded one reduced PBMC object
for every benchmark, so a batch-correction task could start against data with
no batch column at all -- the agent was then asked to diagnose a failure the
harness had created.  This module turns that class of failure into an explicit
error raised *before* a provider call is made.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.benchmarks.schema import BenchmarkSpecification, TaskSpecification
from agent_evals.core.reference_columns import REFERENCE_LABEL_COLUMNS

#: Observation IDs that require a usable technical batch covariate.
BATCH_OBSERVATION_IDS = frozenset({"batch-labels"})

#: Candidate observation columns for a technical batch covariate.
BATCH_COLUMN_CANDIDATES = ("batch", "batch_id", "batch_labels", "donor", "sample")

#: Candidate observation columns holding held-out reference biology. Aliases the
#: canonical list so preflight, redaction, and scoring cannot drift apart.
REFERENCE_COLUMN_CANDIDATES = REFERENCE_LABEL_COLUMNS


class DatasetContractError(RuntimeError):
    """Raised when loaded data cannot satisfy the benchmark's declared contract."""


class DatasetReadiness(BaseModel):
    """Structured, agent-visible statement of what the loaded data supports."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    cells: int
    genes: int
    source: str = "unknown"
    has_batch_key: bool = False
    batch_key: str | None = None
    num_batches: int = 0
    candidate_batch_columns: list[str] = Field(default_factory=list)
    has_reference_labels: bool = False
    reference_key: str | None = None
    expected_observations: dict[str, Any] = Field(default_factory=dict)
    scale_ratio: float | None = None
    warnings: list[str] = Field(default_factory=list)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def describe_readiness(
    adata: Any,
    specification: BenchmarkSpecification,
    task: TaskSpecification,
    *,
    source: str = "unknown",
) -> DatasetReadiness:
    """Summarize which benchmark prerequisites the loaded object satisfies."""
    obs_columns = [str(column) for column in adata.obs.columns]
    batch_key = next(
        (name for name in BATCH_COLUMN_CANDIDATES if name in obs_columns), None
    )
    num_batches = (
        int(adata.obs[batch_key].astype(str).nunique()) if batch_key is not None else 0
    )
    reference_key = next(
        (name for name in REFERENCE_COLUMN_CANDIDATES if name in obs_columns), None
    )
    declared = next(
        (
            dataset
            for dataset in specification.datasets
            if not task.datasets or dataset.id in task.datasets
        ),
        specification.datasets[0] if specification.datasets else None,
    )
    expected = dict(declared.expected_observations) if declared is not None else {}
    expected_cells = _int_or_none(expected.get("cells"))
    cells = int(adata.n_obs)
    scale_ratio = (
        cells / expected_cells if expected_cells not in (None, 0) else None
    )
    warnings: list[str] = []
    if scale_ratio is not None and scale_ratio < 1.0:
        warnings.append(
            f"loaded {cells} cells against a declared {expected_cells}; "
            "scores are not comparable to a full-dataset run"
        )
    # A batch covariate with a single level cannot express batch structure, so
    # treat it as absent rather than reporting a key the agent cannot use.
    if batch_key is not None and num_batches < 2:
        warnings.append(
            f"observation column '{batch_key}' has {num_batches} distinct value(s); "
            "batch correction is not applicable"
        )
    return DatasetReadiness(
        dataset_id=declared.id if declared is not None else "unknown",
        cells=cells,
        genes=int(adata.n_vars),
        source=source,
        has_batch_key=batch_key is not None and num_batches >= 2,
        batch_key=batch_key,
        num_batches=num_batches,
        candidate_batch_columns=[
            name for name in BATCH_COLUMN_CANDIDATES if name in obs_columns
        ],
        has_reference_labels=reference_key is not None,
        reference_key=reference_key,
        expected_observations=expected,
        scale_ratio=scale_ratio,
        warnings=warnings,
    )


def validate_dataset_contract(
    readiness: DatasetReadiness,
    specification: BenchmarkSpecification,
    task: TaskSpecification,
) -> None:
    """Raise when the loaded data cannot support the task's required observations.

    Only *hard* contract violations raise. A reduced-scale dataset is reported
    as a warning on the readiness record, because smoke runs are legitimate;
    a missing batch covariate for a batch-correction task is not.
    """
    required_observations = {
        observation.id
        for observation in specification.observations
        if observation.required and observation.id in set(task.observations)
    }
    if readiness.cells < 1 or readiness.genes < 1:
        raise DatasetContractError(
            f"benchmark '{specification.metadata.id}' loaded an empty dataset "
            f"({readiness.cells} cells x {readiness.genes} genes)"
        )
    if required_observations & BATCH_OBSERVATION_IDS and not readiness.has_batch_key:
        detail = (
            f"observation column '{readiness.batch_key}' has only "
            f"{readiness.num_batches} distinct value(s)"
            if readiness.batch_key is not None
            else "no batch observation column is present"
        )
        raise DatasetContractError(
            f"benchmark '{specification.metadata.id}' task '{task.id}' requires batch "
            f"labels, but {detail}. Resolved dataset "
            f"'{readiness.dataset_id}' ({readiness.cells} cells x {readiness.genes} "
            "genes) cannot satisfy this benchmark; the run was stopped before any "
            "model call."
        )


__all__ = [
    "BATCH_COLUMN_CANDIDATES",
    "BATCH_OBSERVATION_IDS",
    "REFERENCE_COLUMN_CANDIDATES",
    "DatasetContractError",
    "DatasetReadiness",
    "describe_readiness",
    "validate_dataset_contract",
]
