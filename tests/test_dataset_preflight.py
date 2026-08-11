"""Dataset contract validation performed before any agent or model call."""

from pathlib import Path

import pytest

pytest.importorskip("anndata")

import anndata
import numpy as np
import pandas as pd

from agent_evals.benchmarks.io import load_benchmark
from agent_evals.datasets.preflight import (
    DatasetContractError,
    describe_readiness,
    validate_dataset_contract,
)

BENCHMARKS = Path(__file__).parents[1] / "examples" / "benchmarks"


def _adata(*, cells: int = 12, genes: int = 8, batches: int | None = None) -> anndata.AnnData:
    """Build a small in-memory object with optional batch structure."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(index=[f"cell-{index}" for index in range(cells)])
    if batches is not None:
        obs["batch"] = pd.Categorical(
            [f"batch-{index % batches}" for index in range(cells)]
        )
    return anndata.AnnData(
        X=rng.random((cells, genes), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(genes)]),
    )


def test_batch_benchmark_is_rejected_when_no_batch_column_exists() -> None:
    specification = load_benchmark(BENCHMARKS / "pbmc-batch-correction.yaml")
    task = specification.tasks[0]
    readiness = describe_readiness(_adata(), specification, task)

    assert readiness.has_batch_key is False
    with pytest.raises(DatasetContractError, match="requires batch labels"):
        validate_dataset_contract(readiness, specification, task)


def test_batch_benchmark_is_rejected_when_only_one_batch_is_present() -> None:
    specification = load_benchmark(BENCHMARKS / "pbmc-batch-correction.yaml")
    task = specification.tasks[0]
    readiness = describe_readiness(_adata(batches=1), specification, task)

    assert readiness.num_batches == 1
    assert readiness.has_batch_key is False
    with pytest.raises(DatasetContractError):
        validate_dataset_contract(readiness, specification, task)


def test_batch_benchmark_is_accepted_with_real_batch_structure() -> None:
    specification = load_benchmark(BENCHMARKS / "pbmc-batch-correction.yaml")
    task = specification.tasks[0]
    readiness = describe_readiness(_adata(batches=3), specification, task)

    assert readiness.has_batch_key is True
    assert readiness.num_batches == 3
    validate_dataset_contract(readiness, specification, task)


def test_reduced_scale_warns_but_does_not_block_a_smoke_run() -> None:
    """Small runs stay legal; they are just flagged as non-comparable."""
    specification = load_benchmark(BENCHMARKS / "pbmc-cell-annotation.yaml")
    task = specification.tasks[0]
    readiness = describe_readiness(_adata(cells=12), specification, task)

    assert readiness.scale_ratio is not None
    assert readiness.scale_ratio < 1
    assert any("not comparable" in warning for warning in readiness.warnings)
    validate_dataset_contract(readiness, specification, task)


def test_empty_dataset_is_always_rejected() -> None:
    specification = load_benchmark(BENCHMARKS / "pbmc-cell-annotation.yaml")
    task = specification.tasks[0]
    readiness = describe_readiness(_adata(cells=1), specification, task)
    empty = readiness.model_copy(update={"cells": 0})

    with pytest.raises(DatasetContractError, match="empty dataset"):
        validate_dataset_contract(empty, specification, task)
