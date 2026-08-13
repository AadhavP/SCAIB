"""Tests for the granular, auditable QC action space."""

from pathlib import Path

import pytest

pytest.importorskip("anndata")
pytest.importorskip("scanpy")

import anndata
import numpy as np
import pandas as pd

from agent_evals.environment.models import ActionIntent, ActionStatus
from agent_evals.scientific.artifacts.storage import LocalArtifactStore
from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.executor.scanpy import ScanpyExecutor


def _context(tmp_path: Path) -> ScientificContext:
    values = np.array(
        [
            [10, 0, 0, 1],
            [8, 1, 0, 1],
            [7, 1, 1, 1],
            [1, 0, 0, 20],
            [2, 0, 0, 15],
            [9, 2, 1, 0],
        ],
        dtype=float,
    )
    adata = anndata.AnnData(
        X=values,
        obs=pd.DataFrame(index=[f"cell-{i}" for i in range(values.shape[0])]),
        var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC", "MT-CO1"]),
    )
    return ScientificContext(
        adata=adata,
        dataset_metadata={"organism": "Homo sapiens"},
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        workspace=tmp_path,
    )


@pytest.mark.parametrize(
    "method",
    ["fixed_threshold", "mitochondrial_filter", "adaptive_quantile", "mad_outlier"],
)
def test_each_declared_qc_strategy_executes_and_records_method(
    tmp_path: Path,
    method: str,
) -> None:
    context = _context(tmp_path / method)
    result = ScanpyExecutor().execute(
        ActionIntent(
            action_id="qc",
            parameters={
                "method": method,
                "min_genes": 1,
                "max_mito_fraction": 0.95,
            },
        ),
        context,
    )

    assert result.status is ActionStatus.SUCCEEDED
    assert result.outputs["method"] == method
    artifact = context.artifacts["qc_statistics"]
    assert artifact.metadata["method"] == method
    table = pd.read_csv(artifact.path)
    assert {"qc_pass", "qc_fail_reason"} <= set(table.columns)


def test_gene_detection_filter_does_not_remove_cells(tmp_path: Path) -> None:
    context = _context(tmp_path)
    result = ScanpyExecutor().execute(
        ActionIntent(action_id="qc", parameters={"method": "fixed_threshold", "min_cells": 6}),
        context,
    )

    assert result.status is ActionStatus.SUCCEEDED
    assert context.adata.n_obs == 6
    assert context.adata.n_vars == 1
