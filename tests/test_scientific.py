"""Integration tests for the real PBMC/Scanpy vertical slice."""

from pathlib import Path

import pytest

pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from agent_evals.datasets.pbmc import PBMCDataset
from agent_evals.environment.models import ActionIntent, ActionStatus
from agent_evals.scientific.artifacts.storage import LocalArtifactStore
from agent_evals.scientific.benchmarks import register_scientific_benchmarks
from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.executor.scanpy import ScanpyExecutor
from agent_evals.scientific.pipeline import load_pipeline
from agent_evals.scientific.runner import ScientificPipelineRunner

PBMC_CACHE = Path(".cache/datasets/pbmc68k_reduced.h5ad")


def _dataset() -> PBMCDataset:
    if not PBMC_CACHE.exists():
        pytest.skip("real PBMC cache is not available; populate it before integration tests")
    return PBMCDataset(local_path=PBMC_CACHE)


def test_pbmc_loader_validates_real_anndata() -> None:
    dataset = _dataset()
    adata = dataset.load(max_cells=24)
    assert adata.n_obs == 24
    assert adata.n_vars > 0
    assert dataset.metadata.source == "scanpy.datasets.pbmc68k_reduced"
    assert "bulk_labels" in dataset.metadata.metadata_columns


def test_scanpy_executor_runs_real_operations(tmp_path: Path) -> None:
    adata = _dataset().load(max_cells=48)
    context = ScientificContext(
        adata=adata,
        dataset_metadata={"source": "scanpy.datasets.pbmc68k_reduced"},
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        workspace=tmp_path,
    )
    executor = ScanpyExecutor()
    results = [
        executor.execute(ActionIntent(action_id="qc_filter", parameters={"min_genes": 0}), context),
        executor.execute(ActionIntent(action_id="normalize", parameters={"target_sum": 10_000}), context),
        executor.execute(ActionIntent(action_id="select_hvg", parameters={"n_top_genes": 100}), context),
        executor.execute(ActionIntent(action_id="pca", parameters={"n_comps": 10}), context),
    ]
    assert all(result.status is ActionStatus.SUCCEEDED for result in results)
    assert "X_pca" in context.adata.obsm
    assert len(context.artifacts) == 4
    assert all(artifact.path.exists() for artifact in context.artifacts.values())


def test_scientific_runner_persists_reproducible_report(tmp_path: Path) -> None:
    register_scientific_benchmarks()
    report = ScientificPipelineRunner().run(
        "pbmc-cell-annotation",
        load_pipeline("configs/pipelines/pbmc_default.yaml"),
        output_dir=tmp_path,
        max_cells=48,
    )
    report_root = tmp_path / report.run_id
    assert report.trajectory
    assert report.metric_results
    assert report_root.joinpath("report.json").exists()
    assert report_root.joinpath("report.md").exists()
    assert any(metric.metric_id == "annotation_ari" for metric in report.metric_results)
