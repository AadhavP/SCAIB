"""Reference labels are evaluator inputs and must never be scored as output."""

from pathlib import Path

import pytest

pytest.importorskip("anndata")
pytest.importorskip("scanpy")
pytest.importorskip("sklearn")

import anndata
import numpy as np
import pandas as pd

from agent_evals.scientific.artifacts.storage import LocalArtifactStore
from agent_evals.scientific.context import (
    AGENT_PREDICTION_COLUMNS,
    RESERVED_REFERENCE_COLUMNS,
    ScientificContext,
)
from agent_evals.scientific.metrics import annotation_metrics
from agent_evals.scientific.operations.annotate import PREDICTION_COLUMN, annotate
from agent_evals.scientific.operations.cluster import CLUSTER_COLUMN, cluster


def _context(tmp_path: Path) -> ScientificContext:
    """Build a context whose data already ships reference labels."""
    rng = np.random.default_rng(0)
    cells, genes = 40, 6
    matrix = rng.random((cells, genes), dtype=np.float32)
    obs = pd.DataFrame(
        {
            # These are the answer key. Nothing may score them as a prediction.
            "bulk_labels": pd.Categorical(["T" if index % 2 else "B" for index in range(cells)]),
            "louvain": pd.Categorical([str(index % 3) for index in range(cells)]),
        },
        index=[f"cell-{index}" for index in range(cells)],
    )
    adata = anndata.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=["CD3D", "CD3E", "MS4A1", "CD79A", "GNLY", "LYZ"]),
    )
    return ScientificContext(
        adata=adata,
        dataset_metadata={"source": "synthetic"},
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        workspace=tmp_path,
    )


def test_dataset_reference_columns_are_not_scored_as_predictions(tmp_path: Path) -> None:
    """With no agent-produced column, annotation metrics must be unavailable."""
    context = _context(tmp_path)
    results = annotation_metrics(context.adata, set(context.agent_produced_columns))

    assert [result.status.value for result in results] == ["unavailable"] * 3
    assert all("agent-produced" in (result.error or "") for result in results)


def test_preexisting_cluster_column_alone_cannot_produce_a_score(tmp_path: Path) -> None:
    """`louvain` shipped with the data is not evidence the agent clustered."""
    context = _context(tmp_path)
    assert "louvain" in context.adata.obs.columns

    results = annotation_metrics(context.adata, agent_produced_columns=set())

    assert all(result.status.value == "unavailable" for result in results)


def test_recording_a_reserved_reference_column_is_refused(tmp_path: Path) -> None:
    context = _context(tmp_path)
    for column in sorted(RESERVED_REFERENCE_COLUMNS):
        with pytest.raises(ValueError, match="reserved reference column"):
            context.record_produced_columns([column])
    assert not context.agent_produced_columns


def test_annotation_refuses_to_group_on_reference_labels(tmp_path: Path) -> None:
    """Annotating the answer key directly must fail, not score perfectly."""
    context = _context(tmp_path)

    with pytest.raises(ValueError, match="may not group on reference column"):
        annotate(
            context,
            {"group_key": "bulk_labels", "markers": {"T": ["CD3D"], "B": ["MS4A1"]}},
        )


def test_annotation_refuses_a_grouping_the_agent_did_not_produce(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with pytest.raises(ValueError, match="not produced by this run"):
        annotate(
            context,
            {"group_key": "louvain", "markers": {"T": ["CD3D"], "B": ["MS4A1"]}},
        )


def test_annotation_requires_stated_marker_evidence(tmp_path: Path) -> None:
    context = _context(tmp_path)
    cluster_output = cluster(context, {"n_clusters": 3})
    context.record_produced_columns([str(cluster_output.outputs["column"])])

    with pytest.raises(ValueError, match="requires a 'markers' mapping"):
        annotate(context, {})


def test_cluster_then_annotate_yields_a_scoreable_agent_prediction(tmp_path: Path) -> None:
    """The only legitimate path: cluster, then label clusters from markers."""
    context = _context(tmp_path)
    cluster_output = cluster(context, {"n_clusters": 3})
    context.record_produced_columns([str(cluster_output.outputs["column"])])
    assert cluster_output.outputs["column"] == CLUSTER_COLUMN

    annotate_output = annotate(
        context,
        {
            "markers": {"T": ["CD3D", "CD3E"], "B": ["MS4A1", "CD79A"]},
            "label_vocabulary": ["T", "B"],
        },
    )
    context.record_produced_columns([str(annotate_output.outputs["column"])])

    assert annotate_output.outputs["column"] == PREDICTION_COLUMN
    assert PREDICTION_COLUMN in AGENT_PREDICTION_COLUMNS
    assert context.agent_prediction_column() == PREDICTION_COLUMN

    results = annotation_metrics(context.adata, set(context.agent_produced_columns))
    ari = next(result for result in results if result.metric_id == "annotation_ari")
    assert ari.status.value == "succeeded"
    # Scored against the held-out reference, not against itself.
    assert f"obs.{PREDICTION_COLUMN}" in ari.evidence
    assert "obs.bulk_labels" in ari.evidence
