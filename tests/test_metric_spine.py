"""The scored metric spine: one engine, and inputs it can actually find.

Two defects motivate this module, and both were invisible to a green suite.

**The evaluator looked for a column the operations never wrote.** It resolved a
grouping from ``("leiden", "louvain")`` while ``cluster()`` wrote
``predicted_clusters``, so ``clustering.ari`` never found ``cluster_labels``,
failed to its ``failure_score`` of ``0.0``, and -- because the biology domain is
a geometric mean -- zeroed the outcome score and therefore the whole benchmark
score on every typed run since the first commit. Nothing raised, and no test
covered the join.

**The official score came from the engine without provenance.** Two engines
computed the same 39 metrics every run; the one whose results carry pinned
library versions had its score discarded, and the one that recorded only a
backend name produced the number. Unifying them is only safe if the surviving
engine's results key on the same names the profiles do -- ``metric_id``, not the
human-readable ``metric_name`` -- which is what the last test here pins down.
"""

from pathlib import Path

import pytest

pytest.importorskip("anndata")
pytest.importorskip("sklearn")

import anndata
import numpy as np
import pandas as pd

from agent_evals.core.reference_columns import (
    AGENT_CLUSTER_COLUMNS,
    RESERVED_REFERENCE_COLUMNS,
)
from agent_evals.evaluation.profiles import (
    pbmc_annotation_profile,
    pbmc_de_profile,
    pbmc_integration_profile,
)
from agent_evals.evaluation.scientific import ScientificMetricEngine
from agent_evals.metrics import metric_registry
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.results import MetricStatus
from agent_evals.scientific.artifacts.storage import LocalArtifactStore
from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.operations.cluster import CLUSTER_COLUMN, cluster

_CELLS = 60
_GENES = 6


def _context(tmp_path: Path) -> ScientificContext:
    """Build a context whose data ships both a reference label and a grouping."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            # The answer key, plus a grouping the *dataset* provides. Neither is
            # the agent's work, and the second one is the trap.
            "bulk_labels": pd.Categorical(
                ["T" if index % 2 else "B" for index in range(_CELLS)]
            ),
            "louvain": pd.Categorical([str(index % 3) for index in range(_CELLS)]),
        },
        index=[f"cell-{index}" for index in range(_CELLS)],
    )
    adata = anndata.AnnData(
        X=rng.random((_CELLS, _GENES), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["CD3D", "CD3E", "MS4A1", "CD79A", "GNLY", "LYZ"]),
    )
    return ScientificContext(
        adata=adata,
        dataset_metadata={"source": "synthetic"},
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        workspace=tmp_path,
    )


# --------------------------------------------------------------------------- #
# Finding the agent's grouping. The evaluator and the operations must name the
# same column, and a disagreement here does not raise -- it silently scores 0.
# --------------------------------------------------------------------------- #


def test_the_column_clustering_writes_is_one_the_evaluator_looks_for() -> None:
    """The whole defect in one assertion."""
    assert CLUSTER_COLUMN in AGENT_CLUSTER_COLUMNS


def test_no_grouping_column_is_a_reserved_reference_name() -> None:
    """A grouping the agent may write must never be one it may not."""
    assert not set(AGENT_CLUSTER_COLUMNS) & RESERVED_REFERENCE_COLUMNS


def test_clustering_then_recording_makes_the_grouping_resolvable(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assert context.agent_cluster_column() is None

    output = cluster(context, {"n_clusters": 3})
    context.record_produced_columns([str(output.outputs["column"])])

    assert context.agent_cluster_column() == CLUSTER_COLUMN


def test_a_grouping_the_dataset_shipped_is_not_the_agents(tmp_path: Path) -> None:
    """`louvain` present in obs is not evidence that this run clustered."""
    context = _context(tmp_path)
    assert "louvain" in context.adata.obs

    assert context.agent_cluster_column() is None


def test_an_agent_produced_leiden_column_counts_as_a_grouping(tmp_path: Path) -> None:
    """Under free execution the agent runs its own Scanpy, which writes `leiden`."""
    context = _context(tmp_path)
    context.adata.obs["leiden"] = pd.Categorical(
        [str(index % 4) for index in range(_CELLS)]
    )
    context.record_produced_columns(["leiden"])

    assert context.agent_cluster_column() == "leiden"


def test_a_recorded_column_absent_from_obs_is_not_resolved(tmp_path: Path) -> None:
    """Both halves are required: written by this agent *and* still there."""
    context = _context(tmp_path)
    context.record_produced_columns([CLUSTER_COLUMN])

    assert CLUSTER_COLUMN not in context.adata.obs
    assert context.agent_cluster_column() is None


def test_the_agents_grouping_is_preferred_over_a_shipped_one(tmp_path: Path) -> None:
    """With both present, the column this run wrote is the one that is scored."""
    context = _context(tmp_path)
    output = cluster(context, {"n_clusters": 3})
    context.record_produced_columns([str(output.outputs["column"])])

    # `louvain` is still sitting in obs and is earlier in nothing the agent wrote.
    assert context.agent_cluster_column() == CLUSTER_COLUMN


# --------------------------------------------------------------------------- #
# What the wiring buys: a metric that can compute instead of one that fails to
# its failure score and annihilates a geometric mean.
# --------------------------------------------------------------------------- #


def _metric_context(context: ScientificContext) -> ScientificMetricContext:
    """Assemble evaluator inputs the way the scientific loop does."""
    adata = context.adata
    cluster_column = context.agent_cluster_column()
    candidate: dict[str, object] = {
        "prediction": pd.DataFrame(
            {
                "cell_id": [str(name) for name in adata.obs_names],
                "predicted_label": [
                    str(value) for value in adata.obs[CLUSTER_COLUMN].astype(str)
                ],
            }
        )
    }
    if cluster_column is not None:
        candidate["cluster_labels"] = adata.obs[cluster_column].astype(str).to_numpy()
    return ScientificMetricContext(
        adata=adata,
        candidate_artifacts=candidate,
        reference_artifacts={
            "labels": pd.DataFrame(
                {"reference_label": adata.obs["bulk_labels"].astype(str).to_numpy()}
            )
        },
        agent_produced_columns=frozenset(context.agent_produced_columns),
    )


def test_clustering_ari_computes_once_the_grouping_is_wired(tmp_path: Path) -> None:
    """The end the wiring exists for, asserted against the real engine."""
    context = _context(tmp_path)
    output = cluster(context, {"n_clusters": 3})
    context.record_produced_columns([str(output.outputs["column"])])

    results, _, _, _ = ScientificMetricEngine().evaluate(
        ["clustering.ari"], _metric_context(context)
    )

    assert [result.status for result in results] == [MetricStatus.COMPUTED]
    assert results[0].normalized_value is not None


def test_without_a_grouping_clustering_ari_fails_rather_than_computing(
    tmp_path: Path,
) -> None:
    """The pre-fix behaviour, kept as the contrast that gives the fix meaning.

    ``failure_score`` is ``0.0``, so this result does not merely go unscored --
    it drags the biology geometric mean to zero and takes the benchmark score
    with it. That is why the wiring above matters more than its size suggests.
    """
    context = _context(tmp_path)
    context.adata.obs[CLUSTER_COLUMN] = pd.Categorical(
        [str(index % 3) for index in range(_CELLS)]
    )
    # Deliberately not recorded, so provenance is unproven and the grouping is
    # withheld from scoring.
    metric_context = _metric_context(context)
    assert "cluster_labels" not in metric_context.candidate_artifacts

    results, _, _, _ = ScientificMetricEngine().evaluate(
        ["clustering.ari"], metric_context
    )

    assert results[0].status is MetricStatus.FAILED
    assert results[0].missing_artifacts == ["cluster_labels"]
    assert results[0].normalized_value == 0.0


# --------------------------------------------------------------------------- #
# One engine feeds the score, so the names it emits must be the names the
# profiles ask for. This mismatch is the other kind that fails silently.
# --------------------------------------------------------------------------- #


def _unregistered(profile: object) -> set[str]:
    """Return the metric ids a profile names that no backend can compute."""
    registered = {definition.metric_id for definition in metric_registry.list()}
    groups = profile.metric_groups  # type: ignore[attr-defined]
    return {name for group in groups.values() for name in group.metrics} - registered


def test_the_scored_profiles_name_only_registered_metrics() -> None:
    """A profile naming something unregistered can only ever score `None`."""
    for profile in (pbmc_annotation_profile(), pbmc_integration_profile()):
        assert not _unregistered(profile), profile.benchmark


def test_the_de_profile_has_exactly_one_known_unregistered_metric() -> None:
    """A deliberately self-retiring test for a gap another stage owns.

    ``differential_expression.pseudobulk_recall`` is required by the DE profile
    and implemented nowhere, so the DE biology domain can only ever be
    unmeasured. That is survivable today only because profile resolution returns
    the annotation profile for every benchmark, which makes the DE profile
    unreachable -- so fixing resolution without registering this metric would
    trade a silently-wrong score for a silently-absent one.

    This asserts the hole is *exactly* one metric, so registering it fails this
    test and forces the general rule above to be widened rather than letting a
    second gap appear unnoticed.
    """
    assert _unregistered(pbmc_de_profile()) == {
        "differential_expression.pseudobulk_recall"
    }


def test_engine_results_are_keyed_by_the_id_the_profiles_use(tmp_path: Path) -> None:
    """`metric_id` is the join key; `metric_name` is a human title.

    Building the aggregator's inputs from ``metric_name`` would miss every
    profile lookup and report an unmeasured domain instead of a scored one.
    """
    context = _context(tmp_path)
    output = cluster(context, {"n_clusters": 3})
    context.record_produced_columns([str(output.outputs["column"])])
    profile_metrics = set(pbmc_annotation_profile().metric_groups["biology"].metrics)

    results, _, _, _ = ScientificMetricEngine().evaluate(
        sorted(profile_metrics), _metric_context(context)
    )

    assert {result.metric_id for result in results} == profile_metrics
    # And the human titles are genuinely different, so the two are not
    # interchangeable by accident.
    assert not {result.metric_name for result in results} & profile_metrics
