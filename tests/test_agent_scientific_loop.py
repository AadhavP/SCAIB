"""Tests for agent observation, translation, rewards, and scientific episodes."""

from pathlib import Path

import pytest

pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from agent_evals.agents.trajectory import ScientificDecision
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.datasets.pbmc import PBMCDataset
from agent_evals.environment.ports import DeclarativeActionValidator
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.environment.scientific_loop import (
    ScientificActionExecutor,
    ScientificLoop,
)
from agent_evals.evaluators.rewards import RewardEvaluator
from agent_evals.scientific.action_mapper import (
    ActionMappingError,
    ScientificActionMapper,
)
from agent_evals.scientific.artifacts.storage import LocalArtifactStore
from agent_evals.scientific.context import ScientificContext
from agent_evals.scientific.observations import ScientificObservationBuilder

PBMC_CACHE = Path(".cache/datasets/pbmc68k_reduced.h5ad")


def _dataset() -> PBMCDataset:
    if not PBMC_CACHE.exists():
        pytest.skip("real PBMC cache is not available")
    return PBMCDataset(local_path=PBMC_CACHE)


async def _environment(tmp_path: Path) -> ScientificEnvironment:
    specification = load_benchmark("examples/benchmarks/pbmc-cell-annotation.yaml")
    task = specification.tasks[0]
    adata = _dataset().load(max_cells=48)
    context = ScientificContext(
        adata=adata,
        dataset_metadata={"organism": "Homo sapiens", "technology": "10x"},
        artifact_store=LocalArtifactStore(tmp_path / "artifacts"),
        workspace=tmp_path,
    )
    return ScientificEnvironment(
        specification,
        task_id=task.id,
        executor=ScientificActionExecutor(context),
        observation_builder=ScientificObservationBuilder(context),
        reward_evaluator=RewardEvaluator(),
    )


@pytest.mark.asyncio
async def test_observation_is_structured_and_hides_anndata(tmp_path: Path) -> None:
    environment = await _environment(tmp_path)
    snapshot = await environment.reset(seed=0, dataset_id="pbmc68k")
    value = snapshot.state.observations["scientific-observation"].value
    assert value["dataset_summary"]["cells"] == 48
    assert value["pipeline_state"]["qc_complete"] is False
    assert "X" not in value
    assert "qc" in value["available_actions"]


@pytest.mark.asyncio
async def test_action_mapper_accepts_valid_and_rejects_invalid_decisions(tmp_path: Path) -> None:
    environment = await _environment(tmp_path)
    snapshot = await environment.reset(seed=0, dataset_id="pbmc68k")
    mapper = ScientificActionMapper(DeclarativeActionValidator())
    decision = ScientificDecision(
        decision_id="decision-1",
        episode_id=snapshot.state.episode_id,
        step_id="step-1",
        order=0,
        decision_type="method_selection",
        action_category="quality_control",
        method="qc_filter",
        parameters={"min_genes": 200, "max_mito_fraction": 0.2},
        timestamp=snapshot.state.created_at,
    )
    intent = mapper.to_action_intent(decision, environment.specification, environment.task, snapshot)
    assert intent.action_id == "qc"
    with pytest.raises(ActionMappingError):
        mapper.to_action_intent(
            decision.model_copy(update={"method": "not-a-scientific-operation"}),
            environment.specification,
            environment.task,
            snapshot,
        )


@pytest.mark.asyncio
async def test_rule_based_scientific_loop_persists_rewards_and_report(tmp_path: Path) -> None:
    run = await ScientificLoop().run(
        "pbmc-cell-annotation",
        output_dir=tmp_path,
        max_cells=120,
    )
    root = tmp_path / run.run_id
    assert run.agent_run.succeeded
    assert len(run.trajectory) >= 2
    assert run.local_rewards
    assert run.global_reward.value is not None
    assert root.joinpath("trajectory.json").exists()
    assert root.joinpath("report.md").exists()
    assert all(Path(artifact.uri).exists() for artifact in run.artifacts if artifact.uri)
