"""Tests for agent observation, translation, rewards, and scientific episodes."""

from pathlib import Path

import pytest

pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from agent_evals.agents.runtime import (
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentObservation,
    AgentRuntime,
    AgentSession,
    FinalSubmission,
    agent_runtime_registry,
)
from agent_evals.agents.runtime.manager import _observation_from_snapshot
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


#: Marker panel a fake agent must supply; annotation cannot read the answer key.
FAKE_MARKERS = {
    "T": ["CD3D", "CD3E", "IL7R"],
    "B": ["MS4A1", "CD79A"],
    "NK": ["GNLY", "NKG7"],
    "Monocyte": ["CD14", "LYZ"],
}


class FakeScientificRuntime(AgentRuntime):
    """A runtime that stops early, leaving required artifacts unproduced."""

    def __init__(self) -> None:
        self.agent_id = "fake-scientific-runtime"
        self.manifest = AgentManifest(
            name="Fake scientific runtime",
            type="test-runtime",
            capabilities=["structured_actions"],
        )

    async def initialize(self, context: AgentContext) -> AgentSession:
        return AgentSession(context=context, state={"step": 0})

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        del observation
        actions = ["qc", "normalize", "finish"]
        index = int(session.state.get("step", 0))
        session.state["step"] = index + 1
        return AgentAction(action_type=actions[index])

    async def terminate(
        self,
        session: AgentSession,
        observation: AgentObservation | None = None,
    ) -> FinalSubmission:
        del session, observation
        return FinalSubmission(summary="completed")


class CompleteScientificRuntime(FakeScientificRuntime):
    """A runtime that produces every required artifact before terminating."""

    def __init__(self) -> None:
        super().__init__()
        self.agent_id = "complete-scientific-runtime"
        self.manifest = AgentManifest(
            name="Complete scientific runtime",
            type="test-runtime",
            capabilities=["structured_actions"],
        )

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        del observation
        plan: list[tuple[str, dict[str, object]]] = [
            ("qc", {"min_genes": 200, "max_mito_fraction": 0.2}),
            ("normalize", {"target_sum": 10_000}),
            ("pca", {"n_components": 10}),
            ("cluster", {"resolution": 0.5}),
            ("marker-genes", {"group_key": "predicted_clusters"}),
            (
                "annotate",
                {
                    "label_vocabulary": sorted(FAKE_MARKERS),
                    "markers": FAKE_MARKERS,
                },
            ),
            ("finish", {}),
        ]
        index = int(session.state.get("step", 0))
        session.state["step"] = index + 1
        action_type, parameters = plan[index]
        return AgentAction(action_type=action_type, parameters=parameters)


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
    agent_observation = _observation_from_snapshot(snapshot, environment.task)
    assert agent_observation.metadata["goal"] == environment.task.objective
    assert agent_observation.metadata["scenario"]["success_criteria"]


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


@pytest.mark.asyncio
async def test_scientific_loop_runs_universal_runtime_agents(tmp_path: Path) -> None:
    runtime_name = "complete-scientific-runtime"
    if runtime_name not in agent_runtime_registry.list():
        agent_runtime_registry.register(
            runtime_name, CompleteScientificRuntime, capabilities=["structured_actions"]
        )

    run = await ScientificLoop().run(
        "pbmc-cell-annotation",
        agent_type=runtime_name,
        output_dir=tmp_path,
        max_cells=120,
        max_steps=8,
    )

    assert run.agent_run.succeeded
    assert run.agent_run.manifest is not None
    assert run.agent_run.manifest.name == "Complete scientific runtime"
    assert (tmp_path / run.run_id / "agent_run.json").exists()


@pytest.mark.asyncio
async def test_terminating_without_required_artifacts_does_not_report_success(
    tmp_path: Path,
) -> None:
    """A run that stops before producing declared artifacts must not pass."""
    runtime_name = "fake-scientific-runtime"
    if runtime_name not in agent_runtime_registry.list():
        agent_runtime_registry.register(
            runtime_name, FakeScientificRuntime, capabilities=["structured_actions"]
        )

    run = await ScientificLoop().run(
        "pbmc-cell-annotation",
        agent_type=runtime_name,
        output_dir=tmp_path,
        max_cells=120,
        max_steps=4,
    )

    assert not run.agent_run.succeeded
    assert any(
        "required" in failure.message and "artifact" in failure.message
        for failure in run.agent_run.failures
    )


@pytest.mark.asyncio
async def test_annotation_score_requires_an_agent_produced_prediction(
    tmp_path: Path,
) -> None:
    """Reference labels shipped with the dataset must never be scored as output."""
    runtime_name = "fake-scientific-runtime"
    if runtime_name not in agent_runtime_registry.list():
        agent_runtime_registry.register(
            runtime_name, FakeScientificRuntime, capabilities=["structured_actions"]
        )

    run = await ScientificLoop().run(
        "pbmc-cell-annotation",
        agent_type=runtime_name,
        output_dir=tmp_path,
        max_cells=120,
        max_steps=4,
    )

    # The PBMC object ships `bulk_labels` and `louvain`; neither may produce a score.
    assert run.global_reward.value is None
    assert run.global_reward.status == "unavailable"
