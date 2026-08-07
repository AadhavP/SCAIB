"""Tests for the typed scientific environment and episode contracts."""

from pathlib import Path
from typing import Any

import pytest

from agent_evals.benchmarks.io import load_benchmark
from agent_evals.environment import (
    ActionExecutionResult,
    ActionIntent,
    ActionStatus,
    ArtifactRecord,
    EpisodeStatus,
    Observation,
    ResourceUsage,
    RewardRecord,
    ScientificEnvironment,
)
from agent_evals.environment.ports import ExecutionContext

SPECIFICATION = load_benchmark(
    Path(__file__).parents[1] / "examples" / "benchmarks" / "pbmc-cell-annotation.yaml"
)


class FakeObservationBuilder:
    """Provide an in-memory AnnData placeholder for contract tests."""

    async def build(self, specification: Any, task: Any, snapshot: Any) -> list[Observation]:
        return [
            Observation(
                observation_id="current-anndata",
                value={"cells": 4, "genes": 3},
                source="fake-dataset",
            )
        ]


class FakeExecutor:
    """Return declared outputs without running a scientific tool."""

    def __init__(self, *, omit_outputs: bool = False, use_gpu: bool = False) -> None:
        self.omit_outputs = omit_outputs
        self.use_gpu = use_gpu

    async def execute(
        self,
        intent: ActionIntent,
        context: ExecutionContext,
    ) -> ActionExecutionResult:
        artifacts: list[ArtifactRecord] = []
        if intent.action_id == "qc" and not self.omit_outputs:
            artifacts.append(
                ArtifactRecord(
                    artifact_id="qc-table",
                    kind="table",
                    format="parquet",
                    validated=True,
                )
            )
        return ActionExecutionResult(
            intent_id=intent.intent_id,
            action_id=intent.action_id,
            status=ActionStatus.SUCCEEDED,
            artifacts=artifacts,
            resource_usage=ResourceUsage(gpu_used=self.use_gpu),
        )


class FakeRewardEvaluator:
    """Emit one deterministic reward for successful actions."""

    async def evaluate(
        self,
        specification: Any,
        task: Any,
        snapshot: Any,
        result: ActionExecutionResult,
    ) -> RewardRecord:
        return RewardRecord(value=1.0, strategy_id="annotation-reward", step=0)


def make_environment(executor: Any) -> ScientificEnvironment:
    """Construct the environment with only in-memory ports."""
    return ScientificEnvironment(
        SPECIFICATION,
        task_id="cell-annotation",
        executor=executor,
        observation_builder=FakeObservationBuilder(),
        reward_evaluator=FakeRewardEvaluator(),
    )


@pytest.mark.asyncio
async def test_reset_and_successful_step_record_episode_trace() -> None:
    """A valid action advances state, commits artifacts, and records reward."""
    environment = make_environment(FakeExecutor())
    initial = await environment.reset(seed=42, dataset_id="pbmc68k", episode_id="episode-1")

    assert initial.state.status == EpisodeStatus.RUNNING
    assert initial.state.observations["current-anndata"].visible_to_agent

    result = await environment.step(ActionIntent(action_id="qc"))

    assert result.accepted
    assert result.execution is not None
    assert result.execution.status == ActionStatus.SUCCEEDED
    assert result.observation.state.current_step == 1
    assert "qc-table" in result.observation.state.artifacts
    assert result.observation.state.rewards[0].value == 1.0
    assert any(event.event_type.value == "action.submitted" for event in result.observation.events)


@pytest.mark.asyncio
async def test_invalid_action_is_rejected_without_advancing_step() -> None:
    """Unknown or task-disallowed intents cannot reach the executor."""
    environment = make_environment(FakeExecutor())
    await environment.reset(seed=42, dataset_id="pbmc68k")

    result = await environment.step(ActionIntent(action_id="run-arbitrary-python"))

    assert not result.accepted
    assert result.execution is None
    assert result.observation.state.current_step == 0
    assert any(event.event_type.value == "action.rejected" for event in result.observation.events)


@pytest.mark.asyncio
async def test_failed_output_validation_does_not_commit_artifacts() -> None:
    """Executor protocol failures advance history but leave derived outputs unchanged."""
    environment = make_environment(FakeExecutor(omit_outputs=True))
    await environment.reset(seed=42, dataset_id="pbmc68k")

    result = await environment.step(ActionIntent(action_id="qc"))

    assert result.accepted
    assert result.execution is not None
    assert result.execution.status == ActionStatus.FAILED
    assert "qc-table" not in result.observation.state.artifacts
    assert result.observation.state.current_step == 1
    assert result.observation.state.status == EpisodeStatus.RUNNING


@pytest.mark.asyncio
async def test_resource_violation_is_failed_atomically() -> None:
    """CPU-only constraints prevent GPU-backed results from entering state."""
    environment = make_environment(FakeExecutor(use_gpu=True))
    await environment.reset(seed=42, dataset_id="pbmc68k")

    result = await environment.step(ActionIntent(action_id="qc"))

    assert result.execution is not None
    assert result.execution.status == ActionStatus.FAILED
    assert "GPU" in (result.execution.error or "")
    assert "qc-table" not in result.observation.state.artifacts


@pytest.mark.asyncio
async def test_episode_snapshot_is_safe_to_mutate_and_can_terminate() -> None:
    """Returned snapshots do not expose mutable internal episode state."""
    environment = make_environment(FakeExecutor())
    snapshot = await environment.reset(seed=42, dataset_id="pbmc68k")
    snapshot.state.observations.clear()

    unchanged = await environment.observe()
    assert "current-anndata" in unchanged.state.observations

    final = environment.terminate(status=EpisodeStatus.COMPLETED, reason="test complete")
    assert final.state.status == EpisodeStatus.COMPLETED
    assert final.state.finished_at is not None
    assert final.events[-1].event_type.value == "episode.terminated"

