"""Tests for trajectory intelligence signals."""

from pathlib import Path

import pytest

from agent_evals.agents import (
    AgentConfiguration,
    AgentHarness,
    MockActionExecutor,
    MockAgentAdapter,
    MockObservationBuilder,
)
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.evaluation.trajectory import TrajectoryEvaluator


@pytest.mark.asyncio
async def test_trajectory_reports_adaptation_and_is_reproducible() -> None:
    specification = load_benchmark(Path(__file__).parents[1] / "examples" / "benchmarks" / "pbmc-cell-annotation.yaml")
    environment = ScientificEnvironment(
        specification,
        task_id="cell-annotation",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )
    run = await AgentHarness().run(MockAgentAdapter(), environment, AgentConfiguration(agent_type="mock", seed=1))
    evaluator = TrajectoryEvaluator()
    first = evaluator.evaluate(run, specification.tasks[0], 0.5, local_rewards=[0.7])
    second = evaluator.evaluate(run, specification.tasks[0], 0.5, local_rewards=[0.7])

    assert first == second
    assert 0 <= first.adaptation_ability <= 1
    assert first.formula
