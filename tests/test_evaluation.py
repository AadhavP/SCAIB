"""Tests for the scientific task and granular evaluation vertical slice."""

from pathlib import Path

import pytest

from agent_evals.agents import (
    AgentConfiguration,
    AgentHarness,
    MockActionExecutor,
    MockAgentAdapter,
    MockObservationBuilder,
)
from agent_evals.agents.trajectory import AgentRun
from agent_evals.benchmarks.io import benchmark_from_dict, load_benchmark
from agent_evals.benchmarks.schema import Direction
from agent_evals.core.exceptions import RegistryError
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.evaluators import (
    EvaluationEngine,
    EvaluationLevel,
    MetricRegistry,
    MetricStatus,
)
from agent_evals.evaluators.builtin import execution_success
from agent_evals.evaluators.registry import MetricComputation, MetricContext

SPECIFICATION = load_benchmark(
    Path(__file__).parents[1] / "examples" / "benchmarks" / "pbmc-batch-correction.yaml"
)


def make_environment() -> ScientificEnvironment:
    return ScientificEnvironment(
        SPECIFICATION,
        task_id="batch-correction",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )


async def run_policy(policy: str) -> AgentRun:
    return await AgentHarness().run(
        MockAgentAdapter(),
        make_environment(),
        AgentConfiguration(
            agent_type="mock",
            seed=7,
            metadata={"mock_policy": policy},
        ),
    )


@pytest.mark.asyncio
async def test_good_policy_produces_hierarchical_decisions_and_report() -> None:
    run = await run_policy("good")
    report = EvaluationEngine().evaluate(SPECIFICATION, run)

    assert run.succeeded
    assert len(run.trajectory.decisions.decisions) > len(run.final_environment_state.state.actions)
    assert any(item.decision_type == "method_selection" for item in report.decisions.decisions)
    assert any(item.decision_type == "parameter_selection" for item in report.decisions.decisions)
    assert all(item.status == MetricStatus.SUCCEEDED for item in report.metric_results)
    assert report.summary.successful_actions == 4
    # Zero, not four. The mock produces artifact *records* and no files, so there
    # is nothing for the validator to read and no check can pass. This asserted 4
    # while the mock set ``validated=True`` on its own output -- a producer
    # certifying itself, which is the claim Stage 3 exists to stop trusting. Do
    # not restore the 4 by re-asserting the flag; it would have to be earned by
    # giving the mock real files.
    assert report.summary.valid_artifacts == 0
    restored = report.from_json(report.to_json())
    assert restored.metric_results == report.metric_results


@pytest.mark.asyncio
async def test_bad_policy_changes_objective_evaluation() -> None:
    good = EvaluationEngine().evaluate(SPECIFICATION, await run_policy("good"))
    bad = EvaluationEngine().evaluate(SPECIFICATION, await run_policy("bad"))

    good_success = next(item for item in good.metric_results if item.metric_id == "execution-success")
    bad_success = next(item for item in bad.metric_results if item.metric_id == "execution-success")
    assert good_success.raw_value == 1.0
    assert bad_success.raw_value == 0.0
    mixing = next(item for item in bad.metric_results if item.metric_id == "batch-mixing")
    assert mixing.status == MetricStatus.UNAVAILABLE
    assert mixing.error == "missing_artifact"


def test_metric_registry_registration_duplicate_and_missing() -> None:
    registry = MetricRegistry()

    @registry.register(
        "test-metric",
        name="Test metric",
        description="Metric used by the registry test.",
        level=EvaluationLevel.METHOD,
        direction=Direction.HIGHER_IS_BETTER,
    )
    def compute(_: MetricContext) -> MetricComputation:
        return MetricComputation(raw_value=0.5)

    assert registry.get("test-metric").compute is compute
    assert registry.list_metrics() == ["test-metric"]
    with pytest.raises(RegistryError):
        registry.get("missing")
    with pytest.raises(RegistryError):
        registry.register(
            "test-metric",
            name="Duplicate",
            description="Duplicate metric.",
            level=EvaluationLevel.METHOD,
            direction=Direction.HIGHER_IS_BETTER,
        )(compute)


@pytest.mark.asyncio
async def test_invalid_artifact_is_reported_without_crashing() -> None:
    run = await run_policy("good")
    snapshot = run.final_environment_state.model_copy(deep=True)
    snapshot.state.artifacts["corrected-embedding"].validated = False
    altered = run.model_copy(update={"final_environment_state": snapshot})
    report = EvaluationEngine().evaluate(SPECIFICATION, altered)

    artifact_metric = next(item for item in report.metric_results if item.metric_id == "artifact-validity")
    assert artifact_metric.status == MetricStatus.SUCCEEDED
    assert artifact_metric.raw_value is not None and artifact_metric.raw_value < 1.0


@pytest.mark.asyncio
async def test_metric_failure_isolated_from_other_metrics() -> None:
    payload = SPECIFICATION.model_dump(mode="json")
    failing_metric = {
        "id": "failing-metric",
        "name": "Failing metric",
        "description": "Metric intentionally raising for isolation coverage.",
        "direction": "higher_is_better",
    }
    payload["metrics"].append(failing_metric)
    payload["tasks"][0]["metrics"].append("failing-metric")
    payload["tasks"][0]["evaluation"]["metrics"].append("failing-metric")
    specification = benchmark_from_dict(payload)
    registry = MetricRegistry()

    @registry.register(
        "failing-metric",
        name="Failing metric",
        description="Raises.",
        level=EvaluationLevel.METHOD,
        direction=Direction.HIGHER_IS_BETTER,
    )
    def fail(_: MetricContext) -> MetricComputation:
        raise RuntimeError("metric exploded")

    @registry.register(
        "execution-success",
        name="Execution success",
        description="Success.",
        level=EvaluationLevel.EXECUTION,
        direction=Direction.HIGHER_IS_BETTER,
    )
    def success(context: MetricContext) -> MetricComputation:
        return execution_success(context)

    report = EvaluationEngine(registry).evaluate(specification, await run_policy("good"))
    failing = next(item for item in report.metric_results if item.metric_id == "failing-metric")
    assert failing.status == MetricStatus.ERROR
    assert failing.error == "metric exploded"
    assert any(item.status == MetricStatus.SUCCEEDED for item in report.metric_results)
