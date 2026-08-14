"""Tests for versioned scientific metrics and deterministic agent evaluation."""

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
from agent_evals.evaluation import (
    DecisionEvaluator,
    MethodEvaluator,
    TrajectoryEvaluator,
)
from agent_evals.metrics import (
    ApplicabilityContext,
    MetricApplicability,
    MetricApplicabilityEngine,
    MetricCategory,
    MetricDefinition,
    MetricDirection,
    MetricGroup,
    MetricResult,
    MetricRole,
    MetricStatus,
    MetricWeight,
    NormalizationEngine,
    NormalizationSpec,
    aggregate_group,
)


def _definition(metric_id: str, applicability: MetricApplicability | None = None) -> MetricDefinition:
    return MetricDefinition(
        metric_id=metric_id,
        name=metric_id,
        version="1.0",
        description=metric_id,
        category=MetricCategory.ANNOTATION,
        role=MetricRole.PRIMARY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        native_min=0,
        native_max=1,
        applicability=applicability or MetricApplicability(),
        computation_backend="test",
    )


def test_applicability_distinguishes_structure_from_candidate_failure() -> None:
    engine = MetricApplicabilityEngine()
    structural = engine.evaluate(
        _definition("structural", MetricApplicability(structural_metadata=["labels"])),
        ApplicabilityContext(),
    )
    candidate = engine.evaluate(
        _definition("candidate", MetricApplicability(required_artifacts=["prediction"])),
        ApplicabilityContext(),
    )

    assert structural.structurally_ineligible is True
    assert structural.eligible is False
    assert candidate.structurally_ineligible is False
    assert candidate.eligible is True
    assert candidate.missing_candidate_artifacts == ["prediction"]


def test_normalization_preserves_direction_and_anchors() -> None:
    engine = NormalizationEngine()
    lower = _definition("lower").model_copy(
        update={
            "direction": MetricDirection.LOWER_IS_BETTER,
            "normalization": NormalizationSpec(policy="bounded"),
        }
    )
    anchored = _definition("anchor").model_copy(
        update={
            "direction": MetricDirection.LOWER_IS_BETTER,
            "normalization": NormalizationSpec(policy="anchor", bad_anchor=1, target_anchor=0),
        }
    )

    assert engine.normalize(0.0, lower) == 1.0
    assert engine.normalize(1.0, lower) == 0.0
    assert engine.normalize(0.0, anchored) == 1.0
    assert engine.normalize(1.0, anchored) == 0.0


def test_aggregation_does_not_renormalize_candidate_failures() -> None:
    group = MetricGroup(
        group_id="quality",
        metrics=[
            MetricWeight(metric_id="good", weight=1),
            MetricWeight(metric_id="failed", weight=1),
            MetricWeight(metric_id="structural", weight=1),
        ],
        minimum_required=1,
    )
    result = aggregate_group(
        group,
        [
            MetricResult(
                metric_id="good", version="1.0", metric_name="good", role=MetricRole.PRIMARY,
                direction=MetricDirection.HIGHER_IS_BETTER, raw_value=0.8, normalized_value=0.8,
                eligible=True, status=MetricStatus.COMPUTED, eligibility_reason="eligible",
            ),
            MetricResult(
                metric_id="failed", version="1.0", metric_name="failed", role=MetricRole.PRIMARY,
                direction=MetricDirection.HIGHER_IS_BETTER, normalized_value=0.0,
                eligible=True, status=MetricStatus.FAILED, eligibility_reason="candidate failed",
            ),
            MetricResult(
                metric_id="structural", version="1.0", metric_name="structural", role=MetricRole.PRIMARY,
                direction=MetricDirection.HIGHER_IS_BETTER, eligible=False,
                status=MetricStatus.STRUCTURALLY_INELIGIBLE, eligibility_reason="not applicable",
            ),
        ],
    )

    assert result.value == pytest.approx(0.4)
    assert result.excluded_metric_ids == ["structural"]
    assert result.missing_required_count == 0


def test_aggregation_excludes_unimplemented_metrics_and_reports_missing_evidence() -> None:
    group = MetricGroup(
        group_id="quality",
        metrics=[
            MetricWeight(metric_id="available", weight=1),
            MetricWeight(metric_id="unimplemented", weight=1),
        ],
        minimum_required=2,
    )

    result = aggregate_group(
        group,
        [
            MetricResult(
                metric_id="available", version="1.0", metric_name="available", role=MetricRole.PRIMARY,
                direction=MetricDirection.HIGHER_IS_BETTER, normalized_value=0.8,
                eligible=True, status=MetricStatus.SCORED, eligibility_reason="eligible",
            ),
            MetricResult(
                metric_id="unimplemented", version="1.0", metric_name="unimplemented", role=MetricRole.PRIMARY,
                direction=MetricDirection.HIGHER_IS_BETTER, eligible=False,
                status=MetricStatus.UNIMPLEMENTED, eligibility_reason="backend unavailable",
            ),
        ],
    )

    assert result.value is None
    assert result.excluded_metric_ids == ["unimplemented"]
    assert result.missing_required_count == 1


@pytest.mark.asyncio
async def test_decision_method_and_trajectory_scores_are_deterministic() -> None:
    specification = load_benchmark(Path(__file__).parents[1] / "examples" / "benchmarks" / "pbmc-cell-annotation.yaml")
    environment = ScientificEnvironment(
        specification,
        task_id="cell-annotation",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )
    run = await AgentHarness().run(
        MockAgentAdapter(),
        environment,
        AgentConfiguration(agent_type="mock", seed=42),
    )
    task = specification.tasks[0]
    decisions = DecisionEvaluator().evaluate(run, task)
    methods = MethodEvaluator().evaluate(run, task, ["cell_annotation.macro_f1"], 0.5)
    first = TrajectoryEvaluator().evaluate(run, task, 0.5)
    second = TrajectoryEvaluator().evaluate(run, task, 0.5)

    assert decisions and methods
    assert first == second
    assert 0 <= first.trajectory_quality <= 1
