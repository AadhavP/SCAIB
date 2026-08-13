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
from agent_evals.agents.trajectory import AgentRun
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.evaluation.global_score import compute_global_agent_score
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


def _without_actions(run: AgentRun) -> AgentRun:
    """The same run with no action recorded, built without mutating the original.

    Only ``state.actions`` is emptied. The events are deliberately left in place,
    because that is the shape a real unrunnable benchmark produces: the episode
    resets, publishes its observations, and the agent's first proposal is refused
    before the environment ever records an action. Clearing the events too would
    make ``adaptation_ability`` fall to ``0.0`` and hide half the defect below --
    the term divides observation events by ``max(1, action_count)``, so the events
    from a bare reset are what pay it in full.
    """
    state = run.final_environment_state.state.model_copy(update={"actions": []})
    snapshot = run.final_environment_state.model_copy(update={"state": state})
    return run.model_copy(update={"final_environment_state": snapshot})


@pytest.mark.asyncio
async def test_a_run_that_did_nothing_still_collects_most_of_the_trajectory_score() -> None:
    """KNOWN DEFECT, pinned deliberately. This is the D = 1.0 hole on the T axis.

    Stage 2 fixed the decision dimension: a run that recorded nothing reports an
    *unmeasured* ``D`` rather than a perfect one. The trajectory dimension was
    never audited for the same defect, and it has it. Every term below defaults to
    its maximum when its input is empty: ``artifact_validity`` is ``1.0`` when
    there are no artifacts, ``dependency_consistency`` when no action declared an
    input, ``efficiency`` and ``adaptation`` both divide by ``max(1,
    action_count)``, and ``protocol`` divides rejections by ``max(1, attempted)``
    so a run that never got as far as a submission reads as fully compliant. The
    result is that a run which took **zero actions** scores 0.889.
    ``_weighted_quality`` cannot see this: it distinguishes ``None`` from a number,
    and every one of these hands it a number.

    ``adaptation_ability`` is the sharpest of the five, because it is paid for the
    harness's *own* output -- the two ``observations.updated`` events a bare reset
    publishes, divided by an action count of zero floored to one.

    Found by the Stage 8 profile-resolution probe, where the rule-based baseline
    could not execute the DE benchmark at all and produced exactly this run shape.

    **Not fixed here on purpose.** The fix is to report each term as ``None`` when
    it had no input and let the existing renormalization drop it, which widens five
    ``TrajectoryEvaluation`` floats to ``float | None``, forces
    ``trajectory_quality`` itself to be nullable for a wholly unobserved
    trajectory, and re-baselines T for every published run. The plan requires a
    re-baseline to be its own explicit commit rather than folded into an unrelated
    one. When that commit lands, this test fails and must be replaced by its
    inverse: the same run reporting an unmeasured trajectory.

    The second half asserts the *containment* that makes deferring defensible --
    no published score can currently be inflated by this, because a zero-action
    run has no decisions either and an unmeasured D already voids the global score.
    If that ever stops holding, this stops being a reporting defect and becomes a
    gameable one, and this assertion is what says so.
    """
    specification = load_benchmark(
        Path(__file__).parents[1] / "examples" / "benchmarks" / "pbmc-cell-annotation.yaml"
    )
    environment = ScientificEnvironment(
        specification,
        task_id="cell-annotation",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )
    run = await AgentHarness().run(
        MockAgentAdapter(), environment, AgentConfiguration(agent_type="mock", seed=1)
    )

    evaluation = TrajectoryEvaluator().evaluate(_without_actions(run), specification.tasks[0])

    assert evaluation.step_count == 0
    # Five terms paid in full for having nothing to look at. Named individually so
    # a partial fix cannot leave one of them behind unnoticed.
    assert evaluation.protocol_compliance == 1.0
    assert evaluation.artifact_validity == 1.0
    assert evaluation.dependency_consistency == 1.0
    assert evaluation.efficiency == 1.0
    assert evaluation.adaptation_ability == 1.0
    # The only term that does not, and the sole reason the total is not ~0.98.
    assert evaluation.method_exploration_score == 0.0
    # The one term that *is* honest about having no input, and the contrast that
    # shows the fix is a matter of extending an existing mechanism rather than
    # inventing one: five `None`s here would renormalize the same way this does.
    assert evaluation.scientific_progress is None
    assert evaluation.unmeasured_weight == pytest.approx(0.1)
    assert evaluation.trajectory_quality == pytest.approx(0.8 / 0.9)

    # The containment: unmeasured D voids the product before T can contribute.
    assert (
        compute_global_agent_score(None, None, evaluation.trajectory_quality) is None
    )
