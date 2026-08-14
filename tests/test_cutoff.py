"""Controller-owned termination.

Two of these tests carry the stage. Both guard failure modes that are *silent*:
a cutoff layer that never fires looks exactly like a run that stayed inside its
budget, and a leaked progress key looks exactly like a legitimate observation
field.

**A stagnation cutoff must never fire on an unmeasured delta.** ``dS_t`` is
``None`` whenever two consecutive steps shared no comparable metric, which is
normal -- a real annotation run measures five deltas across six steps, and the
mock executor measures none at all. If ``None`` counted as "no improvement", the
controller would end runs for the harness's own blindness and the benchmark
result would be a statement about SCAIB rather than about the agent.

**The agent-visible budget must carry no reference-derived key.** Steps and
seconds remaining are facts about the harness. ``dS_t`` and the stagnation
verdict are computed from the held-out reference, so exposing them would both
leak what is being measured and let an agent defeat the detector by manufacturing
an improvement against whatever it inferred.
"""

from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from agent_evals.agents.baselines.rule_based import BASELINE_MAX_STEPS, baseline_budget
from agent_evals.agents.mock import MockActionExecutor, MockObservationBuilder
from agent_evals.agents.runtime import (
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentObservation,
    AgentRuntime,
    AgentRuntimeManager,
    AgentSession,
    FinalSubmission,
)
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.schema import (
    BenchmarkSpecification,
    ConstraintSpecification,
    CutoffSpecification,
)
from agent_evals.core.progress_keys import PROGRESS_PREFIX
from agent_evals.environment.cutoff import (
    CutoffBudget,
    CutoffController,
    CutoffEnforcement,
    CutoffReason,
    StagnationDetector,
    StagnationVerdict,
    StepObservation,
    budget_from_specification,
)
from agent_evals.environment.runtime import ScientificEnvironment

SPECIFICATION = load_benchmark(
    Path(__file__).parents[1] / "examples" / "benchmarks" / "pbmc-cell-annotation.yaml"
)

#: Every reason a report must account for, so a new one cannot be added without
#: also being classified as enforced, undeclared, or unobservable.
ALL_REASONS = set(CutoffReason)

#: The full cell-annotation workflow, ending in the terminal action. Defined here
#: rather than imported from ``tests/agents/test_runtime_protocol.py``: with no
#: ``__init__.py`` under ``tests/`` a cross-module test import resolves only after
#: the other module has been collected, so it would pass or fail on ordering.
COMPLETE_WORKFLOW: list[tuple[str, dict[str, object]]] = [
    ("qc", {"min_genes": 200, "max_mito_fraction": 0.2}),
    ("normalize", {"target_sum": 10_000}),
    ("pca", {"n_components": 10}),
    ("cluster", {"resolution": 0.5}),
    ("marker-genes", {"group_key": "predicted_clusters"}),
    (
        "annotate",
        {
            "label_vocabulary": ["T", "B"],
            "markers": {"T": ["CD3D"], "B": ["MS4A1"]},
        },
    ),
    ("finish", {}),
]


class ScriptedRuntime(AgentRuntime):
    """Replays a fixed action script, so a cutoff is the only thing that varies."""

    workflow: ClassVar[list[tuple[str, dict[str, object]]]] = COMPLETE_WORKFLOW

    def __init__(self) -> None:
        self.agent_id = "scripted-scientist"
        self.manifest = AgentManifest(name="Scripted scientist", type="test")

    async def initialize(self, context: AgentContext) -> AgentSession:
        return AgentSession(context=context)

    async def act(
        self, session: AgentSession, observation: AgentObservation
    ) -> AgentAction:
        del observation
        index = int(session.state.get("index", 0))
        session.state["index"] = index + 1
        action_type, parameters = self.workflow[index]
        return AgentAction(action_type=action_type, parameters=dict(parameters))

    async def terminate(
        self,
        session: AgentSession,
        observation: AgentObservation | None = None,
    ) -> FinalSubmission:
        del session, observation
        return FinalSubmission(summary="completed")


def make_environment(
    cutoff: CutoffSpecification | None = None,
    *,
    validated_artifacts: bool = False,
) -> ScientificEnvironment:
    """The mock-executor environment, optionally with a declared cutoff block."""
    specification = SPECIFICATION
    if cutoff is not None:
        specification = SPECIFICATION.model_copy(update={"cutoff": cutoff})
    return ScientificEnvironment(
        specification,
        task_id="cell-annotation",
        executor=MockActionExecutor(validated_artifacts=validated_artifacts),
        observation_builder=MockObservationBuilder(),
    )


def context() -> AgentContext:
    return AgentContext(
        benchmark_id="pbmc-cell-annotation",
        task_id="cell-annotation",
        workspace=".",
    )


def feed(
    controller: CutoffController,
    deltas: list[float | None],
    *,
    succeeded: bool = True,
    signature: str | None = None,
) -> None:
    """Observe one step per delta, deciding between each as the real loop does."""
    for index, delta in enumerate(deltas, start=controller.steps_used + 1):
        controller.decide(elapsed_seconds=0.0)
        controller.observe(
            StepObservation(
                step=index,
                succeeded=succeeded,
                signature=signature,
                progress_delta=delta,
            )
        )


# --------------------------------------------------------------------------- #
# StagnationDetector: unmeasured progress is not absent progress
# --------------------------------------------------------------------------- #


def test_a_window_of_unmeasured_deltas_supports_no_verdict() -> None:
    """The whole rule, at its narrowest: no measurement means no conclusion."""
    trace = StagnationDetector(window=3).evaluate([None, None, None, None, None])

    assert trace.verdict is StagnationVerdict.UNDETERMINED
    assert trace.measured_deltas == []
    assert trace.considered == []


def test_unmeasured_steps_are_skipped_rather_than_counted_as_flat() -> None:
    """Two measured zeros plus three blanks is not a full window of zeros.

    If the blanks were read as zeros this window would be stagnant, which is the
    exact substitution that would kill correct runs.
    """
    trace = StagnationDetector(window=3).evaluate([0.0, None, None, 0.0, None])

    assert trace.verdict is StagnationVerdict.UNDETERMINED
    assert trace.measured_deltas == [0.0, 0.0]


def test_a_filled_flat_window_is_stagnant_and_one_improvement_clears_it() -> None:
    detector = StagnationDetector(window=3)

    assert detector.evaluate([0.0, 0.0, 0.0]).verdict is StagnationVerdict.STAGNANT
    assert detector.evaluate([0.0, 0.5, 0.0]).verdict is StagnationVerdict.PROGRESSING


def test_only_the_most_recent_measured_deltas_are_considered() -> None:
    """An early improvement must not exempt a run from later stagnation."""
    trace = StagnationDetector(window=2).evaluate([0.9, 0.0, 0.0])

    assert trace.verdict is StagnationVerdict.STAGNANT
    assert trace.considered == [0.0, 0.0]


def test_epsilon_is_a_threshold_and_a_delta_at_it_is_not_progress() -> None:
    """Epsilon exists to absorb recomputation noise, so it must not count."""
    detector = StagnationDetector(window=2, epsilon=0.01)

    assert detector.evaluate([0.01, 0.01]).verdict is StagnationVerdict.STAGNANT
    assert detector.evaluate([0.01, 0.011]).verdict is StagnationVerdict.PROGRESSING


def test_a_regressing_run_is_stagnant_not_progressing() -> None:
    """Losing ground is not making progress, however large the movement."""
    trace = StagnationDetector(window=3).evaluate([-0.4, -0.9, -0.2])

    assert trace.verdict is StagnationVerdict.STAGNANT


# --------------------------------------------------------------------------- #
# CutoffController: the unmeasured-progress rule, end to end
# --------------------------------------------------------------------------- #


def test_a_run_whose_progress_is_never_measured_is_never_stopped() -> None:
    """The stage's load-bearing claim, at the most aggressive settings possible.

    Window 1 and patience 0 is the tightest stagnation cutoff the schema can
    express. Twenty unmeasured steps must still not stop the run, and the report
    must say the cutoff was unobservable rather than satisfied.
    """
    controller = CutoffController(
        CutoffBudget(stagnation_window=1, patience_steps=0, max_steps=None)
    )

    feed(controller, [None] * 20)

    assert not controller.decide(elapsed_seconds=0.0).stop
    report = controller.report()
    assert not report.stopped
    assert report.enforcement[CutoffReason.STAGNATION] is CutoffEnforcement.UNOBSERVABLE
    assert report.stagnation is not None
    assert report.stagnation.verdict is StagnationVerdict.UNDETERMINED


def test_an_undeclared_stagnation_window_builds_no_detector() -> None:
    """Undeclared must mean absent, not present-and-ignored somewhere later."""
    controller = CutoffController(CutoffBudget())

    feed(controller, [0.0, 0.0, 0.0, 0.0, 0.0])

    assert not controller.decide(elapsed_seconds=0.0).stop
    report = controller.report()
    assert report.stagnation is None
    assert report.enforcement[CutoffReason.STAGNATION] is CutoffEnforcement.UNDECLARED


def test_stagnation_stops_a_flat_run_only_once_patience_is_exceeded() -> None:
    controller = CutoffController(
        CutoffBudget(stagnation_window=2, patience_steps=1, max_steps=None)
    )

    feed(controller, [0.0, 0.0])
    # First stagnant verdict: inside patience.
    assert not controller.decide(elapsed_seconds=0.0).stop
    feed(controller, [0.0])
    # Second consecutive stagnant verdict: patience of 1 exceeded.
    decision = controller.decide(elapsed_seconds=0.0)

    assert decision.stop
    assert decision.reason is CutoffReason.STAGNATION
    assert decision.detail is not None
    assert "0.01" in decision.detail


def test_an_improvement_resets_patience_and_an_unmeasured_step_does_not() -> None:
    """The streak is a claim about consecutive *stagnant* verdicts.

    An unmeasured step neither extends nor resets it: the run may well be
    stagnating and the harness simply could not tell, so neither reading is
    earned. Without this, a single blank step would launder an unproductive run.
    """
    controller = CutoffController(
        CutoffBudget(stagnation_window=2, patience_steps=1, max_steps=None)
    )

    feed(controller, [0.0, 0.0, 0.0])
    report = controller.report()
    assert report.stagnation is not None
    assert report.stagnation.stagnant_streak == 2

    feed(controller, [None])
    held = controller.report()
    assert held.stagnation is not None
    assert held.stagnation.stagnant_streak == 2

    feed(controller, [0.7])
    cleared = controller.report()
    assert cleared.stagnation is not None
    assert cleared.stagnation.verdict is StagnationVerdict.PROGRESSING
    assert cleared.stagnation.stagnant_streak == 0
    assert not controller.decide(elapsed_seconds=0.0).stop


def test_blank_steps_after_a_stagnant_window_never_spend_patience() -> None:
    """Regression: an unmeasured step must not re-bill the previous verdict.

    Found by the test above. Re-evaluating on a blank returns the *same* verdict
    -- the detector reads measured deltas only, and a blank adds none -- so a
    stagnant reading repeated once per unmeasured step and patience drained on
    steps the harness could not measure. The window and patience here are set to
    the minimum that can show it: one measured flat step arms the verdict, and
    without the fix the second blank ends a run that is still working.
    """
    controller = CutoffController(
        CutoffBudget(stagnation_window=1, patience_steps=1, max_steps=None)
    )

    feed(controller, [0.0])
    assert not controller.decide(elapsed_seconds=0.0).stop

    feed(controller, [None] * 10)
    assert not controller.decide(elapsed_seconds=0.0).stop
    held = controller.report()
    assert held.stagnation is not None
    assert held.stagnation.stagnant_streak == 1

    # A second *measured* flat step is what earns the stop.
    feed(controller, [0.0])
    assert controller.decide(elapsed_seconds=0.0).stop


# --------------------------------------------------------------------------- #
# CutoffController: the leakage boundary
# --------------------------------------------------------------------------- #


def test_the_agent_visible_budget_carries_no_reference_derived_key() -> None:
    """Hard budgets only, asserted by exact key set and by substring sweep.

    The exact set is what makes this test fail when a field is added rather than
    only when a known-bad name is added -- a future ``stagnant_streak`` would slip
    past a denylist of today's spellings.
    """
    controller = CutoffController(
        CutoffBudget(
            max_steps=10,
            max_wall_time_seconds=600.0,
            max_total_tokens=50_000,
            max_cost_usd=1.0,
            stagnation_window=2,
        )
    )
    feed(controller, [0.0, 0.0, 0.9])

    budget = controller.agent_visible_budget(elapsed_seconds=12.0)

    assert set(budget) == {
        "steps_used",
        "steps_remaining",
        "seconds_remaining",
        "tokens_remaining",
    }
    rendered = repr(budget)
    assert PROGRESS_PREFIX not in rendered
    for forbidden in ("progress", "delta", "stagnat", "scientific", "reference", "cost"):
        assert forbidden not in rendered.lower(), forbidden


def test_the_visible_budget_reports_headroom_as_of_the_clock_it_is_given() -> None:
    """An agent plans against the remaining clock, so a stale reading misleads it.

    The reading also cannot run backwards: whichever of the controller's own
    elapsed time and the caller's is larger wins, so a caller passing a smaller
    number cannot hand the agent time the run no longer has.
    """
    controller = CutoffController(
        CutoffBudget(max_steps=4, max_wall_time_seconds=100.0)
    )
    controller.decide(elapsed_seconds=30.0)

    assert controller.agent_visible_budget(elapsed_seconds=30.0)["seconds_remaining"] == 70.0
    assert controller.agent_visible_budget(elapsed_seconds=80.0)["seconds_remaining"] == 20.0
    assert controller.agent_visible_budget(elapsed_seconds=1.0)["seconds_remaining"] == 70.0


def test_an_unmeasured_quantity_reports_no_headroom_rather_than_full_headroom() -> None:
    """No token report is not a full token budget."""
    controller = CutoffController(CutoffBudget(max_total_tokens=1_000))

    budget = controller.agent_visible_budget(elapsed_seconds=0.0)

    assert budget["tokens_remaining"] is None
    assert budget["seconds_remaining"] is None
    assert budget["steps_remaining"] is None


# --------------------------------------------------------------------------- #
# CutoffController: hard budgets
# --------------------------------------------------------------------------- #


def test_the_step_budget_stops_the_run_at_its_declared_limit() -> None:
    controller = CutoffController(CutoffBudget(max_steps=3))

    feed(controller, [None, None])
    assert not controller.decide(elapsed_seconds=0.0).stop
    feed(controller, [None])
    decision = controller.decide(elapsed_seconds=0.0)

    assert decision.stop
    assert decision.reason is CutoffReason.MAX_STEPS


def test_wall_time_is_measured_across_the_whole_run() -> None:
    """The defect this bounds is deliberation time, which no executor reports."""
    controller = CutoffController(CutoffBudget(max_wall_time_seconds=60.0))

    assert not controller.decide(elapsed_seconds=59.9).stop
    decision = controller.decide(elapsed_seconds=60.1)

    assert decision.stop
    assert decision.reason is CutoffReason.WALL_TIME
    assert decision.detail is not None
    assert "executor" in decision.detail


def test_elapsed_time_never_runs_backwards() -> None:
    """A clock that jumped back must not resurrect a budget that already fired."""
    controller = CutoffController(CutoffBudget(max_wall_time_seconds=60.0))

    assert controller.decide(elapsed_seconds=99.0).stop
    assert controller.decide(elapsed_seconds=1.0).stop


@pytest.mark.parametrize(
    ("budget", "observation", "expected"),
    [
        (
            CutoffBudget(max_total_tokens=1_000),
            StepObservation(step=1, succeeded=True, total_tokens=1_000),
            CutoffReason.TOKENS,
        ),
        (
            CutoffBudget(max_cost_usd=0.50),
            StepObservation(step=1, succeeded=True, cost_usd=0.75),
            CutoffReason.COST,
        ),
    ],
)
def test_reported_usage_stops_the_run_at_its_budget(
    budget: CutoffBudget, observation: StepObservation, expected: CutoffReason
) -> None:
    controller = CutoffController(budget)

    controller.observe(observation)
    decision = controller.decide(elapsed_seconds=0.0)

    assert decision.stop
    assert decision.reason is expected


def test_planning_usage_is_counted_without_inventing_a_scientific_step() -> None:
    """A plan can consume the token budget before the first action is executed."""
    controller = CutoffController(CutoffBudget(max_total_tokens=100))

    controller.observe_usage(total_tokens=100)

    decision = controller.decide(elapsed_seconds=0.0)

    assert decision.stop
    assert decision.reason is CutoffReason.TOKENS
    assert controller.report().steps_used == 0


def test_cumulative_usage_reports_do_not_resurrect_a_consumed_budget() -> None:
    """Late provider reports are monotonic even when a backend reports out of order."""
    controller = CutoffController(CutoffBudget(max_total_tokens=100))

    controller.observe_usage(total_tokens=120)
    controller.observe_usage(total_tokens=20)

    assert controller.decide(elapsed_seconds=0.0).reason is CutoffReason.TOKENS
    assert controller.report().total_tokens == 120


def test_a_budget_nothing_reports_on_is_unobservable_rather_than_satisfied() -> None:
    """A cost budget on a backend that reports no cost was never in force."""
    controller = CutoffController(
        CutoffBudget(max_cost_usd=0.01, max_total_tokens=1, max_steps=None)
    )

    feed(controller, [None, None, None])

    assert not controller.decide(elapsed_seconds=0.0).stop
    enforcement = controller.report().enforcement
    assert enforcement[CutoffReason.COST] is CutoffEnforcement.UNOBSERVABLE
    assert enforcement[CutoffReason.TOKENS] is CutoffEnforcement.UNOBSERVABLE


def test_a_consumed_budget_is_reported_before_an_inferred_one() -> None:
    """A hard limit is a fact; stagnation is an inference from what was measured.

    When both apply the fact is the better explanation to archive, so the order
    of the checks is asserted rather than left to whichever runs first.
    """
    controller = CutoffController(
        CutoffBudget(max_steps=3, stagnation_window=1, patience_steps=0)
    )

    feed(controller, [0.0, 0.0, 0.0])
    decision = controller.decide(elapsed_seconds=0.0)

    assert decision.stop
    assert decision.reason is CutoffReason.MAX_STEPS


# --------------------------------------------------------------------------- #
# CutoffController: failures and repetition
# --------------------------------------------------------------------------- #


def test_consecutive_failures_stop_a_run_that_cannot_succeed() -> None:
    """This is what an exhausted resource budget looks like from inside the loop.

    ``ConstraintMonitor`` fails the *step*, not the run, so before this cutoff a
    run whose runtime budget was gone spent every remaining step failing.
    """
    controller = CutoffController(
        CutoffBudget(max_consecutive_failures=3, max_steps=None)
    )

    feed(controller, [None, None], succeeded=False)
    assert not controller.decide(elapsed_seconds=0.0).stop
    feed(controller, [None], succeeded=False)
    decision = controller.decide(elapsed_seconds=0.0)

    assert decision.stop
    assert decision.reason is CutoffReason.CONSECUTIVE_FAILURES


def test_one_success_resets_the_failure_streak() -> None:
    """Retrying after a failure is legitimate adaptation, not a stuck run."""
    controller = CutoffController(
        CutoffBudget(max_consecutive_failures=3, max_steps=None)
    )

    feed(controller, [None, None], succeeded=False)
    feed(controller, [None], succeeded=True)
    feed(controller, [None, None], succeeded=False)

    assert not controller.decide(elapsed_seconds=0.0).stop
    assert controller.report().consecutive_failures == 2


def test_repetition_counts_identical_successful_decisions() -> None:
    controller = CutoffController(
        CutoffBudget(max_repeated_decisions=3, max_steps=None)
    )

    feed(controller, [None, None], signature="cluster({})")
    assert not controller.decide(elapsed_seconds=0.0).stop
    feed(controller, [None], signature="cluster({})")
    decision = controller.decide(elapsed_seconds=0.0)

    assert decision.stop
    assert decision.reason is CutoffReason.REPETITION
    assert decision.detail is not None
    assert "cluster({})" in decision.detail


def test_an_unidentifiable_step_is_never_counted_as_a_repeat() -> None:
    """Unidentifiable is not identical, and a failed step is not a decision made.

    ``max_consecutive_failures`` is disarmed because the second feed is five
    failures in a row: leaving it on would stop the run for the failure streak
    and the repetition assertion below would pass without ever being tested.
    """
    controller = CutoffController(
        CutoffBudget(
            max_repeated_decisions=2, max_steps=None, max_consecutive_failures=None
        )
    )

    feed(controller, [None] * 5, signature=None)
    feed(controller, [None] * 5, signature="qc({})", succeeded=False)

    assert not controller.decide(elapsed_seconds=0.0).stop
    assert controller.report().most_repeated_decisions == 0


def test_differing_parameters_are_not_a_repeated_decision() -> None:
    controller = CutoffController(
        CutoffBudget(max_repeated_decisions=2, max_steps=None)
    )

    feed(controller, [None], signature='cluster({"resolution": 0.5})')
    feed(controller, [None], signature='cluster({"resolution": 1.0})')

    assert not controller.decide(elapsed_seconds=0.0).stop


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def test_every_reason_is_classified_and_declared_limits_read_enforced() -> None:
    """A reason with no enforcement entry is a budget nobody can audit."""
    controller = CutoffController(
        CutoffBudget(
            max_steps=5,
            max_wall_time_seconds=10.0,
            max_total_tokens=100,
            max_cost_usd=1.0,
            stagnation_window=1,
            max_repeated_decisions=2,
            max_consecutive_failures=2,
        )
    )
    controller.observe(
        StepObservation(
            step=1,
            succeeded=True,
            signature="qc({})",
            progress_delta=0.4,
            total_tokens=10,
            cost_usd=0.01,
        )
    )

    enforcement = controller.report().enforcement

    assert set(enforcement) == ALL_REASONS
    assert set(enforcement.values()) == {CutoffEnforcement.ENFORCED}


def test_an_empty_budget_marks_every_reason_undeclared() -> None:
    """Distinct from a run that predates the layer, which records no report."""
    enforcement = CutoffController(CutoffBudget()).report().enforcement

    assert set(enforcement) == ALL_REASONS
    assert set(enforcement.values()) - {CutoffEnforcement.UNDECLARED} == {
        CutoffEnforcement.ENFORCED  # max_consecutive_failures defaults on
    }
    assert enforcement[CutoffReason.CONSECUTIVE_FAILURES] is CutoffEnforcement.ENFORCED


def test_the_report_keeps_the_first_breach_not_the_latest() -> None:
    """The run stopped for one reason, and later checks must not overwrite it."""
    controller = CutoffController(
        CutoffBudget(max_steps=1, max_wall_time_seconds=5.0)
    )

    feed(controller, [None])
    assert controller.decide(elapsed_seconds=0.0).reason is CutoffReason.MAX_STEPS
    controller.decide(elapsed_seconds=999.0)

    report = controller.report()
    assert report.stopped
    assert report.reason is CutoffReason.MAX_STEPS


def test_the_report_round_trips_through_json() -> None:
    """It is archived inside ``agent_run.json``, so it has to survive the trip."""
    controller = CutoffController(CutoffBudget(stagnation_window=2, max_steps=4))
    feed(controller, [0.0, 0.0])

    payload = controller.report().model_dump(mode="json")

    assert payload["enforcement"]["stagnation"] == "enforced"
    assert payload["stagnation"]["verdict"] == "stagnant"
    assert payload["budget"]["stagnation_window"] == 2


# --------------------------------------------------------------------------- #
# budget_from_specification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("declared", "requested", "expected"),
    [(10, 50, 10), (100, 5, 5), (None, 7, 7), (7, None, 7), (None, None, None)],
)
def test_step_ceilings_intersect_and_neither_can_raise_the_other(
    declared: int | None, requested: int | None, expected: int | None
) -> None:
    """Both numbers are ceilings, so the effective budget is the smaller one."""
    budget = budget_from_specification(
        CutoffSpecification(max_steps=declared), caller_max_steps=requested
    )

    assert budget.max_steps == expected


def test_wall_time_falls_back_to_the_runtime_constraint() -> None:
    """An undeclared clock is unbounded; the constraint is at least a ceiling."""
    budget = budget_from_specification(
        CutoffSpecification(),
        ConstraintSpecification(max_runtime_seconds=900),
    )

    assert budget.max_wall_time_seconds == 900.0


def test_a_declared_wall_time_is_not_overridden_by_the_constraint() -> None:
    budget = budget_from_specification(
        CutoffSpecification(max_wall_time_seconds=120.0),
        ConstraintSpecification(max_runtime_seconds=900),
    )

    assert budget.max_wall_time_seconds == 120.0


def test_the_default_specification_arms_only_the_failure_cutoff() -> None:
    """A benchmark that declares nothing must not acquire a cutoff by accident."""
    budget = budget_from_specification(CutoffSpecification())

    assert budget.stagnation_window is None
    assert budget.max_repeated_decisions is None
    assert budget.max_steps is None
    assert budget.max_consecutive_failures == 3


# --------------------------------------------------------------------------- #
# The benchmark DSL
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field", ["stagnation_epsilon", "patience_steps"]
)
def test_a_stagnation_setting_without_a_window_is_rejected(field: str) -> None:
    """Both are inert without a window, so accepting them is a silent no-op.

    Caught at load time, because the alternative is a benchmark author believing
    a stagnation cutoff is configured when nothing is armed.
    """
    with pytest.raises(ValidationError, match="stagnation_window"):
        CutoffSpecification(**{field: 0.5 if field == "stagnation_epsilon" else 5})


def test_declaring_the_window_arms_the_other_stagnation_settings() -> None:
    cutoff = CutoffSpecification(
        stagnation_window=4, stagnation_epsilon=0.05, patience_steps=1
    )

    assert cutoff.stagnation_window == 4
    assert cutoff.stagnation_epsilon == 0.05


FREE_SPECIFICATION = load_benchmark(
    Path(__file__).parents[1]
    / "examples"
    / "benchmarks"
    / "pbmc-cell-annotation-free.yaml"
)


def test_the_free_execution_example_arms_the_repeated_decision_cutoff() -> None:
    """Opt-in defaults are only defensible if something actually opts in.

    The free tier is where a repeated decision really is a loop: the signature is
    the whole script rather than an action name plus parameters.
    """
    assert FREE_SPECIFICATION.cutoff.max_repeated_decisions is not None
    assert FREE_SPECIFICATION.cutoff.max_steps is not None
    assert FREE_SPECIFICATION.cutoff.max_wall_time_seconds is not None


@pytest.mark.parametrize("specification", [SPECIFICATION, FREE_SPECIFICATION])
def test_no_example_arms_stagnation_while_the_progress_scale_is_degenerate(
    specification: BenchmarkSpecification,
) -> None:
    """A tripwire, not a rule: stagnation stays disarmed until `S_t` can move.

    Both examples resolve to the annotation metric profile, whose biology domain is
    a geometric mean containing ``cell_annotation.rare_recall``. That metric scores
    exactly 0.0 unless the agent's label spellings match the reference vocabulary,
    and one 0.0 annihilates the product -- so `S_t` is pinned at 0.0 and every
    measured `dS` is 0.0. Verified on a real run: arming a three-step window
    stopped a *correct* rule-based run at step 6 with five measured deltas of 0.0,
    while ``clustering.ari`` was 0.81 and moving.

    The detector is right and the declaration was wrong, which is why this guards
    the YAML rather than the detector. When Stage 8 fixes ``rare_recall`` this test
    fails, and whoever arms the window should read this docstring first to confirm
    the scale is no longer degenerate.
    """
    assert specification.cutoff.stagnation_window is None


# --------------------------------------------------------------------------- #
# The runtime loop
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_run_archives_its_cutoff_report() -> None:
    """A report computed every run and read by nobody would be no report."""
    result = await AgentRuntimeManager().run(
        ScriptedRuntime(), make_environment(), context(), seed=3
    )

    assert result.cutoff is not None
    assert not result.cutoff.stopped
    assert result.cutoff.steps_used == len(COMPLETE_WORKFLOW) - 1
    assert set(result.cutoff.enforcement) == ALL_REASONS


@pytest.mark.asyncio
async def test_the_observation_exposes_hard_budgets_and_no_progress_state() -> None:
    """The leakage boundary at the payload the agent actually receives."""
    result = await AgentRuntimeManager().run(
        ScriptedRuntime(),
        make_environment(CutoffSpecification(max_steps=20, stagnation_window=2)),
        context(),
        seed=3,
    )

    observations = result.trajectory.observations
    assert observations
    for observation in observations:
        budget = observation.metadata.get("budget")
        assert budget is not None
        assert set(budget) == {
            "steps_used",
            "steps_remaining",
            "seconds_remaining",
            "tokens_remaining",
        }
        assert PROGRESS_PREFIX not in observation.model_dump_json()
    assert observations[-1].metadata["budget"]["steps_remaining"] == 14


@pytest.mark.asyncio
async def test_a_mock_run_without_validated_artifacts_is_not_reported_complete() -> None:
    """The end-to-end form of the unmeasured-progress rule.

    The mock executor produces no comparable metrics, so every delta is ``None``.
    An armed stagnation cutoff must therefore record itself unobservable and let
    the run finish, rather than ending a workflow that did everything asked.
    """
    result = await AgentRuntimeManager().run(
        ScriptedRuntime(),
        make_environment(
            CutoffSpecification(stagnation_window=1, patience_steps=0, max_steps=20)
        ),
        context(),
        seed=3,
    )

    assert result.termination_status == "incomplete"
    assert result.cutoff is not None
    assert not result.cutoff.stopped
    assert (
        result.cutoff.enforcement[CutoffReason.STAGNATION]
        is CutoffEnforcement.UNOBSERVABLE
    )


@pytest.mark.asyncio
async def test_exhausting_the_step_budget_after_producing_validated_artifacts_completes() -> None:
    """Preserves what the loop's old ``while``/``else`` clause meant.

    A run that produced every required artifact and then ran out of steps
    completed; it did not time out. The budget here is the exact number of
    working steps, so the controller stops the run before the agent can say
    ``finish``.
    """
    result = await AgentRuntimeManager().run(
        ScriptedRuntime(),
        make_environment(validated_artifacts=True),
        context(),
        seed=3,
        max_steps=len(COMPLETE_WORKFLOW) - 1,
    )

    assert result.termination_status == "completed"
    assert result.cutoff is not None
    assert result.cutoff.reason is CutoffReason.MAX_STEPS


@pytest.mark.asyncio
async def test_exhausting_the_step_budget_without_the_artifacts_times_out() -> None:
    result = await AgentRuntimeManager().run(
        ScriptedRuntime(), make_environment(), context(), seed=3, max_steps=2
    )

    assert result.termination_status == "timeout"
    assert result.cutoff is not None
    assert result.cutoff.reason is CutoffReason.MAX_STEPS
    assert result.step_count == 2


class LoopingRuntime(ScriptedRuntime):
    """An agent that keeps taking the same decision with the same parameters."""

    async def act(
        self, session: AgentSession, observation: AgentObservation
    ) -> AgentAction:
        del session, observation
        return AgentAction(action_type="qc", parameters={"min_genes": 200})


@pytest.mark.asyncio
async def test_a_looping_agent_is_stopped_and_recorded_as_stagnated() -> None:
    """Repetition is not a consumed budget, so it is not a timeout.

    The run had steps left and was not using them to make progress, which is a
    different finding about the agent and gets its own verdict.
    """
    result = await AgentRuntimeManager().run(
        LoopingRuntime(),
        make_environment(CutoffSpecification(max_repeated_decisions=3, max_steps=50)),
        context(),
        seed=3,
    )

    assert result.termination_status == "stagnated"
    assert result.cutoff is not None
    assert result.cutoff.reason is CutoffReason.REPETITION
    assert result.step_count == 3


class RejectedRuntime(ScriptedRuntime):
    """An agent whose every action the environment refuses."""

    async def act(
        self, session: AgentSession, observation: AgentObservation
    ) -> AgentAction:
        del session, observation
        return AgentAction(action_type="run-arbitrary-python", parameters={})


@pytest.mark.asyncio
async def test_a_run_of_rejected_actions_stops_instead_of_spending_the_budget() -> None:
    """Before this cutoff, the loop ground out every remaining step failing."""
    result = await AgentRuntimeManager().run(
        RejectedRuntime(), make_environment(), context(), seed=3, max_steps=50
    )

    assert result.cutoff is not None
    assert result.cutoff.reason is CutoffReason.CONSECUTIVE_FAILURES
    assert result.step_count == 3
    assert result.termination_status == "timeout"


# --------------------------------------------------------------------------- #
# The rule-based baseline, which drives its own loop
# --------------------------------------------------------------------------- #


def test_the_baseline_step_fallback_applies_only_when_nothing_declares_a_ceiling() -> None:
    """Preserves the bound this baseline has always had, and no more."""
    undeclared = SPECIFICATION.model_copy(update={"cutoff": CutoffSpecification()})

    assert baseline_budget(undeclared).max_steps == BASELINE_MAX_STEPS


def test_the_baseline_step_fallback_never_lowers_a_declared_ceiling() -> None:
    """The fallback is a default, not a limit.

    Passed as ``caller_max_steps`` it would be the *stricter* of the two and would
    silently cap a benchmark asking for 40 steps at 8 -- a benchmark quietly
    scored on a horizon it did not choose. The free example declares more than the
    fallback precisely so this is testable against a real declaration.
    """
    assert FREE_SPECIFICATION.cutoff.max_steps is not None
    assert FREE_SPECIFICATION.cutoff.max_steps > BASELINE_MAX_STEPS
    assert baseline_budget(FREE_SPECIFICATION).max_steps == (
        FREE_SPECIFICATION.cutoff.max_steps
    )


def test_a_stricter_caller_limit_still_wins_over_the_declared_ceiling() -> None:
    """``--max-steps`` is how an operator shortens a run, and must keep working."""
    budget = baseline_budget(FREE_SPECIFICATION, caller_max_steps=3)

    assert budget.max_steps == 3


def test_the_baseline_inherits_every_declared_cutoff_not_just_the_step_ceiling() -> None:
    """The point of the wiring: one declaration governs both loops identically."""
    budget = baseline_budget(FREE_SPECIFICATION)

    assert budget.max_repeated_decisions == (
        FREE_SPECIFICATION.cutoff.max_repeated_decisions
    )
    assert budget.max_consecutive_failures == (
        FREE_SPECIFICATION.cutoff.max_consecutive_failures
    )
    assert budget.max_wall_time_seconds == FREE_SPECIFICATION.cutoff.max_wall_time_seconds


@pytest.mark.asyncio
async def test_the_rule_based_baseline_archives_its_cutoff_report(tmp_path: Path) -> None:
    """The regression for a whole layer that was inert on the path that is run.

    This baseline drives its own episode loop, so until this stage the declared
    ``cutoff`` block governed every agent *except* the reference agent the paper
    reports against -- and nothing failed, because a budget nothing consults looks
    exactly like a budget nothing exceeded. The archived report is the only
    artefact that can tell those two apart, which is why the assertion is on the
    stored run rather than on the controller.
    """
    pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    from agent_evals.environment.scientific_loop import ScientificLoop

    run = await ScientificLoop().run(
        "pbmc-cell-annotation", output_dir=tmp_path, max_cells=120
    )

    cutoff = run.agent_run.metadata.get("cutoff")
    assert cutoff is not None
    # The benchmark's declared ceiling, not the fallback.
    assert cutoff["budget"]["max_steps"] == SPECIFICATION.cutoff.max_steps
    assert cutoff["steps_used"] >= 1
    assert not cutoff["stopped"]
    assert (
        cutoff["enforcement"][CutoffReason.STAGNATION.value]
        == CutoffEnforcement.UNDECLARED.value
    )


@pytest.mark.asyncio
async def test_cutting_off_the_baseline_early_is_not_recorded_as_completed(
    tmp_path: Path,
) -> None:
    """The baseline's cutoff branch, which nothing else reaches.

    Without this the branch is present and never executed, which is the shape of
    every defect this stage found: the code is right, and no run proves it. Two
    steps produce two of the six required artifacts, so the run is stopped short of
    the benchmark goal and must say so.
    """
    pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    from agent_evals.environment.scientific_loop import ScientificLoop

    run = await ScientificLoop().run(
        "pbmc-cell-annotation", output_dir=tmp_path, max_cells=120, max_steps=2
    )

    assert not run.agent_run.succeeded
    assert run.agent_run.termination_status == "timeout"
    cutoff = run.agent_run.metadata["cutoff"]
    assert cutoff["stopped"]
    assert cutoff["reason"] == CutoffReason.MAX_STEPS.value
    # The caller's limit, not the benchmark's 20: an operator shortening a run
    # must actually shorten it.
    assert cutoff["budget"]["max_steps"] == 2
