"""Which benchmarks the reference baseline can actually execute.

Per-benchmark metric profiles made all three workflow families *scoreable*
(``tests/test_metric_profiles.py``). That is a different question from whether
anything can be made to *run* them, and when this module was written the answer
was no for one of the three: the rule-based baseline shared exactly one action id
with the differential-expression benchmark, and even that one it parameterised for
a different declaration. So the DE run the Stage 8 probe drove terminated
``failed`` at step 0 with ``invalid_action`` and took zero actions.

That was worth its own module because the failure is invisible from either side
alone. The benchmark YAML is internally valid and loads cleanly; the baseline is
correct for the catalog it was written against. Only the *pairing* is broken, and
nothing pairs them until someone runs the two together and reads the result.

The pairing is now fixed -- the DE benchmark gained a ``cluster`` action and a
required ``group_key``, the baseline gained ``differential-expression`` and
``report`` rules, and ``normalize`` declares the same method enum on all three
catalogs that offer it -- so the assertions below record the repaired state. The
module stays, because the class of defect has not gone anywhere: the fix's own
first attempt updated two of the three ``normalize`` declarations and left the
batch-correction benchmark rejecting the baseline's proposal at step 0, which is
exactly the original finding in miniature. That is what
``test_one_normalization_proposal_validates_against_every_catalog`` is for.
"""

from pathlib import Path

import pytest

from agent_evals.agents.baselines.rule_based import RuleBasedSingleCellAgent
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.schema import ActionSpecification, BenchmarkSpecification
from agent_evals.environment.models import ActionIntent
from agent_evals.environment.ports import DeclarativeActionValidator
from agent_evals.scientific.observations.observations import ScientificObservation

_BENCHMARKS = Path(__file__).parents[1] / "examples" / "benchmarks"

#: Benchmark id -> the action ids its first task declares that the baseline has no
#: rule for. Hand-written rather than computed, so a baseline that gains a rule
#: fails this and has to say so, instead of the table quietly agreeing with it.
#:
#: ``pbmc-cell-annotation-free`` is *not* a gap: its single ``analyze`` action is
#: the free-execution tier, which the baseline is not the agent for.
#: ``neighborhood-graph`` is a real gap and the one remaining one.
EXPECTED_UNSUPPORTED: dict[str, tuple[str, ...]] = {
    "pbmc-cell-annotation": (),
    "pbmc-batch-correction": ("neighborhood-graph",),
    "pbmc-differential-expression": (),
    "pbmc-cell-annotation-free": ("analyze",),
}

#: The action id every typed catalog declares, and the one the three-way drift
#: happened on. Named rather than discovered so that a catalog which stops
#: declaring it has to be noticed.
_SHARED_ACTION = "normalize"

#: Branches of ``choose`` that additionally require an earlier step to have
#: happened. Both test the recovered groups they are about to compare, so neither
#: can be probed from a blank state.
_REQUIRES_CLUSTERS = frozenset({"marker-genes", "differential-expression"})


def _specifications() -> dict[str, BenchmarkSpecification]:
    return {
        specification.metadata.id: specification
        for specification in (
            load_benchmark(path) for path in sorted(_BENCHMARKS.glob("*.yaml"))
        )
    }


def _probe_state(action_id: str) -> dict[str, object]:
    """The pipeline state in which the probed action would be the next step.

    ``RuleBasedSingleCellAgent.choose`` gates each branch on the step *not* having
    happened yet, so a blank state is the right probe for almost everything. The
    exceptions are the two differential-expression branches, which additionally
    require ``clustered``; and setting that flag globally would suppress the
    ``cluster`` branch and report the baseline's own clustering rule as missing.
    """
    return {"clustered": True} if action_id in _REQUIRES_CLUSTERS else {}


def _proposal(
    specification: BenchmarkSpecification, action_id: str
) -> tuple[str, dict[str, object]] | None:
    """Ask the real baseline what it would submit for one declared action.

    Offering a single available action makes the agent's if/elif chain either
    select it or fall through to ``None``, which is what "has a rule for this
    action" means without simulating a whole episode.
    """
    decision = RuleBasedSingleCellAgent().choose(
        ScientificObservation(
            available_actions=[action_id],
            pipeline_state=_probe_state(action_id),
        ),
        specification.tasks[0],
        episode_id="coverage-probe",
        order=0,
    )
    if decision is None:
        return None
    return str(decision.metadata["rule_action"]), dict(decision.parameters)


def _declaration(specification: BenchmarkSpecification, action_id: str) -> ActionSpecification:
    action = next(item for item in specification.actions if item.id == action_id)
    return action


def _parameter_errors(
    specification: BenchmarkSpecification,
    action_id: str,
    parameters: dict[str, object],
) -> list[str]:
    """Judge the proposal with the production rule, never a restatement of it.

    ``_validate_parameters`` is reached directly rather than through ``validate``
    because the public entry point also checks required *inputs* against an
    episode snapshot, and an empty snapshot would add missing-input errors that
    have nothing to do with the mismatch under test. Re-deriving the parameter
    rule here instead would produce a test that agrees with its own reading of the
    contract rather than with the one the runtime enforces.
    """
    return DeclarativeActionValidator._validate_parameters(
        _declaration(specification, action_id),
        ActionIntent(action_id=action_id, parameters=parameters),
    )


def _submittable(specification: BenchmarkSpecification) -> list[str]:
    """The action ids the baseline both proposes and gets past the validator."""
    accepted = []
    for action_id in specification.tasks[0].allowed_actions:
        proposal = _proposal(specification, action_id)
        if proposal is None:
            continue
        emitted_id, parameters = proposal
        if not _parameter_errors(specification, emitted_id, parameters):
            accepted.append(emitted_id)
    return accepted


@pytest.mark.parametrize("benchmark_id", sorted(EXPECTED_UNSUPPORTED))
def test_the_baselines_action_coverage_per_benchmark_is_what_is_recorded(
    benchmark_id: str,
) -> None:
    """The gap, measured against the real mapper rather than assumed."""
    specification = _specifications()[benchmark_id]
    task = specification.tasks[0]

    unsupported = tuple(
        action_id
        for action_id in task.allowed_actions
        if _proposal(specification, action_id) is None
    )

    assert unsupported == EXPECTED_UNSUPPORTED[benchmark_id]


def test_the_de_benchmark_is_driveable_end_to_end_by_the_baseline() -> None:
    """The headline, inverted: every action of the DE task is now submittable.

    Stated as a property of the whole task rather than of ``normalize`` alone,
    because fixing the parameter mismatch without adding the missing rules would
    satisfy a narrower test while leaving the benchmark just as unrunnable -- the
    baseline would normalize and then have nothing to do. The order matters too:
    it is the pipeline order the artifact contract requires, since
    ``differential-expression`` consumes ``cluster``'s table and ``report``
    consumes the DE table.
    """
    specification = _specifications()["pbmc-differential-expression"]

    assert _submittable(specification) == [
        "normalize",
        "cluster",
        "differential-expression",
        "report",
    ]


def test_one_normalization_proposal_validates_against_every_catalog() -> None:
    """The generalization of the original finding, and of the fix's own near-miss.

    The baseline emits one normalization parameter set for every benchmark, so
    every catalog declaring the action has to accept it. Asserted over the
    catalogs *discovered* to declare ``normalize`` rather than over a hand-listed
    pair, because the first attempt at this fix updated two of the three and left
    the third rejecting the proposal at step 0 -- and a test naming only the two it
    had updated would have agreed.
    """
    specifications = _specifications()
    declaring = {
        benchmark_id: specification
        for benchmark_id, specification in specifications.items()
        if any(action.id == _SHARED_ACTION for action in specification.actions)
    }
    # Three of the four examples declare it; the free-execution benchmark declares
    # a single ``analyze`` action instead. Pinned so a new typed benchmark cannot
    # join without this test looking at it.
    assert sorted(declaring) == [
        "pbmc-batch-correction",
        "pbmc-cell-annotation",
        "pbmc-differential-expression",
    ]

    proposals = {}
    for benchmark_id, specification in declaring.items():
        proposal = _proposal(specification, _SHARED_ACTION)
        assert proposal is not None, benchmark_id
        proposals[benchmark_id] = proposal

    # One proposal, not three: the baseline reads no catalog when it chooses, so a
    # per-benchmark parameter set is not something it could produce.
    parameters = proposals["pbmc-cell-annotation"][1]
    assert list(proposals.values()) == [
        (_SHARED_ACTION, parameters) for _ in range(len(declaring))
    ]

    errors = {
        benchmark_id: _parameter_errors(specification, _SHARED_ACTION, dict(parameters))
        for benchmark_id, specification in declaring.items()
    }
    assert errors == {benchmark_id: [] for benchmark_id in declaring}


def test_a_stage_whose_input_is_missing_stops_the_policy_rather_than_reordering_it() -> None:
    """The one property the rule table can silently lose, and what it costs.

    ``choose`` scans an ordered table and each entry may declare flags that must
    already be set. An entry that is offered and unfinished but *not ready* returns
    nothing, and specifically does not go on to consider later entries -- which the
    original if/elif chain got for free from ``elif`` and a loop does not: the
    obvious body is ``continue``.

    The cost of ``continue`` is not a missing step, it is a reordered analysis. With
    no clustering yet, ``marker-genes`` cannot rank anything, so skipping it reaches
    ``annotate`` and the policy labels populations it has not recovered. That is a
    worse trajectory than stopping, and it would be recorded as a decision the agent
    chose rather than as a run that ran out of things it could legitimately do.
    """
    task = _specifications()["pbmc-cell-annotation"].tasks[0]
    agent = RuleBasedSingleCellAgent()
    offered = ["marker-genes", "annotate"]
    upstream = {"qc_complete": True, "normalized": True, "pca_complete": True}

    def choose(clustered: bool) -> str | None:
        decision = agent.choose(
            ScientificObservation(
                available_actions=offered,
                pipeline_state={**upstream, "clustered": clustered},
            ),
            task,
            episode_id="ordering-probe",
            order=0,
        )
        return None if decision is None else str(decision.metadata["rule_action"])

    # The presence half: the fixture can produce a decision, so the ``None`` below
    # is the guard firing and not an observation nothing in the table matches.
    assert choose(clustered=True) == "marker-genes"
    assert choose(clustered=False) is None


def test_reporting_is_the_last_stage_the_policy_will_take() -> None:
    """The rule table's order is documentation, so the unreachable part still binds.

    No example catalog offers ``report`` and ``annotate`` together, so nothing in a
    run can currently observe which comes first -- and the first draft of the table
    put ``report`` above ``annotate``, which would summarize clusters the policy was
    about to label. Found by diffing the refactored policy against its predecessor
    over every combination of offered actions, not by a run.

    Asserted in both directions so this cannot pass by ``report`` simply never being
    selectable: with annotation outstanding the policy annotates, and once it is done
    the same observation yields the report.
    """
    task = _specifications()["pbmc-cell-annotation"].tasks[0]
    agent = RuleBasedSingleCellAgent()
    finished = {
        "qc_complete": True,
        "normalized": True,
        "pca_complete": True,
        "clustered": True,
        "differential_expression_complete": True,
    }

    def choose(annotated: bool) -> str | None:
        decision = agent.choose(
            ScientificObservation(
                available_actions=["report", "annotate"],
                pipeline_state={**finished, "annotated": annotated},
            ),
            task,
            episode_id="terminal-stage-probe",
            order=0,
        )
        return None if decision is None else str(decision.metadata["rule_action"])

    assert choose(annotated=False) == "annotate"
    assert choose(annotated=True) == "report"


def test_the_integration_benchmark_is_driveable_far_enough_to_produce_artifacts() -> None:
    """The contrast that keeps the remaining gap specific to one action.

    Batch correction still has an uncovered action, but its three covered ones are
    the ones that produce artifacts, so that benchmark runs and scores. Recorded so
    ``neighborhood-graph`` is not generalized into "the baseline cannot integrate".
    """
    specification = _specifications()["pbmc-batch-correction"]

    assert _submittable(specification) == ["normalize", "pca", "harmony"]
