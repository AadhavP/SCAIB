"""Which benchmarks the reference baseline can actually execute.

Per-benchmark metric profiles made all three workflow families *scoreable*
(``tests/test_metric_profiles.py``). That is a different question from whether
anything can be made to *run* them, and the answer turns out to be no for one of
the three: the rule-based baseline shares exactly one action id with the
differential-expression benchmark, and even that one it parameterises for a
different declaration. So the DE run the Stage 8 probe drove terminated
``failed`` at step 0 with ``invalid_action`` and took zero actions.

That is worth its own module because the failure is invisible from either side
alone. The benchmark YAML is internally valid and loads cleanly; the baseline is
correct for the catalog it was written against. Only the pairing is broken, and
nothing pairs them until someone runs the two together and reads the result.
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
EXPECTED_UNSUPPORTED: dict[str, tuple[str, ...]] = {
    "pbmc-cell-annotation": (),
    "pbmc-batch-correction": ("neighborhood-graph",),
    "pbmc-differential-expression": ("differential-expression", "report"),
    "pbmc-cell-annotation-free": ("analyze",),
}


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
    exception is ``marker-genes``, whose branch additionally requires
    ``clustered``; and setting that flag globally would suppress the ``cluster``
    branch and report the baseline's own clustering rule as missing.
    """
    return {"clustered": True} if action_id == "marker-genes" else {}


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


def test_the_de_benchmark_cannot_be_driven_by_the_baseline_at_all() -> None:
    """The headline: no proposable action of the DE task is also submittable.

    Stated as a property of the whole task rather than of ``normalize`` alone,
    because fixing the parameter mismatch below without adding the two missing
    rules would satisfy a narrower test while leaving the benchmark just as
    unrunnable -- the baseline would normalize and then have nothing to do.
    """
    specification = _specifications()["pbmc-differential-expression"]
    task = specification.tasks[0]

    submittable = []
    for action_id in task.allowed_actions:
        proposal = _proposal(specification, action_id)
        if proposal is None:
            continue
        emitted_id, parameters = proposal
        if not _parameter_errors(specification, emitted_id, parameters):
            submittable.append(emitted_id)

    assert submittable == []


def test_the_baselines_normalize_parameters_are_written_for_the_annotation_catalog() -> None:
    """Both halves of the mismatch, and the contrast that explains it.

    ``normalize`` is the one action id the annotation and DE benchmarks share, and
    they declare it differently: annotation takes ``target_sum``, DE takes a
    required ``method`` enum and nothing else. The baseline emits ``target_sum``,
    which is why the DE probe died before its first step.
    """
    annotation = _specifications()["pbmc-cell-annotation"]
    de = _specifications()["pbmc-differential-expression"]
    proposal = _proposal(de, "normalize")
    assert proposal is not None
    emitted_id, parameters = proposal
    assert emitted_id == "normalize"

    # Accepted by the catalog it was written for. Without this half, deleting the
    # parameters entirely would "fix" the assertion below.
    assert _parameter_errors(annotation, "normalize", parameters) == []

    errors = _parameter_errors(de, "normalize", parameters)
    assert any("unknown parameter 'target_sum'" in error for error in errors)
    assert any("missing required parameter 'method'" in error for error in errors)


def test_the_integration_benchmark_is_driveable_far_enough_to_produce_artifacts() -> None:
    """The contrast that keeps the DE finding specific to DE.

    Batch correction also has an uncovered action, but its three covered ones are
    the ones that produce artifacts, so that benchmark runs and scores. Recorded
    so the DE gap is not generalized into "the baseline only does annotation".
    """
    specification = _specifications()["pbmc-batch-correction"]

    submittable = []
    for action_id in specification.tasks[0].allowed_actions:
        proposal = _proposal(specification, action_id)
        if proposal is None:
            continue
        emitted_id, parameters = proposal
        if not _parameter_errors(specification, emitted_id, parameters):
            submittable.append(emitted_id)

    assert submittable == ["normalize", "pca", "harmony"]
