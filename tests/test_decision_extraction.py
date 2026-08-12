"""Tests for canonical decision extraction and the free-execution action kind.

Three claims are on trial here, and each is a place the benchmark could look
like it works while scoring the wrong thing:

1. Whatever an agent says about its reasoning becomes typed decision metadata,
   or is recorded as unreadable -- never silently mangled and never a crash.
2. A free-execution turn produces exactly one action record, so decisions and
   trajectories come out of agent-authored work as they do out of typed actions.
3. A run with no decisions is scored as *unmeasured*, not as perfect. This is the
   regression test for the hole that paid an agent better for skipping structure.
"""

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from agent_evals.agents.decisions import (
    CANONICAL_DECISION_KEYS,
    DECISION_FINDINGS_KEY,
    DECISION_QUALITY_KEY,
    DECISION_TEXT_KEY,
    DecisionQuality,
    extract_decision,
)
from agent_evals.agents.runtime.manager import _action_to_intent
from agent_evals.agents.runtime.protocol import AgentAction
from agent_evals.agents.trajectory import (
    AgentConfiguration,
    AgentRun,
    NormalizedTrajectory,
    RunTerminationStatus,
    decision_cascade_from_episode,
)
from agent_evals.benchmarks.io import benchmark_from_dict, load_benchmark
from agent_evals.benchmarks.schema import (
    ActionKind,
    ActionSpecification,
    EnvironmentBackend,
    EnvironmentSpecification,
)
from agent_evals.core.intent_parameters import EXECUTION_PARAMETERS
from agent_evals.environment import (
    ActionExecutionResult,
    ActionIntent,
    ActionStatus,
    ArtifactRecord,
    Observation,
    ResourceUsage,
    RewardRecord,
    ScientificEnvironment,
)
from agent_evals.environment.execution.router import (
    ActionKindRouter,
    free_execution_action_ids,
)
from agent_evals.environment.models import utc_now
from agent_evals.environment.ports import ExecutionContext
from agent_evals.environment.scientific_loop import _score_formula, _unmeasured_or
from agent_evals.evaluation.decisions import DecisionEvaluator
from agent_evals.evaluation.global_score import (
    ScoreWeights,
    compute_global_agent_score,
)

EXAMPLES = Path(__file__).parents[1] / "examples" / "benchmarks"
FREE_SPECIFICATION = load_benchmark(EXAMPLES / "pbmc-cell-annotation-free.yaml")
TYPED_SPECIFICATION = load_benchmark(EXAMPLES / "pbmc-cell-annotation.yaml")

CODE = "import anndata\nprint('ok')\n"


# --------------------------------------------------------------------------- #
# Decision extraction
# --------------------------------------------------------------------------- #


def test_absent_reasoning_is_absent_not_malformed() -> None:
    """Saying nothing is a different finding from saying something unreadable."""
    result = extract_decision({})

    assert result.quality is DecisionQuality.ABSENT
    assert result.findings == []
    assert result.observable is False


def test_fully_structured_reasoning_is_coerced_and_typed() -> None:
    """A response in the protocol survives with every field usable downstream."""
    result = extract_decision(
        {
            "method": "leiden",
            "method_id": "leiden",
            "decision_category": "clustering",
            "hypothesis": "resolution 1.0 separates the monocyte compartment",
            "evidence_used": ["elbow plot", "marker heatmap"],
            "alternatives_considered": ["louvain", "kmeans"],
            "confidence": 0.7,
            "expected_effect": {"cluster_count": 12, "ari": 0.5},
        }
    )

    assert result.quality is DecisionQuality.STRUCTURED
    assert result.findings == []
    assert result.observable is True
    assert result.metadata["evidence_used"] == ["elbow plot", "marker heatmap"]
    assert result.metadata["confidence"] == pytest.approx(0.7)
    assert result.metadata["expected_effect"] == {"cluster_count": 12.0, "ari": 0.5}
    assert result.metadata[DECISION_QUALITY_KEY] == DecisionQuality.STRUCTURED.value


def test_evidence_given_as_prose_is_one_item_not_one_per_character() -> None:
    """Iterating a bare string would invent evidence the agent never offered."""
    result = extract_decision({"evidence_used": "the elbow plot"})

    assert result.metadata["evidence_used"] == ["the elbow plot"]
    assert result.quality is DecisionQuality.STRUCTURED
    assert any("single item" in finding for finding in result.findings)


def test_prose_where_a_decision_object_belongs_is_malformed_not_dropped() -> None:
    """An agent that writes a paragraph is recorded, not silently discarded."""
    result = extract_decision({"decision": "I will cluster the cells with leiden"})

    assert result.quality is DecisionQuality.MALFORMED
    assert result.metadata[DECISION_TEXT_KEY] == "I will cluster the cells with leiden"
    assert any("prose" in finding for finding in result.findings)


def test_nested_decision_block_is_merged_and_top_level_wins() -> None:
    """The protocol's nested block populates the same canonical vocabulary."""
    result = extract_decision(
        {
            "decision": {"method": "louvain", "hypothesis": "graph communities align"},
            "method": "leiden",
        }
    )

    assert result.quality is DecisionQuality.STRUCTURED
    assert result.metadata["method"] == "leiden"
    assert result.metadata["hypothesis"] == "graph communities align"
    assert any("top-level value used" in finding for finding in result.findings)


def test_unreadable_effect_map_is_recorded_rather_than_raised() -> None:
    """``expected_effect`` as prose used to reach ``.items()`` and raise."""
    result = extract_decision({"expected_effect": "a large improvement"})

    assert result.quality is DecisionQuality.MALFORMED
    assert "expected_effect" not in result.metadata
    assert any("expected_effect" in finding for finding in result.findings)


def test_one_bad_field_among_good_ones_is_partial() -> None:
    """Losing part of a response must not discredit the part that was readable."""
    result = extract_decision({"method": "leiden", "confidence": 12})

    assert result.quality is DecisionQuality.PARTIAL
    assert result.metadata["method"] == "leiden"
    assert "confidence" not in result.metadata
    assert result.observable is True


def test_non_numeric_effect_entries_are_dropped_and_named() -> None:
    """A partly numeric effect map keeps its numbers and reports the rest."""
    result = extract_decision({"expected_effect": {"ari": 0.4, "mood": "better"}})

    assert result.metadata["expected_effect"] == {"ari": 0.4}
    assert any("mood" in finding for finding in result.findings)


def test_agent_cannot_declare_its_own_extraction_verdict() -> None:
    """The verdict is the harness's record of the agent, so the agent cannot set it."""
    result = extract_decision(
        {
            DECISION_QUALITY_KEY: DecisionQuality.STRUCTURED.value,
            DECISION_FINDINGS_KEY: ["nothing to see here"],
        }
    )

    assert result.quality is DecisionQuality.ABSENT
    assert result.metadata[DECISION_QUALITY_KEY] == DecisionQuality.ABSENT.value
    assert result.metadata[DECISION_FINDINGS_KEY] != ["nothing to see here"]
    assert len(result.findings) == 2


def test_unrecognized_keys_are_preserved_verbatim() -> None:
    """An unknown key may be a protocol we have not learned; dropping it loses evidence."""
    result = extract_decision({"method": "leiden", "state_claim": {"n_obs": 500}, "vibes": 3})

    assert result.metadata["vibes"] == 3
    assert result.metadata["state_claim"] == {"n_obs": 500}
    assert "state_claim" in CANONICAL_DECISION_KEYS


# --------------------------------------------------------------------------- #
# Wiring into the turn loop
# --------------------------------------------------------------------------- #


def test_turn_metadata_is_extracted_before_it_reaches_the_cascade() -> None:
    """The runtime seam applies extraction, so no consumer sees raw reasoning."""
    action = AgentAction(
        action_type="analyze",
        parameters={"action_id": "analyze", "code": CODE, "method": "leiden"},
        reasoning_metadata={
            "summary": "cluster the cells",
            "evidence_used": "the elbow plot",
            "expected_effect": "much better",
        },
    )

    intent = _action_to_intent(action, FREE_SPECIFICATION)

    assert intent.rationale == "cluster the cells"
    assert intent.metadata["evidence_used"] == ["the elbow plot"]
    assert "expected_effect" not in intent.metadata
    assert intent.metadata[DECISION_QUALITY_KEY] == DecisionQuality.PARTIAL.value
    # Passed through, because the runtime reads them back for its own reporting.
    assert intent.metadata["summary"] == "cluster the cells"


def test_turn_metadata_keeps_the_benchmark_artifact_contract() -> None:
    """Extraction must not displace the declared inputs and outputs."""
    action = AgentAction(
        action_type="qc",
        parameters={"action_id": "qc"},
        reasoning_metadata={"method": "qc_filter"},
    )

    intent = _action_to_intent(action, TYPED_SPECIFICATION)

    assert intent.metadata["expected_outputs"] == ["qc-table"]
    assert intent.metadata["runtime_action_type"] == "qc"


# --------------------------------------------------------------------------- #
# One action record per observed execution
# --------------------------------------------------------------------------- #


class FakeObservationBuilder:
    """Report a workspace listing without touching a filesystem."""

    async def build(self, specification: Any, task: Any, snapshot: Any) -> list[Observation]:
        return [
            Observation(
                observation_id="workspace-tree",
                value={"inputs/pbmc.h5ad": 1024},
                source="fake-workspace",
            )
        ]


class FakeFreeExecutor:
    """Stand in for a workspace backend, recording what it was asked to run."""

    def __init__(self) -> None:
        self.intents: list[ActionIntent] = []

    async def execute(
        self,
        intent: ActionIntent,
        context: ExecutionContext,
    ) -> ActionExecutionResult:
        self.intents.append(intent)
        return ActionExecutionResult(
            intent_id=intent.intent_id,
            action_id=intent.action_id,
            status=ActionStatus.SUCCEEDED,
            artifacts=[
                ArtifactRecord(
                    artifact_id="cell-labels",
                    kind="table",
                    format="csv",
                    validated=True,
                )
            ],
            resource_usage=ResourceUsage(),
        )


class FakeRewardEvaluator:
    """Emit one deterministic reward so the episode advances."""

    async def evaluate(
        self,
        specification: Any,
        task: Any,
        snapshot: Any,
        result: ActionExecutionResult,
    ) -> RewardRecord:
        return RewardRecord(value=1.0, strategy_id="annotation-reward", step=0)


def make_free_environment(executor: Any) -> ScientificEnvironment:
    """Construct the free-execution environment with only in-memory ports."""
    return ScientificEnvironment(
        FREE_SPECIFICATION,
        task_id="cell-annotation-free",
        executor=executor,
        observation_builder=FakeObservationBuilder(),
        reward_evaluator=FakeRewardEvaluator(),
    )


@pytest.mark.asyncio
async def test_free_execution_turn_produces_exactly_one_action_record() -> None:
    """Free-form work must be as countable as a typed action, or T is meaningless."""
    executor = FakeFreeExecutor()
    environment = make_free_environment(executor)
    await environment.reset(seed=42, dataset_id="pbmc68k")

    await environment.step(
        ActionIntent(
            action_id="analyze",
            parameters={"code": CODE, "language": "python", "method": "leiden"},
            metadata={"method": "leiden", "decision_category": "clustering"},
        )
    )

    assert environment.episode is not None
    state = environment.episode.snapshot().state
    assert len(state.actions) == 1
    assert state.actions[0].intent.action_id == "analyze"
    assert executor.intents[0].parameters["code"] == CODE


@pytest.mark.asyncio
async def test_free_execution_decisions_exclude_the_program_as_a_parameter() -> None:
    """A parameter decision whose value is an entire script is not a method choice."""
    environment = make_free_environment(FakeFreeExecutor())
    await environment.reset(seed=42, dataset_id="pbmc68k")
    await environment.step(
        ActionIntent(
            action_id="analyze",
            parameters={"code": CODE, "language": "python", "method": "leiden"},
            metadata={"method": "leiden", "decision_category": "clustering"},
        )
    )
    assert environment.episode is not None

    cascade = decision_cascade_from_episode(environment.episode.snapshot())

    parameter_names = {
        choice.name
        for decision in cascade.decisions
        if decision.method_choice is not None
        for choice in decision.method_choice.parameters
    }
    assert not parameter_names & EXECUTION_PARAMETERS
    assert [decision.method for decision in cascade.decisions if decision.method] == [
        "leiden",
        "leiden",
    ]


@pytest.mark.asyncio
async def test_unreadable_reasoning_still_yields_a_recorded_decision() -> None:
    """A malformed decision is scored as malformed, never skipped."""
    environment = make_free_environment(FakeFreeExecutor())
    await environment.reset(seed=42, dataset_id="pbmc68k")
    action = AgentAction(
        action_type="analyze",
        parameters={"action_id": "analyze", "code": CODE},
        reasoning_metadata={"decision": "I'll just try clustering and see"},
    )
    await environment.step(_action_to_intent(action, FREE_SPECIFICATION))
    assert environment.episode is not None

    cascade = decision_cascade_from_episode(environment.episode.snapshot())

    assert len(cascade.decisions) == 1
    recorded = cascade.decisions[0]
    assert recorded.metadata[DECISION_QUALITY_KEY] == DecisionQuality.MALFORMED.value
    assert recorded.metadata[DECISION_TEXT_KEY] == "I'll just try clustering and see"
    # No method was stated, so no method decision may be invented for one.
    assert recorded.method is None


# --------------------------------------------------------------------------- #
# Kind routing
# --------------------------------------------------------------------------- #


class RecordingExecutor:
    """Record whether it was reached, for routing assertions only."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[str] = []

    async def execute(
        self,
        intent: ActionIntent,
        context: ExecutionContext,
    ) -> ActionExecutionResult:
        self.calls.append(intent.action_id)
        return ActionExecutionResult(
            intent_id=intent.intent_id,
            action_id=intent.action_id,
            status=ActionStatus.SUCCEEDED,
            resource_usage=ResourceUsage(),
        )


@pytest.mark.asyncio
async def test_router_sends_each_action_kind_to_its_own_executor() -> None:
    """Typed actions and free execution coexist behind the one existing port."""
    typed = RecordingExecutor("typed")
    free = RecordingExecutor("free")
    router = ActionKindRouter(typed=typed, free=free, free_execution_ids={"analyze"})
    # The router never reads the context; it only forwards it.
    context = ExecutionContext(
        snapshot=cast(Any, None),
        constraints=FREE_SPECIFICATION.constraints,
    )

    await router.execute(ActionIntent(action_id="analyze", parameters={"code": CODE}), context)
    await router.execute(ActionIntent(action_id="qc", parameters={}), context)

    assert free.calls == ["analyze"]
    assert typed.calls == ["qc"]


def test_router_reads_action_kinds_from_the_benchmark() -> None:
    """The benchmark declares which actions the agent implements itself."""
    router = ActionKindRouter.from_specification(
        FREE_SPECIFICATION,
        typed=RecordingExecutor("typed"),
        free=RecordingExecutor("free"),
    )

    assert router.free_execution_ids == {"analyze"}
    # The typed catalog yields an empty set because every action it declares is
    # typed, not because it declares none -- an empty catalog would satisfy the
    # assertion below for the wrong reason.
    assert TYPED_SPECIFICATION.actions
    assert all(action.kind is ActionKind.TYPED for action in TYPED_SPECIFICATION.actions)
    assert free_execution_action_ids(TYPED_SPECIFICATION) == frozenset()


def test_router_refuses_free_actions_with_no_workspace_to_run_them() -> None:
    """A wiring gap must not be charged to the agent as a failed action."""
    with pytest.raises(ValueError, match="no workspace executor"):
        ActionKindRouter(typed=RecordingExecutor("typed"), free_execution_ids={"analyze"})


# --------------------------------------------------------------------------- #
# Unmeasured is not perfect
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_run_that_never_stepped_has_nothing_to_average() -> None:
    """The OpenHands shape: reset, run your own code, never call ``step()``.

    Both inputs to the decision dimension come out empty, and an average over
    nothing is unmeasured rather than perfect. This is the shape that used to
    collect D = 1.0 for free.
    """
    environment = make_free_environment(FakeFreeExecutor())
    snapshot = await environment.reset(seed=42, dataset_id="pbmc68k")
    run = AgentRun(
        run_id="run-never-stepped",
        agent_id="agent-under-test",
        configuration=AgentConfiguration(agent_type="external"),
        adapter_name="fake",
        adapter_version="0.0.0",
        benchmark_id=FREE_SPECIFICATION.metadata.id,
        task_id=environment.task.id,
        episode_id=snapshot.state.episode_id,
        started_at=snapshot.state.started_at or utc_now(),
        finished_at=utc_now(),
        termination_status=RunTerminationStatus.COMPLETED,
        wall_clock_seconds=1.0,
        trajectory=NormalizedTrajectory(
            run_id="run-never-stepped",
            episode_id=snapshot.state.episode_id,
        ),
        final_environment_state=snapshot,
    )

    assert snapshot.state.actions == []
    assert decision_cascade_from_episode(snapshot).decisions == []
    assert DecisionEvaluator().evaluate(run, environment.task) == []


def test_no_decisions_yields_no_global_score_rather_than_a_free_one() -> None:
    """Substituting a neutral 1.0 paid an agent for recording nothing."""
    assert compute_global_agent_score(0.8, None, 0.9) is None
    assert compute_global_agent_score(None, 0.8, 0.9) is None

    scored = compute_global_agent_score(0.8, 0.5, 1.0)
    assert scored is not None
    assert scored.value == pytest.approx((0.8 * 0.5 * 1.0) ** (1 / 3))


def test_the_persisted_formula_names_the_dimension_it_could_not_measure() -> None:
    """A reader of the result JSON must not have to guess why there is no score."""
    weights = ScoreWeights.neutral()
    complete = _score_formula(
        weights, scientific_outcome=0.8, decision=0.5, selection=0.9
    )
    partial = _score_formula(
        weights, scientific_outcome=0.8, decision=None, selection=None
    )

    assert "not computed" not in complete
    assert "not computed: decision_score, method_selection_score unmeasured" in partial


def test_an_unmeasured_score_reads_as_unmeasured_not_as_a_failure() -> None:
    """Rendering ``None`` as ``0`` or a traceback would misreport a gap."""
    assert _unmeasured_or(None) == "unmeasured"
    assert _unmeasured_or(0.0) == "0.0"


# --------------------------------------------------------------------------- #
# Benchmark DSL
# --------------------------------------------------------------------------- #


def test_free_execution_example_declares_a_workspace_and_no_fixed_outputs() -> None:
    """The shipped example is the proof the DSL can express this benchmark."""
    task = FREE_SPECIFICATION.tasks[0]
    environment = next(
        item for item in FREE_SPECIFICATION.environments if item.id == task.environment
    )
    action = next(item for item in FREE_SPECIFICATION.actions if item.id == "analyze")

    assert action.kind is ActionKind.FREE_EXECUTION
    assert action.expected_outputs == []
    assert EXECUTION_PARAMETERS <= {parameter.name for parameter in action.parameters}
    assert environment.backend is EnvironmentBackend.LOCAL
    assert environment.languages == ["python", "shell"]
    # Artifacts stay optional: a required one would dictate the pipeline shape
    # this benchmark exists to leave open.
    assert not any(artifact.required for artifact in FREE_SPECIFICATION.artifacts)


def test_free_execution_action_may_not_declare_fixed_expected_outputs() -> None:
    """Two invocations of one free action legitimately produce different files."""
    with pytest.raises(ValidationError, match="expected_outputs"):
        ActionSpecification(
            id="analyze",
            name="Run analysis code",
            purpose="Execute agent-authored source.",
            kind=ActionKind.FREE_EXECUTION,
            expected_outputs=["cell-labels"],
        )


def test_container_environment_must_name_an_image() -> None:
    """Without a pinned image the run depends on whatever the host had cached."""
    with pytest.raises(ValidationError, match="image"):
        EnvironmentSpecification(
            id="container-python",
            name="Container workspace",
            description="Isolated container workspace.",
            backend=EnvironmentBackend.CONTAINER,
        )


def test_local_environment_must_not_name_an_image() -> None:
    """A declared image that nothing honours is a false claim about the run."""
    with pytest.raises(ValidationError, match="image"):
        EnvironmentSpecification(
            id="local-python",
            name="Local workspace",
            description="Local workspace.",
            backend=EnvironmentBackend.LOCAL,
            image="scaib/exec:1.0",
        )


def test_environment_must_declare_at_least_one_language() -> None:
    """An environment that runs nothing cannot host a free-execution action."""
    with pytest.raises(ValidationError, match="at least one language"):
        EnvironmentSpecification(
            id="empty",
            name="Empty workspace",
            description="Runs nothing.",
            languages=[],
        )


def test_task_allowing_free_execution_must_declare_an_environment() -> None:
    """Otherwise the benchmark validates and then fails at run time on the agent."""
    payload = FREE_SPECIFICATION.model_dump(mode="json")
    payload["tasks"][0].pop("environment")

    with pytest.raises(ValidationError, match="declares no 'environment'"):
        benchmark_from_dict(payload)


def test_task_may_not_reference_an_undeclared_environment() -> None:
    """A typo in the environment id must fail loudly at load time."""
    payload = FREE_SPECIFICATION.model_dump(mode="json")
    payload["tasks"][0]["environment"] = "local-pythn"

    with pytest.raises(ValidationError, match="unknown environment"):
        benchmark_from_dict(payload)
