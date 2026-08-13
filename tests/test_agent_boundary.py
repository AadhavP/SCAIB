"""The two boundaries an agent can be scored across, and what each can measure.

Three subjects, two of which turn out to be one.

**The black-box boundary.** ``http-step`` is the lowest integration tier: the
agent is a URL. These tests pin the four envelope types, the no-retry rule, and
the fact that a token never reaches the persisted manifest.

**The measurability boundary.** An agent that runs its own code leaves its
results in workspace files, so the object the evaluator scores never sees them --
and an absent prediction column there says nothing whatever about the agent. It
used to be scored anyway, at exactly zero, on every free-execution run. The
outcome must come back *unmeasured* instead, and the tests below are the only
thing that can see that: an unmeasured outcome blocks nothing, so the suite was
green throughout.

The trap worth recording is in ``test_omitting_only_the_candidate_still_...``.
Withholding the unobservable candidate on its own does **not** fix the zero --
a missing entry in ``required_artifacts`` leaves the metric *eligible* and the
engine scores it at ``failure_score``, which is the same manufactured zero by a
longer route. Only an unavailable *reference* is structurally ineligible, so the
two must be withheld in one decision.

**The four consequences nothing was watching.** Mutation-testing this stage found
four fixes that no test in the repository could see being undone, and the last
section here is their guard. Each is a place where the free tier's optional
artifacts and unjoinable candidate change what a previously-safe rule means: the
per-step ``S_t`` re-deciding the tier for itself, an artifact list read as a
requirement list, an *empty* requirement set making a subset test vacuously true,
and a validation rule retargeted onto the answer key. All four fail silently, and
one of them -- the vacuous subset -- had its load-bearing guard asserted only in a
docstring.
"""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("anndata")

import anndata
import numpy as np
import pandas as pd

from agent_evals.agents.backends.http_step import (
    ENDPOINT_VARIABLE,
    TOKEN_VARIABLE,
    HttpStepError,
    HttpStepRuntime,
    public_endpoint,
)
from agent_evals.agents.runtime import agent_runtime_registry
from agent_evals.agents.runtime.manager import RuntimeVerdict, cutoff_termination
from agent_evals.agents.runtime.protocol import (
    AgentContext,
    AgentObservation,
    AgentSession,
)
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.schema import (
    ActionKind,
    ActionSpecification,
    BenchmarkSpecification,
    TaskSpecification,
)
from agent_evals.core.artifact_rules import RuleKind, parse_validation_rule
from agent_evals.core.reference_columns import RESERVED_REFERENCE_COLUMNS
from agent_evals.environment.cutoff import CutoffDecision, CutoffReason
from agent_evals.environment.episode import Episode
from agent_evals.environment.models import (
    ActionExecutionResult,
    ActionStatus,
    ArtifactRecord,
    EpisodeSnapshot,
    RewardRecord,
)
from agent_evals.environment.provisioning import (
    provision_environment,
    select_environment,
)
from agent_evals.environment.runtime import ScientificEnvironment
from agent_evals.evaluation.candidates import (
    UNASSIGNED_LABEL,
    UNJOINABLE_CANDIDATE_GAP,
    build_metric_inputs,
)
from agent_evals.evaluation.profiles.pbmc_annotation import pbmc_annotation_profile
from agent_evals.evaluation.progress import ProgressSignal, ScientificProgressTracker
from agent_evals.evaluation.scientific import ScientificMetricEngine
from agent_evals.evaluation.scoring.aggregation import (
    MetricScoreInput,
    WeightedGeometricAggregator,
)
from agent_evals.evaluation.scoring.domains import aggregate_domains
from agent_evals.evaluation.stage_rewards import StageAwareRewardEvaluator
from agent_evals.metrics.applicability import (
    ApplicabilityContext,
    MetricApplicabilityEngine,
)
from agent_evals.metrics.context import ScientificMetricContext
from agent_evals.metrics.registry import metric_registry
from agent_evals.metrics.results import MetricStatus

EXAMPLES = Path(__file__).parents[1] / "examples" / "benchmarks"

#: Every shipped benchmark, so a rule added to a new one is covered on arrival
#: rather than when somebody remembers to extend a hardcoded list.
BENCHMARKS = sorted(path.name for path in EXAMPLES.glob("*.yaml"))

#: The annotation profile's biology domain, which is 0.6 of the outcome and the
#: domain that collapsed to zero. Every entry consumes the reference.
BIOLOGY_METRICS = ["clustering.ari", "cell_annotation.macro_f1", "cell_annotation.rare_recall"]


def _adata(cells: int = 60) -> anndata.AnnData:
    """A dataset carrying reference labels inline, as the fixtures really do."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "bulk_labels": pd.Categorical(["B", "T", "NK"] * (cells // 3)),
            "batch": pd.Categorical(["a", "b"] * (cells // 2)),
        },
        index=[f"cell-{index}" for index in range(cells)],
    )
    adata = anndata.AnnData(
        X=rng.random((cells, 4), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["CD3D", "MS4A1", "GNLY", "LYZ"]),
    )
    adata.obsm["X_pca"] = rng.random((cells, 3), dtype=np.float32)
    return adata


def _free_task() -> tuple[BenchmarkSpecification, TaskSpecification]:
    specification = load_benchmark(EXAMPLES / "pbmc-cell-annotation-free.yaml")
    return specification, specification.tasks[0]


def _typed_task() -> tuple[BenchmarkSpecification, TaskSpecification]:
    specification = load_benchmark(EXAMPLES / "pbmc-cell-annotation.yaml")
    return specification, specification.tasks[0]


# --------------------------------------------------------------------------- #
# Which tier can observe a prediction at all
# --------------------------------------------------------------------------- #


def test_the_typed_tier_reports_that_the_evaluator_runs_the_science() -> None:
    """SCAIB executes every typed action, so results reach the scored object."""
    specification, task = _typed_task()

    assert specification.evaluator_executes_task_actions(task) is True


def test_the_free_tier_reports_that_the_evaluator_does_not_run_the_science() -> None:
    """Every allowed action is free execution, so the scored copy is untouched."""
    specification, task = _free_task()

    assert task.allowed_actions
    kinds = {
        action.kind
        for action in specification.actions
        if action.id in set(task.allowed_actions)
    }
    assert kinds == {ActionKind.FREE_EXECUTION}
    assert specification.evaluator_executes_task_actions(task) is False


def test_a_task_naming_no_declared_action_is_treated_as_evaluator_executed() -> None:
    """Fail closed, because the permissive answer excuses every metric.

    Answering False here would make the cheapest route to an unpunished bad run a
    benchmark whose action list does not line up with its tasks -- a scoring gap
    that opens on a typo.
    """
    specification, task = _free_task()
    orphan = task.model_copy(update={"allowed_actions": ["no-such-action"]})

    assert specification.evaluator_executes_task_actions(orphan) is True


def test_a_mixed_task_counts_as_evaluator_executed() -> None:
    """One typed action reaching the scored copy is a gap the harness can defend."""
    specification, task = _free_task()
    free_action = next(
        action
        for action in specification.actions
        if action.kind is ActionKind.FREE_EXECUTION
    )
    typed = ActionSpecification(
        id="typed-qc",
        name="Typed QC",
        purpose="A benchmark-executed action beside the free one.",
    )
    mixed = specification.model_copy(
        update={"actions": [*specification.actions, typed]}
    )
    task = task.model_copy(
        update={"allowed_actions": [free_action.id, "typed-qc"]}
    )

    assert mixed.evaluator_executes_task_actions(task) is True


# --------------------------------------------------------------------------- #
# What gets withheld, and that both sides move together
# --------------------------------------------------------------------------- #


def test_an_unobservable_candidate_withholds_the_reference_with_it() -> None:
    """Both sides, one decision: half the withholding manufactures the zero."""
    inputs = build_metric_inputs(
        _adata(),
        prediction_column=None,
        cluster_column=None,
        evaluator_observes_predictions=False,
    )

    assert "prediction" not in inputs.candidate_artifacts
    assert inputs.reference_artifacts == {}
    assert inputs.prediction is None
    assert inputs.reference_join_gap == UNJOINABLE_CANDIDATE_GAP


def test_evidence_the_agent_did_produce_survives_the_withholding() -> None:
    """The join gap is about predictions, not about everything the agent made."""
    adata = _adata()
    adata.obs["leiden"] = pd.Categorical(["0", "1"] * (adata.n_obs // 2))
    inputs = build_metric_inputs(
        adata,
        prediction_column=None,
        cluster_column="leiden",
        evaluator_observes_predictions=False,
    )

    assert "cluster_labels" in inputs.candidate_artifacts
    assert "embedding" in inputs.candidate_artifacts


def test_the_typed_tier_still_scores_an_absent_prediction_at_zero() -> None:
    """An agent that could annotate and did not is measured, not excused."""
    inputs = build_metric_inputs(
        _adata(),
        prediction_column=None,
        cluster_column=None,
        evaluator_observes_predictions=True,
    )

    assert inputs.reference_join_gap is None
    assert inputs.limitations == ()
    labels = set(inputs.candidate_artifacts["prediction"]["predicted_label"])
    assert labels == {UNASSIGNED_LABEL}
    assert "labels" in inputs.reference_artifacts


def test_an_agent_written_prediction_is_scored_on_both_tiers() -> None:
    """Nothing is withheld once the evaluator can see the agent's own column."""
    adata = _adata()
    adata.obs["predicted_label"] = pd.Categorical(["B", "T", "NK"] * (adata.n_obs // 3))
    for observes in (True, False):
        inputs = build_metric_inputs(
            adata,
            prediction_column="predicted_label",
            cluster_column=None,
            evaluator_observes_predictions=observes,
        )

        assert inputs.reference_join_gap is None
        assert "labels" in inputs.reference_artifacts
        assert set(inputs.candidate_artifacts["prediction"]["predicted_label"]) == {
            "B",
            "T",
            "NK",
        }


def test_the_published_limitation_cannot_disagree_with_the_join_gap() -> None:
    """Derived, not stored: two wordings of one gap read as two gaps."""
    withheld = build_metric_inputs(
        _adata(),
        prediction_column=None,
        cluster_column=None,
        evaluator_observes_predictions=False,
    )
    scored = build_metric_inputs(
        _adata(),
        prediction_column=None,
        cluster_column=None,
        evaluator_observes_predictions=True,
    )

    assert withheld.limitations == (withheld.reference_join_gap,)
    assert scored.limitations == ()


# --------------------------------------------------------------------------- #
# The consequence: unmeasured, not zero
# --------------------------------------------------------------------------- #


def _evaluate(inputs: Any, adata: Any) -> Any:
    engine = ScientificMetricEngine()
    context = ScientificMetricContext(
        adata=adata,
        candidate_artifacts=inputs.candidate_artifacts,
        reference_artifacts=inputs.reference_artifacts,
        reference_join_gap=inputs.reference_join_gap,
    )
    results, _decisions, _groups, _score = engine.evaluate(BIOLOGY_METRICS, context)
    return results


def test_reference_metrics_are_ineligible_rather_than_scored_zero() -> None:
    """``INELIGIBLE`` leaves the aggregate; ``MISSING`` is charged at zero."""
    adata = _adata()
    inputs = build_metric_inputs(
        adata,
        prediction_column=None,
        cluster_column=None,
        evaluator_observes_predictions=False,
    )

    results = _evaluate(inputs, adata)

    assert [result.status for result in results] == [MetricStatus.INELIGIBLE] * 3
    assert all(result.status.excluded_from_scoring for result in results)
    assert all(result.normalized_value is None for result in results)


def test_the_recorded_exclusion_names_the_join_and_not_the_benchmark_task() -> None:
    """A persisted exclusion blaming the wrong cause is worse than a vague one."""
    adata = _adata()
    inputs = build_metric_inputs(
        adata,
        prediction_column=None,
        cluster_column=None,
        evaluator_observes_predictions=False,
    )

    results = _evaluate(inputs, adata)

    for result in results:
        assert result.eligibility_reason == UNJOINABLE_CANDIDATE_GAP
        assert "not provided by the benchmark task" not in (result.eligibility_reason or "")


def test_the_biology_domain_is_unmeasured_rather_than_zero_on_the_free_tier() -> None:
    """The whole point: ``O`` comes back ``None``, and 0.0 would be a lie.

    Every biology entry is ``required``, so one excluded entry collapses the
    domain -- which is the honest outcome here and would be the wrong one if the
    exclusion were manufactured.
    """
    adata = _adata()
    inputs = build_metric_inputs(
        adata,
        prediction_column=None,
        cluster_column=None,
        evaluator_observes_predictions=False,
    )
    profile = pbmc_annotation_profile()

    results = _evaluate(inputs, adata)
    domain = WeightedGeometricAggregator().aggregate(
        "biology",
        profile.metric_groups["biology"],
        [MetricScoreInput.from_metric_result(result) for result in results],
    )
    outcome = aggregate_domains([domain])

    assert domain.value is None
    assert sorted(domain.excluded_metrics) == sorted(BIOLOGY_METRICS)
    assert domain.failed_metrics == []
    assert outcome.value is None
    assert "excluding_unmeasured" in outcome.formula


def test_the_typed_tier_keeps_its_measured_zero() -> None:
    """The regression guard on the other side: the fix must not excuse a real gap."""
    adata = _adata()
    inputs = build_metric_inputs(
        adata,
        prediction_column=None,
        cluster_column=None,
        evaluator_observes_predictions=True,
    )
    profile = pbmc_annotation_profile()

    results = _evaluate(inputs, adata)
    domain = WeightedGeometricAggregator().aggregate(
        "biology",
        profile.metric_groups["biology"],
        [MetricScoreInput.from_metric_result(result) for result in results],
    )

    assert MetricStatus.INELIGIBLE not in {result.status for result in results}
    assert domain.value == 0.0


def test_omitting_only_the_candidate_still_manufactures_a_zero() -> None:
    """Why both sides move together, recorded as an executable fact.

    This is the fix that looks right and is not. With the reference still
    presented, a missing ``required_artifacts`` entry is *eligible* with a missing
    input, and the engine charges it at ``failure_score`` -- the same zero.
    """
    adata = _adata()
    definition = metric_registry.get("cell_annotation.macro_f1")
    context = ApplicabilityContext(
        structural_artifacts={"labels"},
        reference_labels_available=True,
        candidate_artifacts=set(),
    )

    decision = MetricApplicabilityEngine().evaluate(definition, context)

    assert decision.eligible is True
    assert decision.structurally_ineligible is False
    assert "prediction" in decision.missing_candidate_artifacts

    results, _decisions, _groups, _score = ScientificMetricEngine().evaluate(
        ["cell_annotation.macro_f1"],
        ScientificMetricContext(adata=adata, reference_artifacts={"labels": object()}),
    )
    assert results[0].status is MetricStatus.MISSING
    assert results[0].normalized_value == definition.failure_score
    assert not results[0].status.excluded_from_scoring


async def test_the_free_tier_disclosure_is_checked_against_behaviour(
    tmp_path: Path,
) -> None:
    """The tier's own record promises this; here the promise is measured.

    ``ProvisionedEnvironment.limitations()`` tells whoever reads a result that on
    this tier "reference-consuming metrics are structurally ineligible ... and the
    scientific outcome is reported as unmeasured rather than as zero". Nothing
    branches on that string, so nothing but this test can notice it going stale.
    Asserting it against *another string* would not notice either -- the half that
    can rot is the behaviour, so both halves are checked in one place.
    """
    specification, task = _free_task()
    environment = select_environment(specification, task)
    assert environment is not None
    adata = _adata()
    provisioned = await provision_environment(
        specification, environment, adata, run_root=tmp_path
    )
    try:
        disclosure = " ".join(provisioned.limitations())
    finally:
        await provisioned.backend.close()

    assert "unmeasured rather than as zero" in disclosure
    assert specification.evaluator_executes_task_actions(task) is False
    inputs = build_metric_inputs(
        adata,
        prediction_column=None,
        cluster_column=None,
        evaluator_observes_predictions=(
            specification.evaluator_executes_task_actions(task)
        ),
    )
    results = _evaluate(inputs, adata)
    assert [result.status for result in results] == [MetricStatus.INELIGIBLE] * 3
    profile = pbmc_annotation_profile()
    domain = WeightedGeometricAggregator().aggregate(
        "biology",
        profile.metric_groups["biology"],
        [MetricScoreInput.from_metric_result(result) for result in results],
    )
    assert aggregate_domains([domain]).value is None


# --------------------------------------------------------------------------- #
# The black-box boundary
# --------------------------------------------------------------------------- #


class RecordingResponse:
    """Minimal stand-in for an httpx response, so no server is needed."""

    def __init__(self, payload: Any, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class RecordingClient:
    """Answer each envelope type from a script and record every POST."""

    def __init__(self, replies: dict[str, Any] | None = None) -> None:
        self.replies = replies or {}
        self.posts: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.closed = False

    async def post(
        self, url: str, json: dict[str, Any], headers: dict[str, str]
    ) -> RecordingResponse:
        del url
        self.posts.append(json)
        self.headers.append(headers)
        reply = self.replies.get(json["type"], {})
        if isinstance(reply, RecordingResponse):
            return reply
        return RecordingResponse(reply)

    async def aclose(self) -> None:
        self.closed = True


def _runtime(client: RecordingClient, **kwargs: Any) -> HttpStepRuntime:
    return HttpStepRuntime(
        endpoint="https://agent.example/step", client=client, **kwargs
    )


def _observation() -> AgentObservation:
    return AgentObservation(state={"step": 1}, available_actions=["run-analysis"])


def _context() -> AgentContext:
    return AgentContext(benchmark_id="pbmc", task_id="task", workspace="/tmp/ws")


async def test_the_step_endpoint_sees_one_envelope_per_turn() -> None:
    """Four envelope types, each POSTed once, each naming itself."""
    client = RecordingClient(
        {
            "initialize": {"state": {"remote": 1}},
            "observation": {"action_type": "analyze", "parameters": {"code": "pass"}},
            "terminate": {"summary": "done"},
        }
    )
    runtime = _runtime(client)

    session = await runtime.initialize(_context())
    action = await runtime.act(session, _observation())
    submission = await runtime.terminate(session, _observation())

    assert [post["type"] for post in client.posts] == [
        "initialize",
        "observation",
        "terminate",
    ]
    assert action.action_type == "analyze"
    assert submission.summary == "done"
    assert session.state["remote"] == {"remote": 1}


async def test_a_declined_plan_is_not_invented() -> None:
    """An empty reply means "this agent does not plan", which is a real answer."""
    client = RecordingClient({"plan": {}})
    runtime = _runtime(client)

    assert await runtime.plan(_context(), _observation()) is None


async def test_a_failed_step_is_not_retried() -> None:
    """A step is not idempotent: a retry could run the same science twice."""
    client = RecordingClient(
        {"observation": RecordingResponse({}, status_code=503, text="unavailable")}
    )
    runtime = _runtime(client)
    session = AgentSession(context=_context())

    with pytest.raises(HttpStepError, match="503"):
        await runtime.act(session, _observation())

    assert len(client.posts) == 1


@pytest.mark.parametrize(
    "reply",
    [
        RecordingResponse(ValueError("not json")),
        RecordingResponse([1, 2, 3]),
        {"no_action_type": True},
    ],
    ids=["not-json", "not-an-object", "not-an-action"],
)
async def test_an_unusable_reply_raises_rather_than_being_guessed_at(
    reply: Any,
) -> None:
    """Coercing a malformed reply would record an action the agent never chose."""
    runtime = _runtime(RecordingClient({"observation": reply}))

    with pytest.raises(HttpStepError):
        await runtime.act(AgentSession(context=_context()), _observation())


async def test_the_bearer_token_is_sent_but_never_published() -> None:
    """The manifest is persisted; where a run's agent lived is not how to be it."""
    client = RecordingClient({"initialize": {}})
    runtime = _runtime(client, api_key="s3cret")

    await runtime.initialize(_context())

    assert client.headers[0]["authorization"] == "Bearer s3cret"
    assert "s3cret" not in str(runtime.manifest.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://u:p@agent.example/step", "https://agent.example/step"),
        ("https://agent.example/step?key=abc", "https://agent.example/step"),
        ("https://agent.example:8443/step", "https://agent.example:8443/step"),
    ],
)
def test_credentials_are_stripped_before_an_endpoint_is_recorded(
    url: str, expected: str
) -> None:
    assert public_endpoint(url) == expected


def test_an_http_step_agent_without_an_endpoint_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Better than defaulting to localhost, which would score a run against nothing."""
    monkeypatch.delenv(ENDPOINT_VARIABLE, raising=False)
    monkeypatch.delenv(TOKEN_VARIABLE, raising=False)
    monkeypatch.setattr(
        "agent_evals.agents.backends.http_step.get_settings",
        lambda: type("S", (), {"scaib_agent_endpoint": None, "scaib_agent_token": None})(),
    )

    with pytest.raises(HttpStepError, match=ENDPOINT_VARIABLE):
        HttpStepRuntime()


def test_the_black_box_tier_is_reachable_by_name() -> None:
    """A tier nothing can select is a tier nobody can submit to."""
    assert "http-step" in agent_runtime_registry.list()


# --------------------------------------------------------------------------- #
# The four fixes nothing was watching
# --------------------------------------------------------------------------- #


class _WorkspaceState:
    """A state provider for an agent whose results are files, not this object.

    Both tiers get this same view deliberately. The only thing that may decide
    whether an absent prediction is measurable is the *benchmark's* tier, so a
    guard that also varied the state could not tell which input moved the answer.
    """

    def __init__(self, adata: Any) -> None:
        self._adata = adata

    @property
    def adata(self) -> Any:
        return self._adata

    def agent_prediction_column(self) -> str | None:
        return None

    def agent_cluster_column(self) -> str | None:
        return None


class _ConstantReward:
    """A delegate with no opinion, so only the progress wrapper is under test."""

    async def evaluate(
        self,
        specification: BenchmarkSpecification,
        task: TaskSpecification,
        snapshot: EpisodeSnapshot,
        result: ActionExecutionResult,
    ) -> RewardRecord:
        del specification, task, result
        return RewardRecord(value=0.0, step=snapshot.state.current_step)


class _UncalledExecutor:
    """Fills the environment's required executor slot for tests that never step."""

    async def execute(self, intent: Any, context: Any) -> ActionExecutionResult:
        raise AssertionError("this test must not execute an action")


async def _progress_over_two_steps(
    specification: BenchmarkSpecification, task: TaskSpecification, adata: Any
) -> tuple[ProgressSignal, ...]:
    """Drive the real per-step wrapper twice, the minimum a delta needs.

    Two steps because the defect is about ``dS``, and the first step of any run
    honestly has nothing to compare against.
    """
    evaluator = StageAwareRewardEvaluator(
        _ConstantReward(),
        ScientificProgressTracker(pbmc_annotation_profile()),
        _WorkspaceState(adata),
        BIOLOGY_METRICS,
    )
    snapshot = Episode.from_specification(
        specification, task_id=task.id, seed=0
    ).snapshot()
    for _ in range(2):
        await evaluator.evaluate(
            specification,
            task,
            snapshot,
            ActionExecutionResult(
                intent_id="intent",
                action_id=task.allowed_actions[0],
                status=ActionStatus.SUCCEEDED,
            ),
        )
    return evaluator.signals


async def test_free_tier_progress_is_unmeasured_not_a_flat_measured_zero() -> None:
    """``S_t`` must not re-decide the tier for itself, or ``dS`` goes flat at zero.

    ``_metric_values`` asks the benchmark whether the evaluator can observe a
    prediction, exactly as the final outcome does. Answering that question
    independently -- assuming yes -- builds the ``__unassigned__`` placeholder on a
    tier where an absent column means nothing, scores every step at the same
    manufactured zero, and hands the stagnation detector a *measured* ``dS`` of
    0.0 on a run that may be working perfectly. Its own docstring says so; nothing
    checked it.
    """
    specification, task = _free_task()

    signals = await _progress_over_two_steps(specification, task, _adata())

    assert len(signals) == 2
    assert signals[1].delta is None, "an unjoinable candidate cannot yield a delta"
    assert signals[1].scored_metrics == ()
    assert signals[1].scientific_state is None
    assert UNJOINABLE_CANDIDATE_GAP in signals[1].limitations


async def test_typed_tier_progress_still_measures_the_agents_own_omission() -> None:
    """The complement, without which the assertion above is satisfied by silence.

    Identical state, identical wrapper, only the benchmark's tier differs. Here
    the evaluator *would* have seen a prediction, so an absent one is the agent's
    omission and is measured -- which is what proves the free-tier result above
    comes from the tier decision rather than from the harness measuring nothing at
    all.
    """
    specification, task = _typed_task()

    signals = await _progress_over_two_steps(specification, task, _adata())

    assert signals[1].scored_metrics, "the typed tier must still score what it can"
    assert signals[1].scientific_state is not None
    assert UNJOINABLE_CANDIDATE_GAP not in signals[1].limitations


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_requiredness_comes_from_the_artifact_not_from_the_task_list(
    benchmark: str,
) -> None:
    """A task's artifact list is what it *may* produce, never what it must.

    Reading the list as the requirement made the free benchmark's four optional
    artifacts mandatory, so no run of it could ever satisfy its own contract and
    every one was filed incomplete. The invariant is checked against each
    benchmark's own declarations rather than against a hardcoded expectation, so
    it also holds for a benchmark added later.
    """
    specification = load_benchmark(EXAMPLES / benchmark)
    required_ids = {
        artifact.id for artifact in specification.artifacts if artifact.required
    }
    for task in specification.tasks:
        resolved = specification.required_task_artifacts(task)

        assert resolved == required_ids & set(task.artifacts)


def test_the_free_benchmark_requires_nothing_by_design() -> None:
    """The case that made the empty-set guard below reachable at all.

    Recorded as its own fact because it is a deliberate design choice, not an
    accident: a required artifact would let the benchmark dictate the pipeline
    shape it is supposed to be measuring, so per-invocation enforcement runs
    through the action's ``produces`` parameter instead.
    """
    free_specification, free_task = _free_task()
    typed_specification, typed_task = _typed_task()

    assert free_task.artifacts, "the benchmark still describes what it accepts"
    assert free_specification.required_task_artifacts(free_task) == set()
    assert typed_specification.required_task_artifacts(typed_task)


def _environment(
    specification: BenchmarkSpecification, task: TaskSpecification
) -> ScientificEnvironment:
    return ScientificEnvironment(
        specification, task_id=task.id, executor=_UncalledExecutor()
    )


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (CutoffReason.MAX_STEPS, RuntimeVerdict.TIMEOUT),
        (CutoffReason.STAGNATION, RuntimeVerdict.STAGNATED),
    ],
)
def test_a_cutoff_run_requiring_no_artifacts_is_not_recorded_as_completed(
    reason: CutoffReason, expected: RuntimeVerdict
) -> None:
    """The vacuous subset test, which only a benchmark requiring nothing reaches.

    ``required.issubset(produced)`` is ``True`` for an empty ``required``, so
    without the ``required and`` guard every cut-off run of a free-execution
    benchmark is filed as having finished its contract -- a stagnating run and a
    complete one becoming indistinguishable in the archive. The guard was called
    load-bearing in its own docstring and asserted nowhere.
    """
    specification, task = _free_task()

    termination = cutoff_termination(
        CutoffDecision(stop=True, reason=reason, detail="the step budget ran out"),
        _environment(specification, task),
    )

    assert specification.required_task_artifacts(task) == set()
    assert termination.verdict is expected
    assert termination.failure_kind is not None
    assert "the step budget ran out" in termination.reason


async def test_a_cutoff_run_that_met_its_contract_is_still_credited() -> None:
    """The other direction: deleting the check is not a fix either.

    An agent that produced everything the benchmark required and then hit its step
    ceiling completed; recording that as a timeout would punish it for finishing
    early enough to be cut off.
    """
    specification, task = _typed_task()
    environment = _environment(specification, task)
    await environment.reset(seed=0)
    required = specification.required_task_artifacts(task)
    assert environment.episode is not None
    environment.episode.record_outputs(
        [],
        [
            ArtifactRecord(artifact_id=artifact_id, kind="table", format="csv")
            for artifact_id in sorted(required)
        ],
    )

    termination = cutoff_termination(
        CutoffDecision(
            stop=True, reason=CutoffReason.MAX_STEPS, detail="the step budget ran out"
        ),
        environment,
    )

    assert required
    assert termination.verdict is RuntimeVerdict.COMPLETED
    assert termination.failure_kind is None


def _rule_targets(rule: str) -> set[str]:
    """Every column name one declared rule refers to, dotted paths unwrapped."""
    parsed = parse_validation_rule(rule)
    names = set(parsed.columns)
    if parsed.kind is not RuleKind.COLUMNS_INCLUDE and parsed.target:
        names.add(parsed.target)
    return {segment for name in names for segment in (name, name.split(".")[-1])}


@pytest.mark.parametrize("benchmark", BENCHMARKS)
def test_no_validation_rule_can_be_satisfied_only_by_the_answer_key(
    benchmark: str,
) -> None:
    """A rule naming a reserved reference column is unsatisfiable except by fraud.

    The agent is refused permission to write ``cell_type``, so a rule demanding
    that column can be met only by the reference the run is scored against -- it
    reads as a validation requirement and functions as an invitation. Two shipped
    benchmarks had one; both were retargeted to ``predicted_label``, and nothing
    stopped either from drifting back.
    """
    specification = load_benchmark(EXAMPLES / benchmark)

    offending = sorted(
        f"{artifact.id}/{rule.name}: {rule.rule}"
        for artifact in specification.artifacts
        for rule in artifact.validation
        if _rule_targets(rule.rule) & RESERVED_REFERENCE_COLUMNS
    )

    assert not offending, f"{benchmark} declares rules only the reference can satisfy: {offending}"


def test_the_reserved_column_guard_would_notice_a_retargeted_rule() -> None:
    """The guard's own predicate, because the assertion above proves a negative."""
    assert _rule_targets("columns include barcode,predicted_label").isdisjoint(
        RESERVED_REFERENCE_COLUMNS
    )
    assert _rule_targets("columns include barcode,cell_type") & RESERVED_REFERENCE_COLUMNS
    assert (
        _rule_targets("obs.cell_type is non-null and belongs to label_vocabulary")
        & RESERVED_REFERENCE_COLUMNS
    )
