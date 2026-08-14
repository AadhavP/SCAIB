"""Universal runtime protocol and manager tests."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from agent_evals.agents.backends import (
    AnthropicRuntime,
    CustomPythonRuntime,
    HttpStepRuntime,
    OpenAICompatibleRuntime,
    OpenAIRuntime,
)
from agent_evals.agents.decisions import (
    DecisionQuality,
    ExtractionMode,
    extract_action_response,
    extract_decision,
)
from agent_evals.agents.harness import AgentHarness
from agent_evals.agents.mock import MockActionExecutor, MockObservationBuilder
from agent_evals.agents.runtime import (
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentObservation,
    AgentPlan,
    AgentRuntime,
    AgentRuntimeManager,
    AgentSession,
    AgentUsage,
    FinalSubmission,
    RuntimeAgentAdapter,
)
from agent_evals.agents.tools import ToolDefinition, ToolExecutor, ToolRegistry
from agent_evals.agents.trajectory import (
    AgentConfiguration,
    decision_cascade_from_episode,
)
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.benchmarks.schema import CutoffSpecification
from agent_evals.environment.cutoff import CutoffReason
from agent_evals.environment.runtime import ScientificEnvironment

SPECIFICATION = load_benchmark(Path(__file__).parents[2] / "examples" / "benchmarks" / "pbmc-cell-annotation.yaml")


def make_environment() -> ScientificEnvironment:
    return ScientificEnvironment(
        SPECIFICATION,
        task_id="cell-annotation",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )


#: Every action the cell-annotation task needs, with the parameters it declares.
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


class FakeRuntime(AgentRuntime):
    """A runtime that terminates before producing the required artifacts."""

    # A class attribute so subclasses can swap in a different workflow.
    workflow: ClassVar[list[tuple[str, dict[str, object]]]] = [
        ("qc", {"min_genes": 200, "max_mito_fraction": 0.2}),
        ("normalize", {"target_sum": 10_000}),
        ("finish", {}),
    ]

    def __init__(self) -> None:
        self.agent_id = "fake-scientist"
        self.manifest = AgentManifest(name="Fake scientist", type="test")

    async def initialize(self, context: AgentContext) -> AgentSession:
        return AgentSession(context=context)

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
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


class CompleteRuntime(FakeRuntime):
    """A runtime that produces every declared artifact before finishing."""

    workflow = COMPLETE_WORKFLOW


class SlowRuntime(CompleteRuntime):
    """Runtime whose provider call must be stopped by the run wall clock."""

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        await asyncio.sleep(0.2)
        return await super().act(session, observation)


class SlowExecutor(MockActionExecutor):
    """Executor whose work, rather than the provider call, exceeds the budget."""

    async def execute(self, intent: Any, context: Any) -> Any:
        await asyncio.sleep(0.2)
        return await super().execute(intent, context)


class _EndpointResponse:
    """Minimal JSON response used to exercise the real HTTP runtime manager path."""

    status_code = 200
    text = ""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def json(self) -> dict[str, object]:
        return self.payload


class _ScriptedEndpoint:
    """A URL-shaped agent that returns one complete typed workflow."""

    def __init__(self) -> None:
        self.posts: list[dict[str, object]] = []
        self.actions = COMPLETE_WORKFLOW

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> _EndpointResponse:
        del url, headers
        self.posts.append(json)
        if json["type"] == "initialize":
            return _EndpointResponse(
                {
                    "state": {"remote_session": "kept"},
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                }
            )
        if json["type"] == "plan":
            return _EndpointResponse(
                {
                    "plan": {
                        "goal": "Complete the declared workflow",
                        "steps": ["Run each evidence-producing action", "Submit"],
                        "success_criteria": ["Required artifacts validate"],
                        "adaptation_policy": "Revise after each observation",
                    }
                }
            )
        if json["type"] == "observation":
            index = sum(post["type"] == "observation" for post in self.posts) - 1
            action_type, parameters = self.actions[index]
            return _EndpointResponse(
                {
                    "action_type": action_type,
                    "parameters": parameters,
                    "state_claim": {"cells_removed": 12} if index == 0 else {},
                }
            )
        if json["type"] == "terminate":
            return _EndpointResponse({"summary": "endpoint workflow complete"})
        raise AssertionError(f"unexpected envelope: {json['type']}")

    async def aclose(self) -> None:
        return None


def test_black_box_extraction_preserves_free_text_provenance_and_claims() -> None:
    """Level-0 responses stay auditable without pretending prose is structured."""
    response = (
        "I will run qc. action_type: qc; method: fixed_threshold; "
        "confidence: 0.8; evidence: low mitochondrial fraction"
    )

    payload, evidence = extract_action_response(
        response,
        available_actions=["qc", "normalize"],
    )

    assert payload["action_type"] == "qc"
    assert payload["reasoning_metadata"]["decision"]["method"] == "fixed_threshold"
    assert evidence.mode is ExtractionMode.FREE_TEXT
    assert len(evidence.raw_sha256) == 64
    assert evidence.raw_length == len(response.encode("utf-8"))
    assert evidence.raw_content_retained is False


def test_decision_extraction_rejects_agent_owned_quality_and_coerces_lists() -> None:
    """Extraction quality is a harness finding, never a score the agent can set."""
    extracted = extract_decision(
        {
            "decision": {
                "method": "harmony",
                "evidence_used": "PCA batch separation",
                "expected_effect": "large improvement",
                "decision_extraction_quality": "structured",
            }
        }
    )

    assert extracted.quality is DecisionQuality.PARTIAL
    assert extracted.metadata["evidence_used"] == ["PCA batch separation"]
    assert "expected_effect" not in extracted.metadata
    assert extracted.metadata["decision_extraction_quality"] == "partial"
    assert any("discarded agent-supplied" in finding for finding in extracted.findings)


@pytest.mark.asyncio
async def test_endpoint_state_claim_is_carried_to_verified_decision_metadata() -> None:
    """The endpoint may claim a delta, but the archive retains it beside actual state."""
    class ClaimingClient:
        async def post(
            self,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> _EndpointResponse:
            del url, headers
            if json["type"] == "observation":
                return _EndpointResponse(
                    {
                        "action_type": "qc",
                        "parameters": {"min_genes": 200, "max_mito_fraction": 0.2},
                        "state_claim": {"cells_removed": 12},
                    }
                )
            return _EndpointResponse({})

        async def aclose(self) -> None:
            return None

    runtime = HttpStepRuntime(
        endpoint="https://agent.example/step", client=ClaimingClient()
    )
    session = await runtime.initialize(
        AgentContext(benchmark_id="b", task_id="t", workspace=".")
    )
    action = await runtime.act(session, AgentObservation())

    assert action.state_claim == {"cells_removed": 12}
    assert action.extraction_evidence is not None


@pytest.mark.asyncio
async def test_http_endpoint_drives_the_same_scientific_loop_contract() -> None:
    """A URL agent must receive feedback after every action and complete normally."""
    endpoint = _ScriptedEndpoint()
    runtime = HttpStepRuntime(endpoint="https://agent.example/step", client=endpoint)

    result = await AgentRuntimeManager().run(
        runtime,
        make_environment(),
        AgentContext(benchmark_id="b", task_id="t", workspace="."),
        seed=3,
    )

    assert result.termination_status == "completed"
    assert result.step_count == len(COMPLETE_WORKFLOW) - 1
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 3
    assert [post["type"] for post in endpoint.posts] == [
        "initialize",
        "plan",
        *("observation" for _ in COMPLETE_WORKFLOW),
        "terminate",
    ]
    observations = [
        post["observation"]
        for post in endpoint.posts
        if post["type"] == "observation"
    ]
    assert observations[0]["metadata"]["task_package"]["task"]["id"] == "cell-annotation"
    assert observations[1]["metadata"]["interaction"]["phase"] == "post_action_review"
    assert result.final_submission is not None
    assert result.final_submission.summary == "endpoint workflow complete"
    first_action = result.final_snapshot.state.actions[0]
    assert first_action.intent.metadata["state_claim"] == {"cells_removed": 12}
    assert first_action.intent.metadata["response_extraction"]["raw_sha256"]


@pytest.mark.asyncio
async def test_termination_failure_is_not_retried_as_a_second_side_effect() -> None:
    class TerminationFailureRuntime(FakeRuntime):
        workflow: ClassVar[list[tuple[str, dict[str, object]]]] = [("finish", {})]

        def __init__(self) -> None:
            super().__init__()
            self.termination_calls = 0

        async def terminate(
            self,
            session: AgentSession,
            observation: AgentObservation | None = None,
        ) -> FinalSubmission:
            del session, observation
            self.termination_calls += 1
            raise RuntimeError("termination endpoint unavailable")

    runtime = TerminationFailureRuntime()
    result = await AgentRuntimeManager().run(
        runtime,
        make_environment(),
        AgentContext(benchmark_id="b", task_id="t", workspace="."),
        seed=3,
    )

    assert result.termination_status == "failed"
    assert runtime.termination_calls == 1
    assert any("no retry was issued" in event.payload.get("error", "") for event in result.trajectory.events)


@pytest.mark.asyncio
async def test_initialization_failure_is_returned_as_a_partial_runtime_result() -> None:
    class BrokenRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        async def initialize(self, context: AgentContext) -> AgentSession:
            del context
            raise RuntimeError("endpoint unavailable")

        async def close(self) -> None:
            self.closed = True

    runtime = BrokenRuntime()
    result = await AgentRuntimeManager().run(
        runtime,
        make_environment(),
        AgentContext(benchmark_id="b", task_id="t", workspace="."),
        seed=3,
    )

    assert result.termination_status == "failed"
    assert "endpoint unavailable" in (result.termination_reason or "")
    assert runtime.closed is True
    assert any(event.event_type.value == "failure" for event in result.trajectory.events)


@pytest.mark.asyncio
async def test_runtime_manager_normalizes_actions_and_captures_full_trajectory() -> None:
    context = AgentContext(benchmark_id="pbmc-cell-annotation", task_id="cell-annotation", workspace=".")
    result = await AgentRuntimeManager().run(
        CompleteRuntime(),
        make_environment(),
        context,
        seed=3,
    )

    assert result.termination_status == "completed"
    assert result.step_count == len(COMPLETE_WORKFLOW) - 1
    assert [action.action_type for action in result.trajectory.actions] == [
        action_type for action_type, _ in COMPLETE_WORKFLOW
    ]
    assert any(event.event_type.value == "environment_response" for event in result.trajectory.events)
    assert result.final_submission is not None
    assert result.final_snapshot.state.actions[0].intent.parameters["method"] == "fixed_threshold"


@pytest.mark.asyncio
async def test_legacy_decision_metadata_is_normalized_before_cascade_persistence() -> None:
    """Malformed direct intents must remain inspectable rather than crash archiving."""
    result = await AgentRuntimeManager().run(
        CompleteRuntime(),
        make_environment(),
        AgentContext(benchmark_id="b", task_id="t", workspace="."),
        seed=3,
    )
    first = result.final_snapshot.state.actions[0]
    first.intent.metadata["expected_effect"] = "not-a-map"
    first.intent.metadata["state_claim"] = "not-a-map"

    cascade = decision_cascade_from_episode(result.final_snapshot)

    assert cascade.decisions
    assert cascade.decisions[0].expected_effect == {}
    assert cascade.decisions[0].claimed_state_delta == {}


@pytest.mark.asyncio
async def test_response_extraction_evidence_is_promoted_to_the_canonical_decision() -> None:
    """The normalized decision graph exposes boundary provenance as typed evidence."""
    result = await AgentRuntimeManager().run(
        HttpStepRuntime(endpoint="https://agent.example/step", client=_ScriptedEndpoint()),
        make_environment(),
        AgentContext(benchmark_id="b", task_id="t", workspace="."),
        seed=3,
        max_steps=1,
    )

    cascade = decision_cascade_from_episode(result.final_snapshot)

    assert cascade.decisions[0].response_extraction is not None
    assert cascade.decisions[0].response_extraction.raw_sha256


@pytest.mark.asyncio
async def test_runtime_wall_clock_bounds_a_slow_environment_step() -> None:
    specification = SPECIFICATION.model_copy(
        update={"cutoff": CutoffSpecification(max_wall_time_seconds=0.05, max_steps=20)}
    )
    environment = ScientificEnvironment(
        specification,
        task_id="cell-annotation",
        executor=SlowExecutor(),
        observation_builder=MockObservationBuilder(),
    )
    context = AgentContext(
        benchmark_id="pbmc-cell-annotation", task_id="cell-annotation", workspace="."
    )

    result = await AgentRuntimeManager().run(
        CompleteRuntime(), environment, context, seed=3
    )

    assert result.termination_status == "timeout"
    assert result.cutoff is not None
    assert result.cutoff.reason is CutoffReason.WALL_TIME
    assert result.step_count == 0


@pytest.mark.asyncio
async def test_runtime_wall_clock_bounds_a_slow_provider_call() -> None:
    specification = SPECIFICATION.model_copy(
        update={"cutoff": CutoffSpecification(max_wall_time_seconds=0.05, max_steps=20)}
    )
    environment = ScientificEnvironment(
        specification,
        task_id="cell-annotation",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )
    context = AgentContext(
        benchmark_id="pbmc-cell-annotation", task_id="cell-annotation", workspace="."
    )

    result = await AgentRuntimeManager().run(
        SlowRuntime(), environment, context, seed=3
    )

    assert result.termination_status == "timeout"
    assert result.termination_reason is not None
    assert "action" in result.termination_reason


@pytest.mark.asyncio
async def test_runtime_usage_reaches_the_cutoff_and_persisted_run_contract() -> None:
    class UsageRuntime(CompleteRuntime):
        async def act(
            self, session: AgentSession, observation: AgentObservation
        ) -> AgentAction:
            action = await super().act(session, observation)
            return action.model_copy(
                update={
                    "usage": AgentUsage(
                        input_tokens=10,
                        output_tokens=5,
                        cost_usd=0.01,
                    )
                }
            )

    specification = SPECIFICATION.model_copy(
        update={"cutoff": CutoffSpecification(max_total_tokens=15, max_steps=20)}
    )
    environment = ScientificEnvironment(
        specification,
        task_id="cell-annotation",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )
    result = await AgentRuntimeManager().run(
        UsageRuntime(),
        environment,
        AgentContext(benchmark_id="b", task_id="t", workspace="."),
        seed=3,
    )

    assert result.termination_status == "timeout"
    assert result.cutoff is not None
    assert result.cutoff.reason is CutoffReason.TOKENS
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 15
    assert result.estimated_cost is not None
    assert result.estimated_cost.amount == 0.01
    assert result.step_count == 1


@pytest.mark.asyncio
async def test_runtime_receives_task_package_and_revised_plan_is_recorded() -> None:
    class RevisingRuntime(CompleteRuntime):
        async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
            del observation
            index = int(session.state.get("index", 0))
            session.state["index"] = index + 1
            action_type, parameters = self.workflow[index]
            return AgentAction(
                action_type=action_type,
                parameters=dict(parameters),
                plan_update=(
                    AgentPlan(
                        goal="Adapt after QC evidence",
                        steps=["Inspect QC result", "Continue with the least destructive valid path"],
                        success_criteria=["Required artifacts validate"],
                        adaptation_policy="Revise after each result",
                    )
                    if index == 1
                    else None
                ),
            )

    context = AgentContext(benchmark_id="pbmc-cell-annotation", task_id="cell-annotation", workspace=".")
    result = await AgentRuntimeManager().run(
        RevisingRuntime(),
        make_environment(),
        context,
        seed=3,
    )

    assert result.termination_status == "completed"
    assert any(event.event_type.value == "plan" and event.payload.get("goal") == "Adapt after QC evidence" for event in result.trajectory.events)
    opening = result.trajectory.observations[0]
    assert opening.metadata["task_package"]["actions"]
    assert opening.metadata["task_package"]["interaction_protocol"]["replanning"]


def test_evaluator_rewards_are_not_in_agent_visible_state() -> None:
    from agent_evals.environment.models import (
        EpisodeState,
        RewardRecord,
        agent_visible_state,
    )

    state = EpisodeState(
        episode_id="e1",
        benchmark_id="b",
        benchmark_version="1.0.0",
        task_id="t",
        seed=0,
        specification_digest="digest",
        rewards=[RewardRecord(value=0.9, metric_values={"held_out_score": 0.9}, step=1)],
    )

    visible = agent_visible_state(state)
    assert visible.rewards == []


@pytest.mark.asyncio
async def test_terminal_action_without_required_artifacts_is_not_completed() -> None:
    """Saying `finish` must not be enough; the declared artifacts must exist."""
    context = AgentContext(benchmark_id="pbmc-cell-annotation", task_id="cell-annotation", workspace=".")
    result = await AgentRuntimeManager().run(
        FakeRuntime(),
        make_environment(),
        context,
        seed=3,
    )

    assert result.termination_status == "incomplete"
    assert any(event.event_type.value == "failure" for event in result.trajectory.events)


class MalformedRuntime(FakeRuntime):
    async def act(self, session: AgentSession, observation: AgentObservation) -> dict[str, object]:
        del session, observation
        return {"parameters": {}}


@pytest.mark.asyncio
async def test_malformed_action_is_recorded_as_runtime_failure() -> None:
    context = AgentContext(benchmark_id="pbmc-cell-annotation", task_id="cell-annotation", workspace=".")
    result = await AgentRuntimeManager().run(MalformedRuntime(), make_environment(), context)

    assert result.termination_status == "failed"
    assert any(event.event_type.value == "failure" for event in result.trajectory.events)


@pytest.mark.asyncio
async def test_tool_registry_executes_only_registered_tools_and_logs_calls() -> None:
    registry = ToolRegistry()
    registry.register(ToolDefinition(name="echo", description="Echo input"), lambda args, context: {**args, "context": context})
    executor = ToolExecutor(registry)

    assert await executor.execute("echo", {"value": 1}, context="test") == {"value": 1, "context": "test"}
    assert executor.call_log[0]["tool"] == "echo"
    with pytest.raises(KeyError):
        await executor.execute("missing")


@pytest.mark.asyncio
async def test_openai_and_anthropic_backends_parse_mock_structured_actions() -> None:
    openai_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text='{"action_type":"qc","parameters":{}}')
        )
    )
    anthropic_client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", name="normalize", input={"target_sum": 10000})]
            )
        )
    )
    context = AgentContext(benchmark_id="b", task_id="t", workspace=".")
    observation = AgentObservation()

    openai = OpenAIRuntime(client=openai_client)
    openai_session = await openai.initialize(context)
    assert (await openai.act(openai_session, observation)).action_type == "qc"
    anthropic = AnthropicRuntime(client=anthropic_client)
    anthropic_session = await anthropic.initialize(context)
    assert (await anthropic.act(anthropic_session, observation)).action_type == "normalize"


@pytest.mark.asyncio
async def test_openai_runtime_builds_sdk_client_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    created_client: dict[str, object] = {}
    create_call: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            created_client.update(kwargs)
            self.responses = SimpleNamespace(
                create=self._create,
            )

        def _create(self, **kwargs: object) -> object:
            create_call.update(kwargs)
            return SimpleNamespace(
                output=[SimpleNamespace(type="function_call", name="qc", arguments='{"min_genes": 200}')]
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    runtime = OpenAIRuntime()
    session = await runtime.initialize(AgentContext(benchmark_id="b", task_id="t", workspace="."))
    action = await runtime.act(session, AgentObservation())

    assert action.action_type == "qc"
    assert action.parameters == {"min_genes": 200}
    assert created_client == {
        "api_key": "test-openai-key",
        "base_url": "https://example.invalid/v1",
    }
    assert isinstance(create_call["instructions"], str)
    assert [message["role"] for message in create_call["input"]] == ["user"]
    assert "api_key" not in runtime.manifest.metadata


@pytest.mark.asyncio
async def test_openai_runtime_parses_responses_text_content_as_action() -> None:
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_kwargs: SimpleNamespace(
                output=[
                    SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                type="output_text",
                                text='{"action_type":"qc","parameters":{"min_genes":200}}',
                            )
                        ]
                    )
                ]
            )
        )
    )
    runtime = OpenAIRuntime(client=client)
    session = await runtime.initialize(AgentContext(benchmark_id="b", task_id="t", workspace="."))

    action = await runtime.act(session, AgentObservation())

    assert action.action_type == "qc"
    assert action.parameters == {"min_genes": 200}
    assert isinstance(session.state["messages"][-1]["content"], str)


@pytest.mark.asyncio
async def test_anthropic_runtime_builds_sdk_client_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    created_client: dict[str, object] = {}
    create_call: dict[str, object] = {}

    class FakeAnthropic:
        def __init__(self, **kwargs: object) -> None:
            created_client.update(kwargs)
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **kwargs: object) -> object:
            create_call.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", name="normalize", input={"target_sum": 10000})]
            )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))

    runtime = AnthropicRuntime()
    session = await runtime.initialize(AgentContext(benchmark_id="b", task_id="t", workspace="."))
    action = await runtime.act(session, AgentObservation())

    assert action.action_type == "normalize"
    assert created_client == {"api_key": "test-anthropic-key"}
    assert isinstance(create_call["system"], str)
    assert "api_key" not in runtime.manifest.metadata


def test_openai_compatible_runtime_does_not_persist_endpoint_metadata() -> None:
    runtime = OpenAICompatibleRuntime(
        model="local-model",
        base_url="https://user:password@example.invalid/v1?token=secret",
    )

    assert runtime.manifest.metadata == {}


@pytest.mark.asyncio
async def test_custom_python_runtime_supports_user_owned_agent() -> None:
    class Agent:
        def act(self, observation: AgentObservation) -> AgentAction:
            del observation
            return AgentAction(action_type="finish")

    runtime = CustomPythonRuntime(Agent())
    session = await runtime.initialize(AgentContext(benchmark_id="b", task_id="t", workspace="."))
    assert (await runtime.act(session, AgentObservation())).action_type == "finish"


@pytest.mark.asyncio
async def test_runtime_adapter_bridges_into_existing_agent_run_contract() -> None:
    run = await AgentHarness().run(
        RuntimeAgentAdapter(CompleteRuntime()),
        make_environment(),
        AgentConfiguration(agent_type="fake-scientist", seed=5),
    )

    assert run.succeeded
    assert run.manifest is not None
    assert run.manifest.name == "Fake scientist"
    assert run.tool_call_count == 0
