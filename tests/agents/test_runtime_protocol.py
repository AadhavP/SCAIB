"""Universal runtime protocol and manager tests."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from agent_evals.agents.backends import (
    AnthropicRuntime,
    CustomPythonRuntime,
    OpenAICompatibleRuntime,
    OpenRouterRuntime,
    OpenAIRuntime,
)
from agent_evals.agents.backends.aliases import build_runtime
from agent_evals.agents.harness import AgentHarness
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
    RuntimeAgentAdapter,
)
from agent_evals.agents.tools import ToolDefinition, ToolExecutor, ToolRegistry
from agent_evals.agents.trajectory import AgentConfiguration
from agent_evals.benchmarks.io import load_benchmark
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


def test_openrouter_runtime_alias_uses_openrouter_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://scaib.local")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "SCAIB")

    runtime = build_runtime("openrouter")

    assert isinstance(runtime, OpenRouterRuntime)
    assert runtime.agent_id == "openrouter"
    assert runtime.model == "openai/gpt-5"
    assert runtime.base_url == "https://openrouter.ai/api/v1"
    assert runtime.default_headers == {
        "HTTP-Referer": "https://scaib.local",
        "X-OpenRouter-Title": "SCAIB",
    }
    assert runtime.manifest.model.provider == "openrouter"
    assert "test-openrouter-key" not in str(runtime.manifest.model_dump(mode="json"))


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
