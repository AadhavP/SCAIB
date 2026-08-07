"""Universal runtime protocol and manager tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_evals.agents.backends import (
    AnthropicRuntime,
    CustomPythonRuntime,
    OpenAIRuntime,
)
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


class FakeRuntime(AgentRuntime):
    def __init__(self) -> None:
        self.agent_id = "fake-scientist"
        self.manifest = AgentManifest(name="Fake scientist", type="test")

    async def initialize(self, context: AgentContext) -> AgentSession:
        return AgentSession(context=context)

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        del observation
        actions = ["qc", "normalize", "finish"]
        index = int(session.state.get("index", 0))
        session.state["index"] = index + 1
        return AgentAction(action_type=actions[index])

    async def terminate(
        self,
        session: AgentSession,
        observation: AgentObservation | None = None,
    ) -> FinalSubmission:
        del session, observation
        return FinalSubmission(summary="completed")


@pytest.mark.asyncio
async def test_runtime_manager_normalizes_actions_and_captures_full_trajectory() -> None:
    context = AgentContext(benchmark_id="pbmc-cell-annotation", task_id="cell-annotation", workspace=".")
    result = await AgentRuntimeManager().run(
        FakeRuntime(),
        make_environment(),
        context,
        seed=3,
    )

    assert result.termination_status == "completed"
    assert result.step_count == 2
    assert [action.action_type for action in result.trajectory.actions] == ["qc", "normalize", "finish"]
    assert any(event.event_type.value == "environment_response" for event in result.trajectory.events)
    assert result.final_submission is not None


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
        RuntimeAgentAdapter(FakeRuntime()),
        make_environment(),
        AgentConfiguration(agent_type="fake-scientist", seed=5),
    )

    assert run.succeeded
    assert run.manifest is not None
    assert run.manifest.name == "Fake scientist"
    assert run.tool_call_count == 0
