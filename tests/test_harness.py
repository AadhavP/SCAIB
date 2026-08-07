"""Vertical-slice tests for adapters, runs, trajectories, and workspaces."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_evals.agents import (
    AgentConfiguration,
    AgentHarness,
    MockActionExecutor,
    MockAgentAdapter,
    MockObservationBuilder,
    OpenHandsAdapter,
    agent_adapter_registry,
)
from agent_evals.agents.trajectory import (
    DecisionCascade,
    RunTerminationStatus,
    ScientificDecision,
)
from agent_evals.benchmarks.io import load_benchmark
from agent_evals.environment import (
    LocalWorkspace,
    ScientificEnvironment,
    WorkspaceStatus,
)

SPECIFICATION = load_benchmark(
    Path(__file__).parents[1] / "examples" / "benchmarks" / "pbmc-cell-annotation.yaml"
)


def make_environment() -> ScientificEnvironment:
    """Construct the mock-backed scientific environment."""
    return ScientificEnvironment(
        SPECIFICATION,
        task_id="cell-annotation",
        executor=MockActionExecutor(),
        observation_builder=MockObservationBuilder(),
    )


@pytest.mark.asyncio
async def test_mock_agent_produces_replayable_agent_run() -> None:
    """The framework-independent mock adapter completes a real vertical slice."""
    run = await AgentHarness().run(
        MockAgentAdapter(),
        make_environment(),
        AgentConfiguration(agent_type="mock", seed=42),
    )

    assert run.succeeded
    assert run.termination_status == RunTerminationStatus.COMPLETED
    assert run.step_count == 1
    assert run.trajectory.decisions.decisions[0].action_category == "qc"
    assert run.raw_events
    assert run.trajectory.events
    restored = run.from_json(run.to_json())
    assert restored.run_id == run.run_id
    assert restored.trajectory.decisions == run.trajectory.decisions


def test_decision_cascade_preserves_parent_relationships() -> None:
    """Hierarchical decisions survive JSON serialization and validation."""
    parent = ScientificDecision(
        decision_id="qc",
        episode_id="episode",
        step_id="step-1",
        order=0,
        decision_type="step_selection",
        action_category="quality_control",
        timestamp=datetime(2026, 8, 7, tzinfo=UTC),
    )
    child = ScientificDecision(
        decision_id="qc-threshold",
        episode_id="episode",
        step_id="step-1",
        order=1,
        decision_type="parameter_selection",
        action_category="quality_control",
        method="mitochondrial_filter",
        parameters={"threshold": 0.1},
        parent_decision_id="qc",
        timestamp=datetime(2026, 8, 7, 0, 0, 1, tzinfo=UTC),
    )
    cascade = DecisionCascade(decisions=[parent, child])

    assert cascade.decisions[1].parent_decision_id == "qc"
    assert DecisionCascade.model_validate_json(cascade.model_dump_json()) == cascade


@pytest.mark.asyncio
async def test_openhands_unavailable_returns_valid_failed_run() -> None:
    """Core harness operation does not depend on OpenHands installation."""
    adapter = OpenHandsAdapter()
    if adapter.available:
        pytest.skip("OpenHands optional extra is installed in this environment")
    run = await OpenHandsAdapter().run(
        SPECIFICATION.tasks[0],
        make_environment(),
        AgentConfiguration(agent_type="openhands", seed=7),
    )

    assert run.termination_status == RunTerminationStatus.UNAVAILABLE
    assert run.failures
    assert run.final_environment_state.state.status.value == "failed"


@pytest.mark.asyncio
async def test_openhands_factory_path_captures_observable_events(tmp_path: Path) -> None:
    """The adapter preserves the same contract for controlled test sessions."""

    class Session:
        def __init__(self) -> None:
            self.events = [
                {"id": "message-1", "kind": "MessageEvent", "timestamp": "2026-08-07T00:00:00Z"},
                {"id": "action-1", "kind": "ActionEvent", "timestamp": "2026-08-07T00:00:01Z"},
            ]

        def send_message(self, prompt: str) -> None:
            assert "Task specification" in prompt

        def run(self) -> None:
            return None

        def close(self) -> None:
            return None

    def factory(**kwargs: object) -> Session:
        assert kwargs["workspace"] is not None
        return Session()

    run = await OpenHandsAdapter(session_factory=factory).run(
        SPECIFICATION.tasks[0],
        make_environment(),
        AgentConfiguration(
            agent_type="openhands",
            seed=7,
            workspace={"root": str(tmp_path / "openhands")},
        ),
    )

    assert run.succeeded
    assert run.termination_status == RunTerminationStatus.COMPLETED
    assert [event.event_type for event in run.raw_events] == ["message", "action"]
    assert run.metadata["workspace"]["status"] == "closed"


def test_adapter_registry_reports_optional_availability() -> None:
    """Mock is available while OpenHands is represented without import failure."""
    availability = agent_adapter_registry.availability()

    assert {"mock", "openhands"}.issubset(availability)
    assert availability["mock"] is True
    assert isinstance(availability["openhands"], bool)


@pytest.mark.asyncio
async def test_local_workspace_has_portable_lifecycle(tmp_path: Path) -> None:
    """The MVP workspace creates inputs, artifacts, and logs without deleting them."""
    workspace = LocalWorkspace(tmp_path / "episode", workspace_id="episode-1")
    ready = await workspace.initialize()

    assert ready.status == WorkspaceStatus.READY
    assert ready.input_dir.exists()
    assert ready.artifact_dir.exists()
    assert ready.log_dir.exists()
    closed = await workspace.close()
    assert closed.status == WorkspaceStatus.CLOSED
