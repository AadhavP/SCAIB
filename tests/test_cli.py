"""Tests for Typer CLI commands."""

from types import SimpleNamespace

from typer.testing import CliRunner

from agent_evals.agents.trajectory import RunTerminationStatus
from agent_evals.cli.main import app
from agent_evals.environment.scientific_loop import DEFAULT_RUNTIME_MAX_STEPS

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "agent-evals version" in result.stdout


def test_cli_list_benchmarks() -> None:
    result = runner.invoke(app, ["list-benchmarks"])
    assert result.exit_code == 0
    assert "Registered Benchmarks" in result.stdout


def test_cli_list_agents() -> None:
    result = runner.invoke(app, ["list-agents"])
    assert result.exit_code == 0
    assert "Registered Agent Adapters" in result.stdout


def test_cli_agent_list_runtimes() -> None:
    result = runner.invoke(app, ["agent", "list"])
    assert result.exit_code == 0
    assert "Available Agents" in result.stdout
    assert "openai" in result.stdout


def test_cli_run_routes_universal_runtime_to_scientific_loop(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeScientificLoop:
        async def run(self, benchmark: str, **kwargs: object) -> object:
            calls.update({"benchmark": benchmark, **kwargs})
            return SimpleNamespace(
                run_id="run-1",
                agent_run=SimpleNamespace(termination_status=RunTerminationStatus.COMPLETED),
                evaluation=SimpleNamespace(global_agent_score=0.5),
                report_path="results/run-1/report.md",
            )

    monkeypatch.setattr("agent_evals.environment.scientific_loop.ScientificLoop", FakeScientificLoop)

    result = runner.invoke(
        app,
        ["run", "--agent", "openai", "--model", "gpt-5", "--max-cells", "12"],
    )

    assert result.exit_code == 0
    assert calls["agent_type"] == "openai"
    assert calls["model"] == "gpt-5"
    assert calls["max_cells"] == 12
    assert calls["max_steps"] == DEFAULT_RUNTIME_MAX_STEPS
