"""Tests for Typer CLI commands."""

from typer.testing import CliRunner

from agent_evals.cli.main import app

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
