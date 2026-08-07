"""Tests for benchmark and agent registries."""

from typing import Any

import pytest

from agent_evals.agents.base import AgentObservation, BaseAgentAdapter
from agent_evals.agents.registry import AgentRegistry
from agent_evals.benchmarks.base import BaseBenchmark
from agent_evals.benchmarks.registry import BenchmarkRegistry
from agent_evals.core.exceptions import RegistryError
from agent_evals.core.types import EvaluationResult, StatusEnum


class DummyBenchmark(BaseBenchmark):
    async def prepare(self, config: dict[str, Any]) -> None:
        pass

    async def evaluate_agent(
        self, agent_adapter: Any, sandbox: Any
    ) -> EvaluationResult:
        return EvaluationResult(
            benchmark_id=self.metadata.id,
            agent_id="dummy_agent",
            status=StatusEnum.COMPLETED,
        )

    async def cleanup(self) -> None:
        pass


class DummyAgent(BaseAgentAdapter):
    async def reset(self) -> None:
        pass

    async def step(self, observation: AgentObservation) -> str:
        return "print('hello')"


def test_benchmark_registry() -> None:
    registry = BenchmarkRegistry()

    @registry.register("dummy_bm")
    class TestBenchmark(DummyBenchmark):
        pass

    assert "dummy_bm" in registry.list_ids()
    cls = registry.get("dummy_bm")
    assert cls == TestBenchmark

    with pytest.raises(RegistryError):
        registry.get("non_existent")


def test_agent_registry() -> None:
    registry = AgentRegistry()

    @registry.register("dummy_agent_type")
    class TestAgent(DummyAgent):
        pass

    assert "dummy_agent_type" in registry.list_types()
    cls = registry.get("dummy_agent_type")
    assert cls == TestAgent

    with pytest.raises(RegistryError):
        registry.get("unknown_agent")
