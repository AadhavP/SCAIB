"""Tests for the provider runtime's asynchronous transport boundary."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from agent_evals.agents.backends.openai import _call_openai


@pytest.mark.asyncio
async def test_sync_provider_call_does_not_block_event_loop() -> None:
    """A synchronous SDK call must run off the API event loop."""
    class Completions:
        @staticmethod
        def create(**kwargs: object) -> object:
            del kwargs
            time.sleep(0.06)
            return SimpleNamespace(ok=True)

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        for _ in range(4):
            await asyncio.sleep(0.01)
            ticks += 1

    result, _ = await asyncio.gather(
        _call_openai(client, "glm-test", [], []),
        heartbeat(),
    )

    assert result.ok is True
    assert ticks == 4
