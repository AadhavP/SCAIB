"""Async helper utilities."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")


async def run_with_timeout[T](coro: Awaitable[T], timeout_seconds: float) -> T:
    """Execute awaitable with a timeout constraint.

    Args:
        coro: The async awaitable to run.
        timeout_seconds: Max execution duration in seconds.

    Returns:
        Result of the coroutine.

    Raises:
        TimeoutError: If execution exceeds timeout.
    """
    return await asyncio.wait_for(coro, timeout=timeout_seconds)
