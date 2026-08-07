"""JSON-lines external process runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.protocol import (
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentObservation,
    AgentSession,
    FinalSubmission,
)


class ExternalProcessRuntime(AgentRuntime):
    """Communicate with Python/JS/Rust agents using stdin/stdout JSON lines."""

    def __init__(self, command: Sequence[str], *, agent_id: str = "external-process") -> None:
        self.command = list(command)
        self.agent_id = agent_id
        self.manifest = AgentManifest(
            name=agent_id,
            type="external_process",
            capabilities=["external_process", "structured_actions"],
        )

    async def initialize(self, context: AgentContext) -> AgentSession:
        process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        session = AgentSession(context=context, state={"process": process})
        await self._send(session, {"type": "initialize", "context": context.model_dump(mode="json")})
        return session

    async def act(self, session: AgentSession, observation: AgentObservation) -> AgentAction:
        await self._send(session, {"type": "observation", "observation": observation.model_dump(mode="json")})
        response = await self._receive(session)
        return AgentAction.model_validate(response)

    async def terminate(
        self,
        session: AgentSession,
        observation: AgentObservation | None = None,
    ) -> FinalSubmission:
        del observation
        process = session.state.get("process")
        if process is None:
            return FinalSubmission()
        await self._send(session, {"type": "terminate"})
        if process.stdin is not None:
            process.stdin.close()
        await process.wait()
        return FinalSubmission()

    async def _send(self, session: AgentSession, payload: dict[str, Any]) -> None:
        process = session.state["process"]
        if process.stdin is None:
            raise RuntimeError("external agent stdin is unavailable")
        process.stdin.write((json.dumps(payload) + "\n").encode())
        await process.stdin.drain()

    async def _receive(self, session: AgentSession) -> dict[str, Any]:
        process = session.state["process"]
        if process.stdout is None:
            raise RuntimeError("external agent stdout is unavailable")
        line = await process.stdout.readline()
        if not line:
            raise RuntimeError("external agent exited without an action")
        payload = json.loads(line.decode())
        if not isinstance(payload, dict):
            raise ValueError("external agent response must be a JSON object")
        return payload


__all__ = ["ExternalProcessRuntime"]
