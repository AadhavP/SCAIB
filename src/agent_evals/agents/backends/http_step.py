"""A black-box agent reached over one ``POST /step`` endpoint.

This is the benchmark's lowest integration tier: the agent is a URL. It does not
import SCAIB, does not speak Python, and is never asked to run in this process.
Every turn is one POST of a JSON envelope and one JSON reply, so an agent written
in any language can be scored by standing up a single route.

The contract, which the endpoint must answer for all four envelope types::

    POST <endpoint>
    {"type": "initialize", "session_id": "...", "step": 0, "context": {...}}
        -> {} | {"state": {...}}

    {"type": "plan", "session_id": "...", "step": 0,
     "context": {...}, "observation": {...}}
        -> {} | {"plan": {"goal": "...", "steps": [...], ...}}

    {"type": "observation", "session_id": "...", "step": N, "observation": {...}}
        -> {"action_type": "...", "parameters": {...}} | {"action": {...}}

    {"type": "terminate", "session_id": "...", "step": N, "observation": {...}|null}
        -> {} | {"submission": {...}} | {"output_artifacts": [...], ...}

Three deliberate properties:

- **No retries.** A step is not idempotent -- the agent's own code may already
  have written files by the time a reply is lost -- so a retry could double-run a
  scientific operation and record it once. A failed POST raises, and the harness
  records the failure against the run rather than papering over it.
- **A missing plan is declined, not inferred.** An empty reply to the ``plan``
  envelope means "this agent does not plan", and a transport failure raises. The
  manager already distinguishes the two: a raise is recorded as
  ``benchmark_fallback: HttpStepError`` beside the substituted plan, so a broken
  endpoint cannot be mistaken for a non-planning agent.
- **The reply is evidence, never state.** An action and a submission are what the
  harness verifies against observation; nothing in a reply is allowed to set
  benchmark state directly.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.protocol import (
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentObservation,
    AgentPlan,
    AgentSession,
    FinalSubmission,
)
from agent_evals.core.config import get_settings

#: Read when no endpoint is passed, so ``--agent http-step`` works without a
#: bespoke config file. Declared in :mod:`agent_evals.core.config` too, which is
#: the project's configuration boundary.
ENDPOINT_VARIABLE = "SCAIB_AGENT_ENDPOINT"

#: Bearer token variable. Never defaulted, never logged, never echoed back.
TOKEN_VARIABLE = "SCAIB_AGENT_TOKEN"

DEFAULT_STEP_TIMEOUT_SECONDS = 300.0

#: Error bodies are attached to the raised exception, so they are bounded.
MAX_ERROR_BODY_CHARS = 2000


class HttpStepError(RuntimeError):
    """Raised when the endpoint is unreachable or answers unusably."""


def public_endpoint(url: str) -> str:
    """Drop credentials and query string before an endpoint is published.

    The manifest is persisted into the run record, and endpoints in the wild
    carry tokens in ``?key=`` and in ``user:password@``. Recording where a run's
    agent lived should not record how to authenticate as it.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


class HttpStepRuntime(AgentRuntime):
    """Drive a remote agent through a single JSON step endpoint."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = DEFAULT_STEP_TIMEOUT_SECONDS,
        client: Any | None = None,
        agent_id: str = "http-step",
        model: str | None = None,
    ) -> None:
        # Settings before os.getenv for the same reason the credential resolver
        # does it: pydantic-settings parses .env, which os.getenv cannot see.
        settings = get_settings()
        resolved = (
            endpoint or settings.scaib_agent_endpoint or os.getenv(ENDPOINT_VARIABLE)
        )
        if not resolved:
            raise HttpStepError(
                "an http-step agent needs an endpoint URL, passed as 'endpoint' "
                f"or set in {ENDPOINT_VARIABLE}"
            )
        self.endpoint = resolved
        self.api_key = api_key or settings.scaib_agent_token or os.getenv(TOKEN_VARIABLE)
        self.extra_headers = dict(headers or {})
        self.timeout_seconds = timeout_seconds
        self.client = client
        self._owns_client = client is None
        self.agent_id = agent_id
        self._last_session_id: str | None = None
        metadata: dict[str, Any] = {"endpoint": public_endpoint(self.endpoint)}
        if model:
            metadata["model"] = model
        self.manifest = AgentManifest(
            name="HTTP step agent",
            type="http_step",
            capabilities=["http_step", "structured_actions"],
            metadata=metadata,
        )

    async def initialize(self, context: AgentContext) -> AgentSession:
        """Announce the episode and keep whatever opaque state is returned."""
        session = AgentSession(context=context)
        self._last_session_id = session.session_id
        reply = await self._post(
            {
                "type": "initialize",
                "session_id": session.session_id,
                "step": 0,
                "context": context.model_dump(mode="json"),
            }
        )
        remote_state = reply.get("state")
        if isinstance(remote_state, dict):
            session.state["remote"] = remote_state
        return session

    async def plan(
        self, context: AgentContext, observation: AgentObservation
    ) -> AgentPlan | None:
        """Ask for a plan, and accept a decline without inventing one."""
        reply = await self._post(
            {
                "type": "plan",
                "session_id": self._last_session_id,
                "step": 0,
                "context": context.model_dump(mode="json"),
                "observation": observation.model_dump(mode="json"),
            }
        )
        payload = reply.get("plan")
        if not isinstance(payload, dict) or not payload:
            return None
        return AgentPlan.model_validate(payload)

    async def act(
        self, session: AgentSession, observation: AgentObservation
    ) -> AgentAction:
        """Exchange one observation for exactly one action."""
        step = int(session.state.get("step", 0)) + 1
        session.state["step"] = step
        reply = await self._post(
            {
                "type": "observation",
                "session_id": session.session_id,
                "step": step,
                "observation": observation.model_dump(mode="json"),
            }
        )
        nested = reply.get("action")
        payload = nested if isinstance(nested, dict) else reply
        try:
            return AgentAction.model_validate(payload)
        except ValueError as error:
            raise HttpStepError(
                f"http-step reply is not an agent action: {error}"
            ) from error

    async def terminate(
        self,
        session: AgentSession,
        observation: AgentObservation | None = None,
    ) -> FinalSubmission:
        """Collect the closing submission and release an owned client."""
        try:
            reply = await self._post(
                {
                    "type": "terminate",
                    "session_id": session.session_id,
                    "step": int(session.state.get("step", 0)),
                    "observation": (
                        observation.model_dump(mode="json")
                        if observation is not None
                        else None
                    ),
                }
            )
        finally:
            await self._close()
        nested = reply.get("submission")
        payload = nested if isinstance(nested, dict) else reply
        if not payload:
            return FinalSubmission()
        try:
            return FinalSubmission.model_validate(payload)
        except ValueError as error:
            raise HttpStepError(
                f"http-step reply is not a final submission: {error}"
            ) from error

    def _request_headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST one envelope and return the decoded object reply."""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout_seconds)
            self._owns_client = True
        try:
            response = await self.client.post(
                self.endpoint, json=payload, headers=self._request_headers()
            )
        except Exception as error:  # httpx raises a family, not one class
            raise HttpStepError(
                f"http-step POST to {self.endpoint} failed: {type(error).__name__}: {error}"
            ) from error
        status = int(getattr(response, "status_code", 0))
        if status < 200 or status >= 300:
            body = str(getattr(response, "text", ""))[:MAX_ERROR_BODY_CHARS]
            raise HttpStepError(
                f"http-step endpoint answered {status} for "
                f"'{payload['type']}': {body}"
            )
        try:
            decoded = response.json()
        except Exception as error:
            raise HttpStepError(
                f"http-step reply to '{payload['type']}' is not JSON: {error}"
            ) from error
        if not isinstance(decoded, dict):
            raise HttpStepError(
                f"http-step reply to '{payload['type']}' must be a JSON object, "
                f"got {type(decoded).__name__}"
            )
        return decoded

    async def _close(self) -> None:
        if self.client is not None and self._owns_client:
            closer = getattr(self.client, "aclose", None)
            if closer is not None:
                await closer()
            self.client = None


__all__ = [
    "DEFAULT_STEP_TIMEOUT_SECONDS",
    "ENDPOINT_VARIABLE",
    "MAX_ERROR_BODY_CHARS",
    "TOKEN_VARIABLE",
    "HttpStepError",
    "HttpStepRuntime",
    "public_endpoint",
]
