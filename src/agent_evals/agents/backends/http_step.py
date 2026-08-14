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

import asyncio
import hashlib
import ipaddress
import json
import math
import os
import socket
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import NAMESPACE_URL, uuid5

import httpx

from agent_evals.agents.decisions import (
    ExtractionMode,
    extract_action_response,
    response_evidence,
)
from agent_evals.agents.runtime.base import AgentRuntime
from agent_evals.agents.runtime.protocol import (
    PROTOCOL_VERSION,
    AgentAction,
    AgentContext,
    AgentManifest,
    AgentObservation,
    AgentPlan,
    AgentSession,
    AgentUsage,
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

#: A boundary response is an input to the extractor, not an unbounded log sink.
#: Rejecting an oversized reply before parsing keeps a faulty endpoint from
#: consuming the worker's memory or causing the archived trajectory to balloon.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

#: Observations are evaluator-owned input to an endpoint. Bound them too: a
#: malicious or misconfigured observation builder must not turn one remote POST
#: into an unbounded request-body allocation or an oversized persisted trace.
MAX_REQUEST_BYTES = 8 * 1024 * 1024

#: Bound the public transport audit as well as request/response bytes. A run
#: should already have a controller step budget, but this independent ceiling
#: prevents a misconfigured endpoint lifecycle from turning the audit into an
#: unbounded in-memory log.
MAX_EXCHANGE_RECORDS = 4096


class HttpStepError(RuntimeError):
    """Raised when the endpoint is unreachable or answers unusably."""


def validate_endpoint_url(url: str, *, allow_private: bool = False) -> str:
    """Validate an endpoint before it can reach the network.

    The endpoint is an operator-supplied destination, not agent data. Requiring a
    plain absolute HTTP(S) URL and keeping credentials out of its query string
    prevents the URL from becoming a second secret channel and makes error logs
    safe to persist. Authentication belongs in ``SCAIB_AGENT_TOKEN`` or explicit
    headers, both of which stay outside the public manifest.
    """
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
        # Accessing ``port`` is where urlsplit rejects values such as
        # ``:not-a-port``; validate it before the endpoint reaches httpx.
        port = parts.port
    except ValueError as error:
        raise HttpStepError(
            "http-step endpoint must contain a valid host and port"
        ) from error
    if parts.scheme not in {"http", "https"} or not parts.netloc or not hostname:
        raise HttpStepError("http-step endpoint must be an absolute http(s) URL")
    if port is not None and not 1 <= port <= 65535:
        raise HttpStepError("http-step endpoint port must be between 1 and 65535")
    if parts.username is not None or parts.password is not None:
        raise HttpStepError(
            "http-step endpoint must not embed credentials; use SCAIB_AGENT_TOKEN"
        )
    if parts.query or parts.fragment:
        raise HttpStepError(
            "http-step endpoint must not contain a query string or fragment; "
            "use SCAIB_AGENT_TOKEN or headers for authentication"
        )
    hostname_lower = hostname.lower().rstrip(".")
    private_names = {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata.goog",
    }
    try:
        address = ipaddress.ip_address(hostname_lower)
    except ValueError:
        address = None
    if not allow_private and (
        hostname_lower in private_names
        or (address is not None and not address.is_global)
    ):
        raise HttpStepError(
            "http-step endpoint resolves to a private or non-global address; "
            "enable the trusted local endpoint setting for integration tests"
        )
    if not allow_private:
        if parts.scheme == "http":
            raise HttpStepError(
                "http-step endpoints must use HTTPS outside trusted local testing"
            )
        _reject_private_dns_resolution(
            hostname_lower,
            port or (443 if parts.scheme == "https" else 80),
        )
    return url


def _reject_private_dns_resolution(hostname: str, port: int) -> None:
    """Reject DNS names that currently resolve into non-public address space.

    DNS can change after validation, so the worker also performs this check before
    each request. Unresolvable names are left to the transport layer so a
    temporary DNS outage is reported as an endpoint failure rather than turning
    submission validation into a hidden network dependency.
    """
    try:
        addresses = {
            ipaddress.ip_address(info[4][0])
            for info in socket.getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        }
    except (OSError, ValueError):
        return
    if any(not address.is_global for address in addresses):
        raise HttpStepError(
            "http-step endpoint DNS resolves to a private or non-global address"
        )


def public_endpoint(url: str) -> str:
    """Drop credentials and query string before an endpoint is published.

    The manifest is persisted into the run record, and endpoints in the wild
    carry tokens in ``?key=`` and in ``user:password@``. Recording where a run's
    agent lived should not record how to authenticate as it.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    try:
        port = parts.port
    except ValueError as error:
        raise HttpStepError("http-step endpoint must contain a valid port") from error
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port:
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _logical_request_id(payload: Mapping[str, Any]) -> str:
    """Derive a stable id for one session/type/step envelope.

    The endpoint can safely make its handler idempotent without trusting the
    request body as an identity: the session, lifecycle phase, and step are the
    benchmark's logical turn key. A changed action is a different step and gets a
    different id; a transport replay of the same turn gets the same id.
    """
    session_id = str(payload.get("session_id") or "no-session")
    phase = str(payload.get("type") or "unknown")
    step = str(payload.get("step") or 0)
    return str(uuid5(NAMESPACE_URL, f"scaib:{session_id}:{phase}:{step}"))


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
        strict_protocol: bool = False,
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
        api_settings = getattr(settings, "api", None)
        allow_private = bool(
            getattr(api_settings, "allow_private_agent_endpoints", False)
        )
        self.endpoint = validate_endpoint_url(resolved, allow_private=allow_private)
        self._allow_private_endpoints = allow_private
        self.api_key = api_key or settings.scaib_agent_token or os.getenv(TOKEN_VARIABLE)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise HttpStepError("http-step timeout_seconds must be finite and greater than zero")
        self.extra_headers = dict(headers or {})
        self.timeout_seconds = timeout_seconds
        self.strict_protocol = strict_protocol
        self.client = client
        self._owns_client = client is None
        #: Public, secret-free transport provenance. Payloads are represented by
        #: hashes and structural keys only; the raw observation/response remains
        #: governed by the trajectory and extraction evidence limits.
        self.exchange_log: list[dict[str, Any]] = []
        self.agent_id = agent_id
        self._last_session_id: str | None = None
        metadata: dict[str, Any] = {
            "endpoint": public_endpoint(self.endpoint),
            "strict_protocol": strict_protocol,
        }
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
        if not isinstance(reply, Mapping):
            raise HttpStepError(
                "http-step initialize reply must be a JSON object"
            )
        remote_state = reply.get("state")
        if isinstance(remote_state, dict):
            session.state["remote"] = remote_state
        usage = reply.get("usage")
        if isinstance(usage, dict):
            try:
                # The manager consumes this once; keeping it on the opaque
                # session avoids widening initialize() just for accounting.
                session.state["_boundary_usage"] = AgentUsage.model_validate(usage)
            except ValueError:
                # A malformed optional usage report must not turn a usable agent
                # response into a scientific action failure. The hard wall/step
                # budgets remain authoritative.
                session.state["_boundary_usage_error"] = "invalid initialize usage"
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
                "task": context.task_package,
                "observation": observation.model_dump(mode="json"),
            }
        )
        if not isinstance(reply, Mapping):
            return None
        payload = reply.get("plan")
        if not isinstance(payload, dict) or not payload:
            return None
        payload = dict(payload)
        if "usage" not in payload and isinstance(reply.get("usage"), dict):
            payload["usage"] = reply["usage"]
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
                "run_id": session.context.run_id,
                "benchmark_id": session.context.benchmark_id,
                "task_id": session.context.task_id,
                # These are repeated at the envelope level so a minimal endpoint
                # can route the turn without understanding the entire observation
                # model. They are still the same evaluator-owned values embedded
                # in the canonical observation below.
                "task": session.context.task_package,
                "available_actions": list(observation.available_actions),
                "previous_decision": observation.previous_decision,
                "state_delta": observation.state_delta,
                "observation": observation.model_dump(mode="json"),
            }
        )
        try:
            payload, evidence = extract_action_response(
                reply,
                available_actions=observation.available_actions,
            )
            payload["extraction_evidence"] = evidence
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
        if isinstance(reply, str):
            return FinalSubmission(
                summary=reply[:2_000],
                extraction_evidence=response_evidence(
                    reply,
                    mode=ExtractionMode.FREE_TEXT,
                    source="agent_boundary_submission",
                    extracted_fields=("summary",),
                ),
            )
        if not isinstance(reply, Mapping):
            raise HttpStepError("http-step terminate reply must be a JSON object or text")
        nested = reply.get("submission")
        payload = dict(nested) if isinstance(nested, dict) else dict(reply)
        # Correlation is transport metadata, not a FinalSubmission field. The
        # endpoint may echo it on every lifecycle response, and the boundary
        # verifier has already checked it above; passing it into the strict
        # scientific submission model would turn a valid echoed response into a
        # false termination failure.
        payload.pop("request_id", None)
        payload.pop("protocol_version", None)
        if not payload:
            return FinalSubmission(
                extraction_evidence=response_evidence(
                    reply,
                    mode=ExtractionMode.STRUCTURED,
                    source="agent_boundary_submission",
                )
            )
        try:
            submission = FinalSubmission.model_validate(payload)
            return submission.model_copy(
                update={
                    "extraction_evidence": response_evidence(
                        reply,
                        mode=ExtractionMode.STRUCTURED,
                        source="agent_boundary_submission",
                        extracted_fields=tuple(
                            key
                            for key in ("output_artifacts", "metadata", "summary", "explanation", "usage")
                            if key in payload
                        ),
                    )
                }
            )
        except ValueError as error:
            raise HttpStepError(
                f"http-step reply is not a final submission: {error}"
            ) from error

    def _request_headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _post(self, payload: dict[str, Any]) -> Any:  # noqa: C901
        """POST one envelope and return decoded JSON without narrowing text replies.

        Every logical turn carries a protocol version and a stable correlation id.
        The endpoint may log those public values and deduplicate a transport-level
        replay without SCAIB retrying a scientific turn.
        """
        if len(self.exchange_log) >= MAX_EXCHANGE_RECORDS:
            raise HttpStepError(
                f"http-step exchange audit exceeded the {MAX_EXCHANGE_RECORDS}-record limit"
            )
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                # The evaluator must not inherit HTTP(S)_PROXY or NO_PROXY
                # from the host. A proxy can otherwise bypass the endpoint DNS
                # check and receive bearer-authenticated scientific observations.
                trust_env=False,
            )
            self._owns_client = True
        if not self._allow_private_endpoints:
            parts = urlsplit(self.endpoint)
            _reject_private_dns_resolution(
                parts.hostname or "",
                parts.port or (443 if parts.scheme == "https" else 80),
            )
        envelope = {
            **payload,
            "protocol_version": PROTOCOL_VERSION,
            # A logical turn keeps one id across transport-level replay. SCAIB
            # itself never retries a scientific POST, but a proxy or endpoint
            # may need this key to deduplicate a request whose response was lost.
            # Protocol metadata is written last so a caller cannot spoof the
            # correlation identity through an action parameter or test double.
            "request_id": _logical_request_id(payload),
        }
        request_bytes = len(
            json.dumps(
                envelope,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        if request_bytes > MAX_REQUEST_BYTES:
            raise HttpStepError(
                f"http-step request to '{payload['type']}' exceeded the "
                f"{MAX_REQUEST_BYTES}-byte request limit"
            )
        if isinstance(self.client, httpx.AsyncClient):
            return await self._post_streaming(envelope, payload)
        try:
            response = await asyncio.wait_for(
                self.client.post(
                    self.endpoint, json=envelope, headers=self._request_headers()
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as error:
            raise HttpStepError(
                f"http-step POST to {self.endpoint} timed out after "
                f"{self.timeout_seconds:g}s"
            ) from error
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
        raw_content = getattr(response, "content", None)
        if isinstance(raw_content, (bytes, bytearray)) and len(raw_content) > MAX_RESPONSE_BYTES:
            raise HttpStepError(
                f"http-step reply to '{payload['type']}' exceeded the "
                f"{MAX_RESPONSE_BYTES}-byte response limit"
            )
        try:
            decoded = response.json()
        except Exception as error:
            # Level-0 black-box agents may answer with plain text. Keep that
            # response bounded and let the decision extractor decide whether the
            # text explicitly names an action; transport decoding must not turn a
            # valid free-form boundary response into a protocol failure.
            text = getattr(response, "text", "")
            if not isinstance(text, str) or not text.strip():
                raise HttpStepError(
                    f"http-step reply to '{payload['type']}' is not JSON: {error}"
                ) from error
            if len(text.encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise HttpStepError(
                    f"http-step reply to '{payload['type']}' exceeded the "
                    f"{MAX_RESPONSE_BYTES}-byte response limit"
                ) from error
            decoded = text
        # Record a bounded response digest before validating protocol metadata, so
        # a malformed endpoint reply remains auditable rather than disappearing
        # from the exchange ledger with only a raised exception.
        self._record_exchange(envelope, decoded, status=status, raw_bytes=raw_content)
        self._validate_protocol_reply(payload, envelope, decoded)
        return decoded

    async def _post_streaming(
        self,
        envelope: dict[str, Any],
        payload: dict[str, Any],
    ) -> Any:
        """Read real HTTP responses with a hard byte ceiling.

        ``AsyncClient.post`` buffers the complete response before returning. A
        remote endpoint is untrusted input, so production clients use streaming
        reads and abort before a giant response can enter the worker heap. Small
        injected test clients intentionally use the compatibility path above.
        """
        assert isinstance(self.client, httpx.AsyncClient)
        try:
            async with self.client.stream(
                "POST",
                self.endpoint,
                json=envelope,
                headers=self._request_headers(),
            ) as response:
                status = int(response.status_code)
                if status < 200 or status >= 300:
                    body = await _read_stream(
                        response.aiter_bytes(), MAX_ERROR_BODY_CHARS
                    )
                    text = body.decode("utf-8", errors="replace")
                    raise HttpStepError(
                        f"http-step endpoint answered {status} for "
                        f"'{payload['type']}': {text}"
                    )
                declared_length = response.headers.get("content-length")
                if declared_length is not None:
                    try:
                        if int(declared_length) > MAX_RESPONSE_BYTES:
                            raise HttpStepError(
                                f"http-step reply to '{payload['type']}' exceeded the "
                                f"{MAX_RESPONSE_BYTES}-byte response limit"
                            )
                    except ValueError:
                        # An invalid content-length is treated like a chunked
                        # response; the streaming byte ceiling remains authoritative.
                        pass
                content = await _read_stream(
                    response.aiter_bytes(), MAX_RESPONSE_BYTES
                )
        except HttpStepError:
            raise
        except Exception as error:
            raise HttpStepError(
                f"http-step POST to {self.endpoint} failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        try:
            decoded = json.loads(content)
        except Exception as error:
            text = content.decode("utf-8", errors="replace")
            if not text.strip():
                raise HttpStepError(
                    f"http-step reply to '{payload['type']}' is not JSON: {error}"
                ) from error
            decoded = text
        self._record_exchange(envelope, decoded, status=status, raw_bytes=content)
        self._validate_protocol_reply(payload, envelope, decoded)
        return decoded

    def _validate_protocol_reply(
        self,
        payload: Mapping[str, Any],
        envelope: Mapping[str, Any],
        decoded: Any,
    ) -> None:
        """Enforce lifecycle correlation when strict protocol mode is enabled.

        Level-0 black-box mode intentionally accepts text, but strict conformance
        mode must not silently turn a list, scalar, or plain-text response into a
        declined plan or an extractor-specific error. The protocol envelope is a
        JSON object in every lifecycle phase, so strict mode rejects any other
        shape before the manager can interpret it as scientific behavior.
        """
        phase = str(payload.get("type", "unknown"))
        if self.strict_protocol and not isinstance(decoded, Mapping):
            raise HttpStepError(
                f"http-step reply to '{phase}' must be a JSON object in strict protocol mode"
            )
        if not isinstance(decoded, Mapping):
            return
        returned_protocol = decoded.get("protocol_version")
        if returned_protocol is not None and returned_protocol != PROTOCOL_VERSION:
            raise HttpStepError(
                f"http-step reply to '{phase}' used unsupported "
                f"protocol_version {returned_protocol!r}"
            )
        returned_request_id = decoded.get("request_id")
        if self.strict_protocol and returned_request_id is None:
            raise HttpStepError(
                f"http-step reply to '{phase}' omitted the required request_id"
            )
        if self.strict_protocol and returned_protocol is None:
            raise HttpStepError(
                f"http-step reply to '{phase}' omitted the required protocol_version"
            )
        if returned_request_id is not None and returned_request_id != envelope["request_id"]:
            raise HttpStepError(
                f"http-step reply to '{phase}' carried a mismatched request_id"
            )

    def _record_exchange(
        self,
        envelope: Mapping[str, Any],
        response: Any,
        *,
        status: int,
        raw_bytes: bytes | bytearray | None,
    ) -> None:
        """Record bounded, secret-free transport provenance for one exchange."""
        if len(self.exchange_log) >= MAX_EXCHANGE_RECORDS:
            raise HttpStepError(
                f"http-step exchange audit exceeded the {MAX_EXCHANGE_RECORDS}-record limit"
            )
        if isinstance(raw_bytes, (bytes, bytearray)):
            response_size = len(raw_bytes)
        else:
            encoded = json.dumps(
                response,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            response_size = len(encoded)
        response_digest = hashlib.sha256(
            bytes(raw_bytes)
            if isinstance(raw_bytes, (bytes, bytearray))
            else json.dumps(
                response,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        self.exchange_log.append(
            {
                "phase": str(envelope.get("type", "unknown")),
                "step": envelope.get("step"),
                "request_id": envelope.get("request_id"),
                "protocol_version": envelope.get("protocol_version"),
                "status_code": status,
                "request_sha256": _sha256_json(envelope),
                "response_sha256": response_digest,
                "response_bytes": response_size,
                "response_keys": sorted(response) if isinstance(response, Mapping) else [],
            }
        )

    async def close(self) -> None:
        """Close an owned HTTP client even when initialization never completed."""
        await self._close()

    async def _close(self) -> None:
        if self.client is not None and self._owns_client:
            closer = getattr(self.client, "aclose", None)
            if closer is not None:
                await closer()
            self.client = None


def _sha256_json(value: Any) -> str:
    """Hash a JSON-compatible transport envelope without retaining its body."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def _read_stream(stream: Any, maximum_bytes: int) -> bytes:
    """Collect an async byte stream without exceeding ``maximum_bytes``."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in stream:
        if not isinstance(chunk, bytes):
            chunk = bytes(chunk)
        total += len(chunk)
        if total > maximum_bytes:
            raise HttpStepError(
                f"http-step reply exceeded the {maximum_bytes}-byte response limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


__all__ = [
    "DEFAULT_STEP_TIMEOUT_SECONDS",
    "ENDPOINT_VARIABLE",
    "MAX_ERROR_BODY_CHARS",
    "MAX_EXCHANGE_RECORDS",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "TOKEN_VARIABLE",
    "HttpStepError",
    "HttpStepRuntime",
    "public_endpoint",
    "validate_endpoint_url",
]
