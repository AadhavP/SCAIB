"""Production boundary tests for the universal HTTP agent endpoint."""

from __future__ import annotations

import asyncio

import pytest

from agent_evals.agents.backends.http_step import (
    MAX_REQUEST_BYTES,
    HttpStepError,
    HttpStepRuntime,
)


class _NeverCalledClient:
    """Client double proving oversized requests fail before transport."""

    def __init__(self) -> None:
        self.called = False

    async def post(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.called = True
        raise AssertionError("oversized endpoint request reached the transport")


@pytest.mark.asyncio
async def test_http_endpoint_rejects_oversized_requests_before_transport() -> None:
    client = _NeverCalledClient()
    runtime = HttpStepRuntime(
        endpoint="https://agent.example/step",
        client=client,
    )

    with pytest.raises(HttpStepError, match="request limit"):
        await runtime._post(
            {
                "type": "observation",
                "observation": {"large": "x" * (MAX_REQUEST_BYTES + 1)},
            }
        )

    assert client.called is False


@pytest.mark.asyncio
async def test_http_endpoint_rejects_an_unsupported_response_protocol_version() -> None:
    class Client:
        async def post(
            self,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> object:
            del url, json, headers
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "content": b'{"protocol_version":"0.0"}',
                    "text": '{"protocol_version":"0.0"}',
                    "json": lambda self: {"protocol_version": "0.0"},
                },
            )()

    runtime = HttpStepRuntime(endpoint="https://agent.example/step", client=Client())

    with pytest.raises(HttpStepError, match="unsupported protocol_version"):
        await runtime._post({"type": "observation"})


def test_http_endpoint_rejects_non_finite_or_non_positive_timeouts() -> None:
    for timeout in (0, -1, float("nan"), float("inf")):
        with pytest.raises(HttpStepError, match="timeout_seconds"):
            HttpStepRuntime(
                endpoint="https://agent.example/step",
                timeout_seconds=timeout,
            )


@pytest.mark.asyncio
async def test_strict_http_protocol_requires_response_correlation() -> None:
    class Client:
        async def post(
            self,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> object:
            del url, json, headers
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "content": b'{"protocol_version":"1.0"}',
                    "text": '{"protocol_version":"1.0"}',
                    "json": lambda self: {"protocol_version": "1.0"},
                },
            )()

    runtime = HttpStepRuntime(
        endpoint="https://agent.example/step",
        client=Client(),
        strict_protocol=True,
    )

    with pytest.raises(HttpStepError, match="omitted the required request_id"):
        await runtime._post({"type": "observation"})


@pytest.mark.asyncio
async def test_strict_http_protocol_rejects_non_object_lifecycle_replies() -> None:
    class Client:
        async def post(
            self,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> object:
            del url, json, headers
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "content": b"[]",
                    "text": "[]",
                    "json": lambda self: [],
                },
            )()

    runtime = HttpStepRuntime(
        endpoint="https://agent.example/step",
        client=Client(),
        strict_protocol=True,
    )

    with pytest.raises(HttpStepError, match="must be a JSON object"):
        await runtime._post({"type": "plan"})


@pytest.mark.asyncio
async def test_custom_http_client_is_bound_by_the_endpoint_timeout() -> None:
    class SlowClient:
        async def post(
            self,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> object:
            del url, json, headers
            await asyncio.sleep(0.05)
            return object()

    runtime = HttpStepRuntime(
        endpoint="https://agent.example/step",
        client=SlowClient(),
        timeout_seconds=0.001,
    )

    with pytest.raises(HttpStepError, match="timed out"):
        await runtime._post({"type": "observation"})


@pytest.mark.asyncio
async def test_http_endpoint_accepts_a_small_request_and_preserves_request_id() -> None:
    class Client:
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        async def post(
            self,
            url: str,
            *,
            json: dict[str, object],
            headers: dict[str, str],
        ) -> object:
            del url, headers
            self.payload = json
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "content": b"{}",
                    "text": "{}",
                    "json": lambda self: {"request_id": json["request_id"]},
                },
            )()

    client = Client()
    runtime = HttpStepRuntime(endpoint="https://agent.example/step", client=client)

    reply = await runtime._post({"type": "observation", "step": 1})

    assert reply["request_id"] == client.payload["request_id"]  # type: ignore[index]
    assert client.payload["protocol_version"] == "1.0"  # type: ignore[index]
    assert len(runtime.exchange_log) == 1
    exchange = runtime.exchange_log[0]
    assert exchange["phase"] == "observation"
    assert exchange["request_id"] == client.payload["request_id"]  # type: ignore[index]
    assert len(exchange["request_sha256"]) == 64
    assert len(exchange["response_sha256"]) == 64
