"""FastAPI application factory and bootstrap."""

import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from structlog.contextvars import bound_contextvars

from agent_evals import __version__
from agent_evals.api.routes import job_manager, public_router, router
from agent_evals.core.config import get_settings
from agent_evals.core.logging import configure_logging, get_logger

logger = get_logger(__name__)
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
MAX_REQUEST_BODY_BYTES = 64 * 1024
_API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._~-]{16,256}$")


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Bound API request bodies before Pydantic parses untrusted JSON."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method in {"POST", "PUT", "PATCH"}:
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError:
                    return JSONResponse(
                        {"detail": "content-length must be an integer"},
                        status_code=400,
                    )
                if declared < 0:
                    return JSONResponse(
                        {"detail": "content-length must not be negative"},
                        status_code=400,
                    )
                if declared > MAX_REQUEST_BODY_BYTES:
                    return JSONResponse(
                        {"detail": "request body exceeds the 64 KiB limit"},
                        status_code=413,
                    )
            body_parts: list[bytes] = []
            body_size = 0
            async for chunk in request.stream():
                body_size += len(chunk)
                if body_size > MAX_REQUEST_BODY_BYTES:
                    return JSONResponse(
                        {"detail": "request body exceeds the 64 KiB limit"},
                        status_code=413,
                    )
                body_parts.append(chunk)
            body = b"".join(body_parts)
            # ``request.stream()`` marks the Starlette request consumed. Cache the
            # bounded bytes so downstream JSON parsing sees the same body instead
            # of an empty stream/422 after this middleware has inspected it.
            request._body = body
            sent = False

            async def receive() -> dict[str, object]:
                nonlocal sent
                if sent:
                    return {"type": "http.request", "body": b"", "more_body": False}
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}

            # Starlette's request body parser reads from this callable after the
            # middleware has inspected the bytes. Replaying the body avoids
            # consuming it before FastAPI validation.
            request._receive = receive
        return await call_next(request)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a bounded correlation ID to logs and every API response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _REQUEST_ID.fullmatch(supplied) else str(uuid4())
        with bound_contextvars(request_id=request_id):
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI application lifecycle startup and shutdown handler."""
    settings = get_settings()
    configure_logging(log_level=settings.log_level, json_format=settings.log_json)
    if settings.api.execute_jobs_in_process and settings.api.workers > 1:
        raise RuntimeError(
            "in-process evaluation execution requires one API worker; use "
            "AGENT_EVALS_API__EXECUTE_JOBS_IN_PROCESS=false with a dedicated "
            "agent-evals worker when scaling the API"
        )
    if settings.environment.lower() not in {"development", "testing", "local"}:
        if not settings.api.api_key or not _API_KEY_PATTERN.fullmatch(
            settings.api.api_key
        ):
            raise RuntimeError(
                "production-like API startup requires an "
                "AGENT_EVALS_API__API_KEY containing 16-256 URL-safe characters"
            )
        if "*" in settings.api.cors_origins:
            raise RuntimeError(
                "wildcard CORS is not allowed in production-like environments"
            )
    logger.info(
        "Starting agent-evals API service",
        app_name=settings.app_name,
        env=settings.environment,
        job_store=str(settings.storage.job_db_path),
    )
    await job_manager.start(
        execute_jobs=settings.api.execute_jobs_in_process
    )
    try:
        yield
    finally:
        await job_manager.shutdown()
        logger.info("Shutting down agent-evals API service")


def create_app() -> FastAPI:
    """Application factory for FastAPI service."""
    settings = get_settings()

    production_like = settings.environment.lower() not in {
        "development",
        "testing",
        "local",
    }
    app = FastAPI(
        title="agent-evals API",
        description="REST API service for evaluating autonomous AI agents on single-cell biology tasks.",
        version=__version__,
        lifespan=lifespan,
        # Interactive API documentation is useful locally but should not be an
        # unauthenticated discovery surface on a production control plane.
        docs_url=None if production_like else "/docs",
        redoc_url=None if production_like else "/redoc",
        openapi_url=None if production_like else "/openapi.json",
    )

    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Idempotency-Key",
            "X-Request-ID",
        ],
    )

    app.include_router(public_router)
    app.include_router(router)

    return app


app = create_app()
