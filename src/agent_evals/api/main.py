"""FastAPI application factory and bootstrap."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_evals.api.routes import router
from agent_evals.core.config import get_settings
from agent_evals.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI application lifecycle startup and shutdown handler."""
    settings = get_settings()
    configure_logging(log_level=settings.log_level, json_format=settings.log_json)
    logger.info(
        "Starting agent-evals API service",
        app_name=settings.app_name,
        env=settings.environment,
    )
    yield
    logger.info("Shutting down agent-evals API service")


def create_app() -> FastAPI:
    """Application factory for FastAPI service."""
    settings = get_settings()

    app = FastAPI(
        title="agent-evals API",
        description="REST API service for evaluating autonomous AI agents on single-cell biology tasks.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()
