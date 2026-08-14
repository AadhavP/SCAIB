"""Pydantic Settings configuration system with YAML file loader support."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_evals.core.exceptions import ConfigurationError


def _env_shadowed_paths() -> set[tuple[str, ...]]:
    """Return dotted paths supplied through ``AGENT_EVALS_`` environment variables.

    Environment variables must outrank YAML configuration, so any YAML key that
    is also configured through the environment is removed before validation.
    """
    paths: set[tuple[str, ...]] = set()
    for name, value in os.environ.items():
        if name.startswith("AGENT_EVALS_") and value != "":
            paths.add(
                tuple(part.lower() for part in name[len("AGENT_EVALS_") :].split("__"))
            )
    return paths


def _strip_env_shadowed(
    data: dict[str, Any],
    paths: set[tuple[str, ...]],
    prefix: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Remove YAML entries whose path is overridden by an environment variable."""
    return {
        key: value
        for key, value in (
            (
                child_key,
                (
                    _strip_env_shadowed(child_value, paths, (*prefix, child_key.lower()))
                    if isinstance(child_value, dict)
                    else child_value
                ),
            )
            for child_key, child_value in data.items()
        )
        if (*prefix, key.lower()) not in paths
    }


class APISettings(BaseModel):
    """API server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = Field(default=1, ge=1, le=32)
    # Development can execute jobs in API BackgroundTasks. Production should
    # set this false and run the dedicated ``agent-evals worker`` service so API
    # restarts do not interrupt scientific work and web workers stay responsive.
    execute_jobs_in_process: bool = True
    api_key: str | None = Field(default=None, repr=False)
    # Remote agent URLs are server-side network destinations. Private address
    # space is disabled by default to prevent an API caller from turning SCAIB
    # into an SSRF proxy; local integration tests can opt in explicitly.
    allow_private_agent_endpoints: bool = False
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:8000"]
    )


class StorageSettings(BaseModel):
    """Storage directory settings."""

    data_dir: Path = Path("./data")
    reports_dir: Path = Path("./reports_output")
    cache_dir: Path = Path("./.cache")
    # SQLite is the default durable control-plane store. Production deployments
    # should place this file on persistent storage; ``:memory:`` remains useful
    # for isolated unit tests through the explicit manager constructor.
    job_db_path: Path = Path("./data/evaluation_jobs.sqlite3")


class SandboxSettings(BaseModel):
    """Execution sandbox settings."""

    timeout_seconds: int = 300
    max_memory_gb: int = 16
    allowed_imports: list[str] = Field(
        default_factory=lambda: [
            "scanpy",
            "anndata",
            "numpy",
            "pandas",
            "scipy",
        ]
    )


class Settings(BaseSettings):
    """Main Application Settings loaded from environment variables and YAML files."""

    app_name: str = "agent-evals"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    log_json: bool = False
    llm_model: str | None = Field(default=None, validation_alias="LLM_MODEL")
    llm_api_key: str | None = Field(default=None, validation_alias="LLM_API_KEY", repr=False)
    llm_base_url: str | None = Field(default=None, validation_alias="LLM_BASE_URL")
    # Provider-specific aliases are loaded from .env too; os.getenv() cannot see
    # values parsed by pydantic-settings from the dotenv file.
    glm_model: str | None = Field(default=None, validation_alias="GLM_MODEL")
    glm_api_key: str | None = Field(default=None, validation_alias="GLM_API_KEY", repr=False)
    glm_base_url: str | None = Field(default=None, validation_alias="GLM_BASE_URL")
    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY", repr=False)
    openrouter_base_url: str | None = Field(default=None, validation_alias="OPENROUTER_BASE_URL")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY", repr=False)
    # Black-box agent boundary: a URL that answers POST /step, plus its bearer
    # token. Declared here so the endpoint is configuration rather than a flag
    # someone has to remember on every invocation.
    scaib_agent_endpoint: str | None = Field(default=None, validation_alias="SCAIB_AGENT_ENDPOINT")
    scaib_agent_token: str | None = Field(default=None, validation_alias="SCAIB_AGENT_TOKEN", repr=False)

    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)

    model_config = SettingsConfigDict(
        env_prefix="AGENT_EVALS_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "Settings":
        """Load settings from a YAML file, merging with default settings.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Instantiated Settings object.
        """
        if not yaml_path.exists():
            raise ConfigurationError(f"Configuration file not found: {yaml_path}")

        try:
            with open(yaml_path, encoding="utf-8") as f:
                data: dict[str, Any] | None = yaml.safe_load(f)
            if data is None:
                data = {}
            return cls(**_strip_env_shadowed(data, _env_shadowed_paths()))
        except Exception as err:
            raise ConfigurationError(
                f"Failed to parse configuration YAML {yaml_path}: {err}"
            ) from err


def get_settings(config_path: Path | None = None) -> Settings:
    """Helper factory to retrieve settings instance."""
    if config_path and config_path.exists():
        return Settings.load_from_yaml(config_path)
    default_config = Path("configs/default.yaml")
    if default_config.exists():
        return Settings.load_from_yaml(default_config)
    return Settings()
