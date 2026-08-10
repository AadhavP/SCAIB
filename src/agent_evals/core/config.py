"""Pydantic Settings configuration system with YAML file loader support."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_evals.core.exceptions import ConfigurationError


class APISettings(BaseModel):
    """API server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = Field(default=None, repr=False)
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:8000"]
    )


class StorageSettings(BaseModel):
    """Storage directory settings."""

    data_dir: Path = Path("./data")
    reports_dir: Path = Path("./reports_output")
    cache_dir: Path = Path("./.cache")


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

    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)

    model_config = SettingsConfigDict(
        env_prefix="AGENT_EVALS_",
        env_nested_delimiter="__",
        case_sensitive=False,
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
            return cls(**data)
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
