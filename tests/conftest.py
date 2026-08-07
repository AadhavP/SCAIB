"""Pytest test suite configuration and fixtures."""

from pathlib import Path

import pytest

from agent_evals.core.config import Settings


@pytest.fixture
def tmp_config_file(tmp_path: Path) -> Path:
    """Fixture providing temporary YAML config file."""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        "app_name: 'test-agent-evals'\n"
        "log_level: 'DEBUG'\n"
        "api:\n"
        "  port: 9000\n",
        encoding="utf-8",
    )
    return config_file


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing default test Settings object."""
    return Settings(app_name="test-agent-evals", environment="testing")
