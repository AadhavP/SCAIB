"""Tests for settings system and YAML loader."""

from pathlib import Path

from agent_evals.core.config import Settings, get_settings


def test_default_settings() -> None:
    settings = Settings()
    assert settings.app_name == "agent-evals"
    assert settings.api.port == 8000


def test_load_from_yaml(tmp_config_file: Path) -> None:
    settings = Settings.load_from_yaml(tmp_config_file)
    assert settings.app_name == "test-agent-evals"
    assert settings.log_level == "DEBUG"
    assert settings.api.port == 9000


def test_get_settings_fallback() -> None:
    settings = get_settings(Path("non_existent_config.yaml"))
    assert settings.app_name == "agent-evals"
