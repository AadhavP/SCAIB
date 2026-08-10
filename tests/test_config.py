"""Tests for settings system and YAML loader."""

from pathlib import Path

import pytest

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


def test_provider_aliases_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLM_API_KEY", "glm-secret")
    monkeypatch.setenv("GLM_MODEL", "glm-test")
    settings = Settings()
    assert settings.glm_api_key == "glm-secret"
    assert settings.glm_model == "glm-test"


def test_environment_variables_override_yaml(
    tmp_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_EVALS_API__PORT", "1234")
    settings = Settings.load_from_yaml(tmp_config_file)
    assert settings.app_name == "test-agent-evals"
    assert settings.api.port == 1234
