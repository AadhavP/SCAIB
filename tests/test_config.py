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
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://scaib.local")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "SCAIB")
    settings = Settings()
    assert settings.glm_api_key == "glm-secret"
    assert settings.glm_model == "glm-test"
    assert settings.openrouter_api_key == "openrouter-secret"
    assert settings.openrouter_model == "openai/gpt-5"
    assert settings.openrouter_http_referer == "https://scaib.local"
    assert settings.openrouter_app_title == "SCAIB"


def test_environment_variables_override_yaml(
    tmp_config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_EVALS_API__PORT", "1234")
    settings = Settings.load_from_yaml(tmp_config_file)
    assert settings.app_name == "test-agent-evals"
    assert settings.api.port == 1234
