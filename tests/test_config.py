import importlib

from src import config


def test_defaults_present():
    assert config.OPENROUTER_API_KEY
    assert config.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert config.DEFAULT_OPENROUTER_MODEL == "openrouter/free"
    assert config.OPENROUTER_MODEL == ""


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "override-or-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.com/api/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.OPENROUTER_API_KEY == "override-or-key"
        assert reloaded.OPENROUTER_BASE_URL == "https://example.com/api/v1"
        assert reloaded.OPENROUTER_MODEL == "google/gemma-4-26b-a4b-it:free"
    finally:
        monkeypatch.undo()
        importlib.reload(config)
