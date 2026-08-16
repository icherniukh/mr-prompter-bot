import importlib

from src import config


def test_defaults_present():
    assert config.GEMINI_API_KEY
    assert config.GEMINI_MODEL
    assert config.SUPPORT_ACCOUNT == "@kappa_alive"
    assert config.TTS_VOICE == "en-US-ChristopherNeural"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "override-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("SUPPORT_ACCOUNT", "@other_support")
    monkeypatch.setenv("SUPPORT_CHAT_ID", "123456789")
    monkeypatch.setenv("TTS_VOICE", "en-GB-RyanNeural")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.GEMINI_API_KEY == "override-key"
        assert reloaded.GEMINI_MODEL == "gemini-test-model"
        assert reloaded.SUPPORT_ACCOUNT == "@other_support"
        assert reloaded.SUPPORT_CHAT_ID == "123456789"
        assert reloaded.TTS_VOICE == "en-GB-RyanNeural"
    finally:
        monkeypatch.undo()
        importlib.reload(config)
