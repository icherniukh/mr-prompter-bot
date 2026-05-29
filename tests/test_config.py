import importlib

from src import config


def test_shortlist_parsing():
    assert config._parse_shortlist("a, b ,c") == ["a", "b", "c"]
    assert config._parse_shortlist("") == []


def test_defaults_present():
    assert config.FREE_TIER_LIMIT >= 1
    assert config.MODEL_SHORTLIST, "shortlist must not be empty"
    assert config.DEFAULT_MODEL in config.MODEL_SHORTLIST


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MODEL_SHORTLIST", "x/one,x/two")
    monkeypatch.setenv("FREE_TIER_LIMIT", "5")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.MODEL_SHORTLIST == ["x/one", "x/two"]
        assert reloaded.FREE_TIER_LIMIT == 5
        assert reloaded.DEFAULT_MODEL == "x/one"
    finally:
        monkeypatch.undo()
        importlib.reload(config)
