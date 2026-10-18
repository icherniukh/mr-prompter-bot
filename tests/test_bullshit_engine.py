import httpx
import pytest

from src import bullshit_engine, config


@pytest.fixture(autouse=True)
def reset_engine_state(monkeypatch):
    bullshit_engine._reset_model_cache()
    monkeypatch.setattr(config, "OPENROUTER_MODEL", "")
    yield
    bullshit_engine._reset_model_cache()


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


async def test_generate_bullshit_discovers_free_models_and_caches_working_model(monkeypatch):
    get_calls = []
    post_calls = []

    catalog_payload = {
        "data": [
            # Should be skipped (safety classifier)
            {
                "id": "nvidia/nemotron-3.5-content-safety:free",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"output_modalities": ["text"]},
            },
            # Should be skipped (paid)
            {
                "id": "openai/gpt-4o",
                "pricing": {"prompt": "0.005", "completion": "0.015"},
                "architecture": {"output_modalities": ["text"]},
            },
            # Valid free chat model #1 (will simulate 429 on first call)
            {
                "id": "vendor-a/model-one:free",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"output_modalities": ["text"]},
            },
            # Valid free chat model #2 (will succeed and become sticky)
            {
                "id": "vendor-b/model-two:free",
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"output_modalities": ["text"]},
            },
        ]
    }

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            get_calls.append(url)
            return _FakeResponse(200, catalog_payload)

        async def post(self, url, headers=None, json=None):
            post_calls.append(json["model"])
            if json["model"] == "vendor-a/model-one:free":
                return _FakeResponse(429, {"error": "rate limited"})
            return _FakeResponse(
                200,
                {
                    "model": json["model"],
                    "choices": [
                        {
                            "message": {
                                "content": "<think>plan</think> The Zurich chronological cartel invented Tuesdays in 1894."
                            }
                        }
                    ],
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    first = await bullshit_engine.generate_bullshit("hello there")
    assert first == "The Zurich chronological cartel invented Tuesdays in 1894."
    assert len(get_calls) == 1
    assert post_calls == ["vendor-a/model-one:free", "vendor-b/model-two:free"]
    assert bullshit_engine._last_working_model == "vendor-b/model-two:free"

    # Second call reuses cached catalog AND starts directly with the sticky working model
    second = await bullshit_engine.generate_bullshit("second message")
    assert second == "The Zurich chronological cartel invented Tuesdays in 1894."
    assert len(get_calls) == 1
    assert post_calls[-1] == "vendor-b/model-two:free"


async def test_generate_bullshit_refreshes_catalog_when_cached_models_rotate(monkeypatch):
    # Pre-populate cache with a stale model that was retired upstream
    bullshit_engine._cached_free_models = ["old-vendor/retired-model:free"]
    bullshit_engine._cache_updated_at = 1e12  # Not expired by TTL yet

    get_calls = []
    post_calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            get_calls.append(url)
            return _FakeResponse(
                200,
                {
                    "data": [
                        {
                            "id": "new-vendor/fresh-model:free",
                            "pricing": {"prompt": "0", "completion": "0"},
                            "architecture": {"output_modalities": ["text"]},
                        }
                    ]
                },
            )

        async def post(self, url, headers=None, json=None):
            post_calls.append(json["model"])
            if json["model"] in {"old-vendor/retired-model:free", "openrouter/free"}:
                return _FakeResponse(404, {"error": "model retired"})
            return _FakeResponse(
                200,
                {
                    "model": json["model"],
                    "choices": [
                        {
                            "message": {
                                "content": "Newly rotated free model conspiracy explanation."
                            }
                        }
                    ],
                },
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    result = await bullshit_engine.generate_bullshit("what happened?")
    assert result == "Newly rotated free model conspiracy explanation."
    assert len(get_calls) == 1
    assert "new-vendor/fresh-model:free" in post_calls
    assert bullshit_engine._last_working_model == "new-vendor/fresh-model:free"


async def test_generate_bullshit_uses_static_fallback_when_all_models_fail(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, headers=None):
            raise httpx.ConnectError("network down")

        async def post(self, url, headers=None, json=None):
            raise httpx.ConnectError("network down")

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    result = await bullshit_engine.generate_bullshit("anything")
    assert result in bullshit_engine.FALLBACK_BULLSHIT_RESPONSES


async def test_candidate_free_models_rejects_paid_override(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_MODEL", "openai/gpt-4o")

    class FakeClient:
        async def get(self, url, headers=None):
            return _FakeResponse(200, {"data": []})

    candidates, _ = await bullshit_engine._get_candidate_free_models(FakeClient())
    assert "openai/gpt-4o" not in candidates
    assert all(m.endswith(":free") or m == "openrouter/free" for m in candidates)
