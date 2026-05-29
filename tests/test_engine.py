import base64

import httpx
import pytest

from src import engine
from src.engine import ProcessingError, remove_overlays

PNG_BYTES = b"\x89PNG\r\n\x1a\nfakepngdata"


def _data_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _mock_transport(handler):
    return httpx.MockTransport(handler)


@pytest.fixture
def patch_client(monkeypatch):
    """Patch httpx.AsyncClient so engine calls hit a MockTransport handler."""

    def _install(handler):
        real_async_client = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = _mock_transport(handler)
            return real_async_client(*args, **kwargs)

        monkeypatch.setattr(engine.httpx, "AsyncClient", factory)

    return _install


async def test_returns_decoded_image(patch_client):
    def handler(request):
        body = {
            "choices": [
                {"message": {"images": [{"image_url": {"url": _data_url(PNG_BYTES)}}]}}
            ]
        }
        return httpx.Response(200, json=body)

    patch_client(handler)
    out = await remove_overlays("sk-or-x", PNG_BYTES, "google/gemini-2.5-flash-image", "image/png")
    assert out == PNG_BYTES


async def test_sends_image_and_instruction(patch_client):
    captured = {}

    def handler(request):
        import json

        captured.update(json.loads(request.content))
        body = {
            "choices": [
                {"message": {"images": [{"image_url": {"url": _data_url(PNG_BYTES)}}]}}
            ]
        }
        return httpx.Response(200, json=body)

    patch_client(handler)
    await remove_overlays("sk-or-x", PNG_BYTES, "some/model", "image/png")
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "watermark" in content[0]["text"].lower()
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert captured["model"] == "some/model"
    assert captured["modalities"] == ["image", "text"]


async def test_no_image_in_response_raises(patch_client):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "sorry"}}]})

    patch_client(handler)
    with pytest.raises(ProcessingError):
        await remove_overlays("sk-or-x", PNG_BYTES, "m", "image/png")


@pytest.mark.parametrize("code", [401, 402, 429, 500])
async def test_http_errors_raise(patch_client, code):
    def handler(request):
        return httpx.Response(code, json={"error": "nope"})

    patch_client(handler)
    with pytest.raises(ProcessingError):
        await remove_overlays("sk-or-x", PNG_BYTES, "m", "image/png")


async def test_network_error_raises(patch_client):
    def handler(request):
        raise httpx.ConnectError("boom")

    patch_client(handler)
    with pytest.raises(ProcessingError):
        await remove_overlays("sk-or-x", PNG_BYTES, "m", "image/png")
