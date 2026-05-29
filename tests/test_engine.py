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


# ── Watermark removal operation tests (single image) ─────────────────────────


async def test_single_image_watermark_removal_sends_exact_instruction_and_one_image(patch_client):
    """Test the actual watermark removal operation on one image.

    Verifies that remove_overlays sends the fixed REMOVAL_INSTRUCTION together
    with exactly one image (never a collage or multi-image payload).
    """
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
    await remove_overlays("sk-or-x", PNG_BYTES, "google/gemini-3.1-flash-image-preview", "image/png")

    content = captured["messages"][0]["content"]
    # The instruction must be present verbatim (or at least the key phrase)
    assert content[0]["type"] == "text"
    instruction = content[0]["text"]
    assert "Remove every watermark, logo, text overlay" in instruction
    assert "Return only the cleaned image." in instruction

    # Exactly one image in this request — this is the core "single image" contract
    assert content[1]["type"] == "image_url"
    assert len([c for c in content if c.get("type") == "image_url"]) == 1
    assert captured["model"] == "google/gemini-3.1-flash-image-preview"


async def test_remove_overlays_payload_never_contains_multiple_images(patch_client):
    """Engine-level guarantee: remove_overlays is strictly single-image.

    Even if a caller mistakenly tried to send a 'collage', the function only
    ever puts one image_url entry into the content array sent to the model.
    """
    captured = {}

    def handler(request):
        import json
        captured.update(json.loads(request.content))
        body = {"choices": [{"message": {"images": [{"image_url": {"url": _data_url(PNG_BYTES)}}]}}]}
        return httpx.Response(200, json=body)

    patch_client(handler)
    # Call as the bot always does — one image at a time
    await remove_overlays("sk-or-x", PNG_BYTES, "some/model", "image/jpeg")

    content = captured["messages"][0]["content"]
    image_entries = [c for c in content if c.get("type") == "image_url"]
    assert len(image_entries) == 1, "remove_overlays must never emit a multi-image/collage payload"
