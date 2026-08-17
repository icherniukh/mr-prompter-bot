import pytest
from unittest.mock import MagicMock

from src.errors import ProcessingError
from src import gemini_engine
from src.gemini_defaults import REMOVAL_PROMPT


def test_sanitize_prompt():
    assert gemini_engine._sanitize_prompt("remove watermark") == "remove overlay"
    assert gemini_engine._sanitize_prompt("Remove Watermarks please") == "Remove overlays and stamps please"
    assert gemini_engine._sanitize_prompt("clean image") == "clean image"


def test_process_sync_extracts_inline_data(monkeypatch):
    mock_client = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(data=b"GENERATED_BYTES")
    mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]))
    mock_response = MagicMock(candidates=[mock_candidate])
    mock_client.models.generate_content.return_value = mock_response

    monkeypatch.setattr(gemini_engine.genai, "Client", lambda **kwargs: mock_client)

    result = gemini_engine._process_sync(b"input", "image/png", "remove watermark")
    assert result == b"GENERATED_BYTES"
    call_args = mock_client.models.generate_content.call_args
    assert "remove overlay" in call_args.kwargs["contents"][0]


def test_process_sync_raises_on_text_refusal(monkeypatch):
    mock_client = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data = None
    mock_part.text = "Refusal reason"
    mock_candidate = MagicMock(content=MagicMock(parts=[mock_part]))
    mock_response = MagicMock(candidates=[mock_candidate])
    mock_client.models.generate_content.return_value = mock_response

    monkeypatch.setattr(gemini_engine.genai, "Client", lambda **kwargs: mock_client)

    with pytest.raises(ProcessingError, match="Gemini returned no image: Refusal reason"):
        gemini_engine._process_sync(b"input", "image/png", "test")


async def test_remove_overlays_gemini_paid_uses_default_prompt(monkeypatch):
    captured = {}

    async def fake_to_thread(func, image_bytes, mime_type, prompt):
        captured["image_bytes"] = image_bytes
        captured["mime_type"] = mime_type
        captured["prompt"] = prompt
        return b"CLEANED"

    monkeypatch.setattr(gemini_engine.asyncio, "to_thread", fake_to_thread)

    out = await gemini_engine.remove_overlays_gemini_paid(b"img", "image/png")
    assert out == b"CLEANED"
    assert captured["prompt"] == REMOVAL_PROMPT
    assert gemini_engine.REMOVAL_PROMPT == REMOVAL_PROMPT


async def test_remove_overlays_gemini_paid_passes_custom_prompt(monkeypatch):
    captured = {}

    async def fake_to_thread(func, image_bytes, mime_type, prompt):
        captured["prompt"] = prompt
        return b"CLEANED"

    monkeypatch.setattr(gemini_engine.asyncio, "to_thread", fake_to_thread)

    await gemini_engine.remove_overlays_gemini_paid(b"img", "image/png", "custom prompt")
    assert captured["prompt"] == "custom prompt"


async def test_remove_overlays_gemini_paid_wraps_unexpected_errors(monkeypatch):
    async def fake_to_thread(func, image_bytes, mime_type, prompt):
        raise RuntimeError("boom")

    monkeypatch.setattr(gemini_engine.asyncio, "to_thread", fake_to_thread)

    with pytest.raises(ProcessingError, match="Gemini error: boom"):
        await gemini_engine.remove_overlays_gemini_paid(b"img", "image/png")


async def test_remove_overlays_gemini_detects_then_inpaints(monkeypatch):
    captured = {}

    async def fake_detect(image_bytes, mime_type, extra_guidance):
        captured["detect_args"] = (image_bytes, mime_type, extra_guidance)
        return [(10, 20, 30, 40)]

    def fake_inpaint(image_bytes, mime_type, boxes):
        captured["inpaint_args"] = (image_bytes, mime_type, boxes)
        return b"CLEANED"

    monkeypatch.setattr(gemini_engine, "detect_overlay_regions", fake_detect)
    monkeypatch.setattr(gemini_engine, "inpaint_regions", fake_inpaint)

    out = await gemini_engine.remove_overlays_gemini(b"img", "image/png", "custom prompt")

    assert out == b"CLEANED"
    assert captured["detect_args"] == (b"img", "image/png", "custom prompt")
    assert captured["inpaint_args"] == (b"img", "image/png", [(10, 20, 30, 40)])


async def test_remove_overlays_gemini_raises_when_nothing_detected(monkeypatch):
    async def fake_detect(image_bytes, mime_type, extra_guidance):
        return []

    monkeypatch.setattr(gemini_engine, "detect_overlay_regions", fake_detect)

    with pytest.raises(ProcessingError, match="No overlays detected"):
        await gemini_engine.remove_overlays_gemini(b"img", "image/png")


async def test_remove_overlays_gemini_raises_when_local_inpainting_unavailable(monkeypatch):
    async def fake_detect(image_bytes, mime_type, extra_guidance):
        return [(10, 20, 30, 40)]

    monkeypatch.setattr(gemini_engine, "detect_overlay_regions", fake_detect)
    monkeypatch.setattr(gemini_engine, "inpaint_regions", lambda *a: None)

    with pytest.raises(ProcessingError, match="local inpainting is unavailable"):
        await gemini_engine.remove_overlays_gemini(b"img", "image/png")
