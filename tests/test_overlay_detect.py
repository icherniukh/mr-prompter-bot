import json
from unittest.mock import MagicMock

from src import overlay_detect


def _mock_client(response_text: str | None):
    mock_client = MagicMock()
    mock_response = MagicMock(text=response_text)
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


def test_build_prompt_without_guidance():
    assert overlay_detect._build_prompt(None) == overlay_detect.DETECTION_PROMPT


def test_build_prompt_with_guidance():
    prompt = overlay_detect._build_prompt("remove the logo in the top-left corner")
    assert overlay_detect.DETECTION_PROMPT in prompt
    assert "remove the logo in the top-left corner" in prompt


def test_detect_sync_parses_boxes(monkeypatch):
    payload = json.dumps([{"box_2d": [10, 20, 100, 200]}, {"box_2d": [0, 0, 50, 50]}])
    mock_client = _mock_client(payload)
    monkeypatch.setattr(overlay_detect.genai, "Client", lambda **kwargs: mock_client)

    boxes = overlay_detect._detect_sync(b"img", "image/png", None)
    assert boxes == [(10, 20, 100, 200), (0, 0, 50, 50)]


def test_detect_sync_drops_invalid_boxes(monkeypatch):
    payload = json.dumps(
        [
            {"box_2d": [10, 20, 100, 200]},
            {"box_2d": [10, 20, 100]},  # wrong length
            {"box_2d": [50, 50, 10, 10]},  # inverted, ymax<ymin, xmax<xmin
            {"not_a_box": True},
        ]
    )
    mock_client = _mock_client(payload)
    monkeypatch.setattr(overlay_detect.genai, "Client", lambda **kwargs: mock_client)

    boxes = overlay_detect._detect_sync(b"img", "image/png", None)
    assert boxes == [(10, 20, 100, 200)]


def test_detect_sync_returns_empty_on_no_text(monkeypatch):
    mock_client = _mock_client(None)
    monkeypatch.setattr(overlay_detect.genai, "Client", lambda **kwargs: mock_client)

    assert overlay_detect._detect_sync(b"img", "image/png", None) == []


def test_detect_sync_returns_empty_on_unparseable_json(monkeypatch):
    mock_client = _mock_client("not json")
    monkeypatch.setattr(overlay_detect.genai, "Client", lambda **kwargs: mock_client)

    assert overlay_detect._detect_sync(b"img", "image/png", None) == []


async def test_detect_overlay_regions_returns_empty_on_exception(monkeypatch):
    async def fake_to_thread(func, *args):
        raise RuntimeError("boom")

    monkeypatch.setattr(overlay_detect.asyncio, "to_thread", fake_to_thread)

    boxes = await overlay_detect.detect_overlay_regions(b"img", "image/png")
    assert boxes == []


async def test_detect_overlay_regions_passes_guidance_through(monkeypatch):
    captured = {}

    async def fake_to_thread(func, image_bytes, mime_type, extra_guidance):
        captured["extra_guidance"] = extra_guidance
        return [(1, 2, 3, 4)]

    monkeypatch.setattr(overlay_detect.asyncio, "to_thread", fake_to_thread)

    boxes = await overlay_detect.detect_overlay_regions(b"img", "image/png", "custom guidance")
    assert boxes == [(1, 2, 3, 4)]
    assert captured["extra_guidance"] == "custom guidance"
