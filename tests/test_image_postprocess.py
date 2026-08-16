import io
import os

import pytest
from PIL import Image

from src import image_postprocess
from src.image_postprocess import upscale_to_full_hd_if_needed


def _jpeg_bytes(size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size, color=(120, 80, 40))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.size


def test_upscale_to_full_hd_increases_small_landscape_image():
    output = upscale_to_full_hd_if_needed(_jpeg_bytes((960, 540)), "image/jpeg")
    assert _image_size(output) == (1920, 1080)


def test_upscale_to_full_hd_preserves_portrait_aspect_ratio():
    output = upscale_to_full_hd_if_needed(_jpeg_bytes((540, 960)), "image/jpeg")
    assert _image_size(output) == (1080, 1920)


def test_upscale_to_full_hd_leaves_large_image_unchanged():
    original = _jpeg_bytes((1920, 1080))
    output = upscale_to_full_hd_if_needed(original, "image/jpeg")
    assert output == original


def test_upscale_falls_back_to_lanczos_when_sr_model_missing(monkeypatch):
    monkeypatch.setattr(image_postprocess, "_sr_session", None)
    monkeypatch.setattr(image_postprocess, "_sr_unavailable", False)
    monkeypatch.setattr(image_postprocess, "SR_MODEL_PATH", "models/does-not-exist.onnx")
    output = upscale_to_full_hd_if_needed(_jpeg_bytes((960, 540)), "image/jpeg")
    assert _image_size(output) == (1920, 1080)


_needs_sr_model = pytest.mark.skipif(
    (
        not os.path.exists(image_postprocess.SR_MODEL_PATH)
        or os.getenv("RUN_SR_MODEL_TESTS") != "1"
    ),
    reason="SR model tests are opt-in; set RUN_SR_MODEL_TESTS=1 after downloading the model",
)


@_needs_sr_model
def test_super_resolve_returns_target_size():
    image = Image.effect_noise((200, 150), 40).convert("RGB")
    result = image_postprocess._super_resolve(image, (400, 300))
    assert result is not None
    assert result.size == (400, 300)


@_needs_sr_model
def test_super_resolve_preserves_alpha_channel():
    image = Image.new("RGBA", (160, 120), (120, 80, 40, 200))
    result = image_postprocess._super_resolve(image, (320, 240))
    assert result is not None
    assert result.mode == "RGBA"
    assert result.size == (320, 240)
