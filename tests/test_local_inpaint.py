import io
import os

import pytest
from PIL import Image

from src import local_inpaint
from src.local_inpaint import inpaint_regions


def _png_bytes(size: tuple[int, int], color=(120, 80, 40)) -> bytes:
    image = Image.new("RGB", size, color=color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_inpaint_regions_returns_none_without_boxes():
    assert inpaint_regions(_png_bytes((100, 100)), "image/png", []) is None


def test_inpaint_regions_returns_none_when_model_unavailable(monkeypatch):
    monkeypatch.setattr(local_inpaint, "_migan_session", None)
    monkeypatch.setattr(local_inpaint, "_migan_unavailable", False)
    monkeypatch.setattr(local_inpaint, "MIGAN_MODEL_PATH", "models/does-not-exist.onnx")

    result = inpaint_regions(_png_bytes((100, 100)), "image/png", [(0, 0, 500, 500)])
    assert result is None


def test_pixel_rects_pads_and_clamps_to_bounds():
    rects = local_inpaint._pixel_rects([(0, 0, 1000, 1000)], width=200, height=100)
    assert rects == [(0, 0, 200, 100)]


def test_union_with_context_expands_and_clamps():
    box = local_inpaint._union_with_context([(90, 90, 110, 110)], width=100, height=100)
    x0, y0, x1, y1 = box
    assert x0 <= 90 and y0 <= 90
    assert x1 == 100 and y1 == 100


def test_fit_for_inference_leaves_small_crops_unchanged():
    image = Image.new("RGB", (200, 150), (10, 20, 30))
    mask = Image.new("L", (200, 150), 0)
    out_image, out_mask = local_inpaint._fit_for_inference(image, mask)
    assert out_image.size == (200, 150)
    assert out_mask.size == (200, 150)


def test_fit_for_inference_downscales_large_crops():
    image = Image.new("RGB", (2000, 1000), (10, 20, 30))
    mask = Image.new("L", (2000, 1000), 0)
    out_image, out_mask = local_inpaint._fit_for_inference(image, mask)
    assert max(out_image.size) == local_inpaint._MIGAN_MAX_EDGE
    assert out_image.size == out_mask.size


_needs_migan_model = pytest.mark.skipif(
    (
        not os.path.exists(local_inpaint.MIGAN_MODEL_PATH)
        or os.getenv("RUN_MIGAN_MODEL_TESTS") != "1"
    ),
    reason="MI-GAN model tests are opt-in; set RUN_MIGAN_MODEL_TESTS=1 after downloading the model",
)


@_needs_migan_model
def test_inpaint_regions_preserves_pixels_outside_boxes():
    width, height = 300, 200
    image = Image.new("RGB", (width, height), (10, 20, 30))
    for x in range(250, 280):
        for y in range(20, 50):
            image.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")

    box = (round(20 / height * 1000), round(250 / width * 1000), round(50 / height * 1000), round(280 / width * 1000))
    result = inpaint_regions(buf.getvalue(), "image/png", [box])
    assert result is not None

    out = Image.open(io.BytesIO(result)).convert("RGB")
    assert out.size == (width, height)
    assert out.getpixel((10, 10)) == (10, 20, 30)
