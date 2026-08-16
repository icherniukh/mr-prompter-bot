import io
import logging
import math
import os
import threading

from PIL import Image

logger = logging.getLogger(__name__)

FULL_HD_PIXELS = 1920 * 1080
FULL_HD_LONG_EDGE = 1920

SR_MODEL_PATH = os.getenv("SR_MODEL_PATH", "models/realesr-general-x4v3.onnx")

# The ONNX export has a fixed 1x3x128x128 input and upscales 4x.
_SR_TILE = 128
_SR_SCALE = 4
_SR_OVERLAP = 8
_SR_CORE = _SR_TILE - 2 * _SR_OVERLAP

# Inference is serialized: the VPS has 2 vCPUs and little spare RAM, so
# concurrent SR runs would starve the bot's event loop.
_sr_lock = threading.Lock()
_sr_session = None
_sr_unavailable = False


def upscale_to_full_hd_if_needed(image_bytes: bytes, mime_type: str) -> bytes:
    """Upscale below-Full-HD images locally while preserving aspect ratio.

    Uses Real-ESRGAN (compact, ONNX, CPU) when the model is available and
    falls back to plain Lanczos otherwise. Blocking — call via a thread
    executor from async code.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            target_size = _full_hd_target_size(width, height)
            if target_size == (width, height):
                return image_bytes

            resized = _super_resolve(image, target_size)
            if resized is None:
                resized = image.resize(target_size, Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image_format = _format_for_mime(mime_type, image.format)
            if image_format == "JPEG" and resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            resized.save(output, format=image_format)
            return output.getvalue()
    except Exception as e:
        logger.warning("Image upscaling failed, returning original bytes: %s", e)
        return image_bytes


def _super_resolve(image: Image.Image, target_size: tuple[int, int]) -> Image.Image | None:
    """Real-ESRGAN 4x then Lanczos to target size; None if SR is unavailable."""
    session = _sr_session_or_none()
    if session is None:
        return None
    try:
        import numpy as np

        alpha = image.getchannel("A") if "A" in image.getbands() else None
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        with _sr_lock:
            sr = _sr_tiled(session, arr)
        sr_uint8 = (sr.clip(0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        result = Image.fromarray(sr_uint8).resize(target_size, Image.Resampling.LANCZOS)
        if alpha is not None:
            result.putalpha(alpha.resize(target_size, Image.Resampling.LANCZOS))
        return result
    except Exception as e:
        logger.warning("Super-resolution failed, falling back to Lanczos: %s", e)
        return None


def _sr_session_or_none():
    global _sr_session, _sr_unavailable
    if _sr_session is not None or _sr_unavailable:
        return _sr_session
    with _sr_lock:
        if _sr_session is not None or _sr_unavailable:
            return _sr_session
        try:
            import onnxruntime as ort

            _sr_session = ort.InferenceSession(
                SR_MODEL_PATH, providers=["CPUExecutionProvider"]
            )
            logger.info("Loaded SR model from %s", SR_MODEL_PATH)
        except Exception as e:
            _sr_unavailable = True
            logger.warning(
                "SR model unavailable (%s); using Lanczos upscaling: %s",
                SR_MODEL_PATH,
                e,
            )
    return _sr_session


def _sr_tiled(session, arr):
    """Run fixed-size tiled SR over an HWC float32 array in [0, 1]."""
    import numpy as np

    height, width = arr.shape[:2]
    tiles_y = math.ceil(height / _SR_CORE)
    tiles_x = math.ceil(width / _SR_CORE)
    padded = np.pad(
        arr,
        (
            (_SR_OVERLAP, _SR_OVERLAP + tiles_y * _SR_CORE - height),
            (_SR_OVERLAP, _SR_OVERLAP + tiles_x * _SR_CORE - width),
            (0, 0),
        ),
        mode="edge",
    )
    out = np.empty((height * _SR_SCALE, width * _SR_SCALE, 3), dtype=np.float32)
    input_name = session.get_inputs()[0].name
    margin = _SR_OVERLAP * _SR_SCALE
    core_out = _SR_CORE * _SR_SCALE
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0, x0 = ty * _SR_CORE, tx * _SR_CORE
            tile = padded[y0 : y0 + _SR_TILE, x0 : x0 + _SR_TILE]
            batch = tile.transpose(2, 0, 1)[np.newaxis]
            sr_tile = session.run(None, {input_name: batch})[0][0].transpose(1, 2, 0)
            core = sr_tile[margin : margin + core_out, margin : margin + core_out]
            oy, ox = y0 * _SR_SCALE, x0 * _SR_SCALE
            crop_h = min(core_out, height * _SR_SCALE - oy)
            crop_w = min(core_out, width * _SR_SCALE - ox)
            out[oy : oy + crop_h, ox : ox + crop_w] = core[:crop_h, :crop_w]
    return out


def _full_hd_target_size(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return width, height
    if width * height >= FULL_HD_PIXELS:
        return width, height

    scale = math.sqrt(FULL_HD_PIXELS / (width * height))
    long_edge = max(width, height) * scale
    if long_edge > FULL_HD_LONG_EDGE:
        scale = FULL_HD_LONG_EDGE / max(width, height)
    if scale <= 1:
        return width, height

    return round(width * scale), round(height * scale)


def _format_for_mime(mime_type: str, fallback: str | None) -> str:
    if "png" in mime_type:
        return "PNG"
    if "webp" in mime_type:
        return "WEBP"
    if fallback:
        return fallback
    return "JPEG"
