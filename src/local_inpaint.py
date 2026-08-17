import io
import logging
import os
import threading

from PIL import Image, ImageDraw, ImageOps

logger = logging.getLogger(__name__)

MIGAN_MODEL_PATH = os.getenv("MIGAN_MODEL_PATH", "models/migan_pipeline_v2.onnx")

# MI-GAN's ONNX pipeline takes dynamic HxW uint8 image/mask tensors. Its own
# memory footprint stays flat (~470 MB peak RSS) up to at least 768px on the
# long edge -- verified empirically on the production VPS (2 vCPU, ~1.9 GB RAM,
# frequently near its memory limit). Crops are downscaled to this cap before
# inference and the result is resized back up before pasting into the original.
_MIGAN_MAX_EDGE = 768
# Grows each detected box slightly so the fill has a soft, blend-friendly edge
# instead of a hard rectangle boundary.
_EDGE_PAD_FRAC = 0.03
# Expands the inference crop beyond the union of overlay boxes so the model
# sees real surrounding texture to reconstruct from, not just the hole itself.
_CONTEXT_FRAC = 0.4

_migan_lock = threading.Lock()
_migan_session = None
_migan_unavailable = False


def inpaint_regions(
    image_bytes: bytes, mime_type: str, boxes_norm: list[tuple[int, int, int, int]]
) -> bytes | None:
    """Remove the given overlay regions locally via MI-GAN. None if unavailable or it fails.

    boxes_norm are (ymin, xmin, ymax, xmax) normalized to a 0-1000 scale, as returned
    by src.overlay_detect.detect_overlay_regions. Pixels outside the (padded) boxes
    are left byte-identical to the original -- only the detected regions are touched.
    """
    if not boxes_norm:
        return None
    session = _migan_session_or_none()
    if session is None:
        return None
    try:
        import numpy as np

        with Image.open(io.BytesIO(image_bytes)) as original:
            width, height = original.size
            image_format = _format_for_mime(mime_type, original.format)
            rgb = original.convert("RGB")

            rects = _pixel_rects(boxes_norm, width, height)
            if not rects:
                return None
            crop_box = _union_with_context(rects, width, height)
            x0, y0, x1, y1 = crop_box

            hole_mask_full = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(hole_mask_full)
            for rect in rects:
                draw.rectangle(rect, fill=255)

            crop_img = rgb.crop(crop_box)
            crop_hole_mask = hole_mask_full.crop(crop_box)
            crop_size = crop_img.size

            infer_img, infer_hole_mask = _fit_for_inference(crop_img, crop_hole_mask)
            infer_keep_mask = ImageOps.invert(infer_hole_mask)

            img_arr = np.asarray(infer_img, dtype=np.uint8).transpose(2, 0, 1)[np.newaxis]
            mask_arr = np.asarray(infer_keep_mask, dtype=np.uint8)[np.newaxis, np.newaxis]

            input_names = {i.name for i in session.get_inputs()}
            feed = {"image": img_arr, "mask": mask_arr}
            if not input_names.issuperset(feed):
                logger.warning("MI-GAN model has unexpected input names: %s", input_names)
                return None

            with _migan_lock:
                out = session.run(None, feed)[0]

            inpainted_crop = Image.fromarray(out[0].transpose(1, 2, 0)).resize(
                crop_size, Image.Resampling.LANCZOS
            )

            result = rgb.copy()
            result.paste(inpainted_crop, (x0, y0), crop_hole_mask)

            output = io.BytesIO()
            if image_format == "JPEG":
                result = result.convert("RGB")
            result.save(output, format=image_format)
            return output.getvalue()
    except Exception as e:
        logger.warning("Local inpainting failed: %s", e)
        return None


def _pixel_rects(
    boxes_norm: list[tuple[int, int, int, int]], width: int, height: int
) -> list[tuple[int, int, int, int]]:
    pad = round(_EDGE_PAD_FRAC * max(width, height))
    rects = []
    for ymin, xmin, ymax, xmax in boxes_norm:
        x0 = max(0, round(xmin / 1000 * width) - pad)
        y0 = max(0, round(ymin / 1000 * height) - pad)
        x1 = min(width, round(xmax / 1000 * width) + pad)
        y1 = min(height, round(ymax / 1000 * height) + pad)
        if x1 > x0 and y1 > y0:
            rects.append((x0, y0, x1, y1))
    return rects


def _union_with_context(
    rects: list[tuple[int, int, int, int]], width: int, height: int
) -> tuple[int, int, int, int]:
    ux0 = min(r[0] for r in rects)
    uy0 = min(r[1] for r in rects)
    ux1 = max(r[2] for r in rects)
    uy1 = max(r[3] for r in rects)
    margin_x = round((ux1 - ux0) * _CONTEXT_FRAC)
    margin_y = round((uy1 - uy0) * _CONTEXT_FRAC)
    return (
        max(0, ux0 - margin_x),
        max(0, uy0 - margin_y),
        min(width, ux1 + margin_x),
        min(height, uy1 + margin_y),
    )


def _fit_for_inference(image: Image.Image, hole_mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    long_edge = max(image.size)
    if long_edge <= _MIGAN_MAX_EDGE:
        return image, hole_mask
    scale = _MIGAN_MAX_EDGE / long_edge
    new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return (
        image.resize(new_size, Image.Resampling.LANCZOS),
        hole_mask.resize(new_size, Image.Resampling.NEAREST),
    )


def _migan_session_or_none():
    global _migan_session, _migan_unavailable
    if _migan_session is not None or _migan_unavailable:
        return _migan_session
    with _migan_lock:
        if _migan_session is not None or _migan_unavailable:
            return _migan_session
        try:
            import onnxruntime as ort

            _migan_session = ort.InferenceSession(
                MIGAN_MODEL_PATH, providers=["CPUExecutionProvider"]
            )
            logger.info("Loaded MI-GAN inpainting model from %s", MIGAN_MODEL_PATH)
        except Exception as e:
            _migan_unavailable = True
            logger.warning(
                "MI-GAN model unavailable (%s); local inpainting disabled: %s",
                MIGAN_MODEL_PATH,
                e,
            )
    return _migan_session


def _format_for_mime(mime_type: str, fallback: str | None) -> str:
    if "png" in mime_type:
        return "PNG"
    if "webp" in mime_type:
        return "WEBP"
    if fallback:
        return fallback
    return "JPEG"
