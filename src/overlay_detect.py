import asyncio
import json
import logging

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_VISION_MODEL
from src.gemini_defaults import DETECTION_PROMPT

logger = logging.getLogger(__name__)

# Gemini bounding boxes are normalized to a 0-1000 scale as [ymin, xmin, ymax, xmax],
# independent of the source image's actual pixel dimensions.
_BOX_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "box_2d": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.INTEGER),
            ),
        },
        required=["box_2d"],
    ),
)


def _build_prompt(extra_guidance: str | None) -> str:
    if not extra_guidance:
        return DETECTION_PROMPT
    return f"{DETECTION_PROMPT}\n\nAdditional guidance from the user: {extra_guidance}"


def _detect_sync(image_bytes: bytes, mime_type: str, extra_guidance: str | None) -> list[tuple[int, int, int, int]]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_BOX_SCHEMA,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = client.models.generate_content(
        model=GEMINI_VISION_MODEL,
        contents=[_build_prompt(extra_guidance), part],
        config=config,
    )

    if not response.text:
        return []
    try:
        raw_boxes = json.loads(response.text)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Overlay detection returned unparseable JSON: %s", e)
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for item in raw_boxes if isinstance(raw_boxes, list) else []:
        box = item.get("box_2d") if isinstance(item, dict) else None
        if isinstance(box, list) and len(box) == 4:
            try:
                ymin, xmin, ymax, xmax = (int(v) for v in box)
            except (TypeError, ValueError):
                continue
            if ymax > ymin and xmax > xmin:
                boxes.append((ymin, xmin, ymax, xmax))
    return boxes


async def detect_overlay_regions(
    image_bytes: bytes, mime_type: str, extra_guidance: str | None = None
) -> list[tuple[int, int, int, int]]:
    """Locate overlay/watermark regions via a free-tier Gemini vision call.

    Returns a list of (ymin, xmin, ymax, xmax) boxes normalized to a 0-1000 scale.
    Returns an empty list (never raises) on any detection failure, since detection
    failure should fall back to "nothing found" rather than break the pipeline.
    """
    try:
        return await asyncio.to_thread(_detect_sync, image_bytes, mime_type, extra_guidance)
    except Exception as e:
        logger.warning("Overlay detection failed: %s", e)
        return []
