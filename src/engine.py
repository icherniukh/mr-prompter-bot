import asyncio
import base64
import logging

import httpx

from src.config import OPENROUTER_BASE_URL, PROCESS_TIMEOUT

logger = logging.getLogger(__name__)

# The instruction sent alongside every image. The bot is "functional": users
# normally send no prompt, so this fixed instruction defines the whole job.
#
# This version is deliberately conservative after real-world testing:
# - Only targets artificial post-capture overlays/watermarks.
# - Explicitly protects real scene content (signage, architecture, etc.).
# - Strongly emphasizes exact dimension, resolution, and composition preservation.
REMOVAL_INSTRUCTION = (
    "Carefully remove ONLY artificial watermarks, logos, text overlays, captions, "
    "labels, timestamps, signatures, and other superimposed graphics that appear "
    "to have been added after the original photo was taken (such as stock photo "
    "watermarks or post-processing branding).\n\n"
    "DO NOT remove, alter, or inpaint over any text, signs, logos, numbers, or "
    "markings that are physically part of the real-world scene, including building "
    "names, addresses, entrance signage, architectural details, or any other "
    "legitimate environmental text.\n\n"
    "For the areas being cleaned, seamlessly reconstruct by extending the exact "
    "surrounding textures, lighting, shadows, colors, perspective, grain, noise, "
    "and material properties so the result looks completely natural and untouched. "
    "Make the edit invisible with no artifacts, halos, or inconsistencies.\n\n"
    "CRITICAL: Preserve the exact original image dimensions, aspect ratio, "
    "resolution, composition, framing, and pixel fidelity. Do not crop, pad, "
    "resize, rotate, recolor, stylize, or change anything about the overall image "
    "structure. Return only the cleaned image at precisely the same size and "
    "proportions as the input."
)


class ProcessingError(Exception):
    """Raised when an image could not be processed into a cleaned image."""


def _data_url(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime_type};base64,{b64}"


def _extract_image(message: dict) -> bytes:
    """Pull the first returned image out of an OpenRouter chat-completion message."""
    images = message.get("images") or []
    for img in images:
        url = (img.get("image_url") or {}).get("url", "")
        if url.startswith("data:") and "base64," in url:
            return base64.b64decode(url.split("base64,", 1)[1])
    raise ProcessingError("Model returned no image.")


def _extract_cost_from_response(data: dict) -> float | None:
    """
    Best-effort extraction of cost from OpenRouter response.

    OpenRouter includes cost in different places depending on the model/provider.
    We try the most common locations.
    """
    usage = data.get("usage") or {}

    # Common locations seen in practice
    for key in ("cost", "total_cost", "price"):
        if key in usage and isinstance(usage[key], (int, float)):
            return float(usage[key])
        if key in data and isinstance(data[key], (int, float)):
            return float(data[key])

    # Sometimes it's nested deeper
    if "total_cost" in usage:
        return float(usage["total_cost"])

    return None


async def remove_overlays(
    api_key: str,
    image_bytes: bytes,
    model: str,
    mime_type: str = "image/jpeg",
    *,
    image_config: dict | None = None,
    instruction: str | None = None,
    output_modalities: list[str] | None = None,
) -> bytes:
    """Send one image to an OpenRouter image-output model and return the cleaned bytes.

    image_config (optional): Passed directly to OpenRouter as `image_config`.
        Useful keys include:
        - "aspect_ratio", "image_size", "strength" (on supported models)

    instruction (optional): Override the default REMOVAL_INSTRUCTION.

    output_modalities (optional): List of desired output modalities.
        Default: ["image"] (pure image output — works for most editing models).
        Use ["image", "text"] only for models that require/return text alongside the image
        (some Gemini and GPT image variants).

    Returns:
        (cleaned_bytes, metadata_dict)
        metadata_dict contains: generation_id, cost (if available), model, modalities, etc.

    Raises ProcessingError on any failure so the caller can decide how to report
    it and (for free-tier usage) refund the reserved slot.
    """
    effective_instruction = instruction or REMOVAL_INSTRUCTION
    modalities = output_modalities if output_modalities is not None else ["image"]

    payload: dict = {
        "model": model,
        "modalities": modalities,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": effective_instruction},
                    {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime_type)}},
                ],
            }
        ],
    }

    if image_config:
        payload["image_config"] = image_config
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional attribution headers recommended by OpenRouter.
        "X-Title": "Mr Prompter Watermark Remover",
    }
    try:
        async with httpx.AsyncClient(timeout=PROCESS_TIMEOUT) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
    except (httpx.TimeoutException, asyncio.TimeoutError) as e:
        logger.warning("OpenRouter timeout [%s]: %s", model, e)
        raise ProcessingError("The image service took too long to respond.") from e
    except httpx.HTTPError as e:
        logger.error("OpenRouter request error [%s]: %s", model, e)
        raise ProcessingError("Could not reach the image service.") from e

    if resp.status_code == 401:
        raise ProcessingError("The API key was rejected (401). Check your OpenRouter key.")
    if resp.status_code == 402:
        raise ProcessingError("The API key has no remaining credit (402).")
    if resp.status_code == 429:
        raise ProcessingError("Rate limited by the image service (429). Try again shortly.")
    if resp.status_code >= 400:
        logger.error("OpenRouter HTTP %s [%s]: %s", resp.status_code, model, resp.text[:500])
        raise ProcessingError(f"Image service error ({resp.status_code}).")

    try:
        data = resp.json()
        message = data["choices"][0]["message"]
    except (ValueError, KeyError, IndexError) as e:
        logger.error("Unexpected OpenRouter response [%s]: %s", model, e)
        raise ProcessingError("Unexpected response from the image service.") from e

    cleaned_bytes = _extract_image(message)

    # Extract cost information if available from OpenRouter response
    cost_info = _extract_cost_from_response(data)

    metadata = {
        "generation_id": data.get("id"),
        "cost": cost_info,
        "model": model,
        "modalities": modalities,
        "image_config": image_config,
        "raw_usage": data.get("usage"),
    }

    return cleaned_bytes, metadata
