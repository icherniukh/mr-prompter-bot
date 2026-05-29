import asyncio
import base64
import logging

import httpx

from src.config import OPENROUTER_BASE_URL, PROCESS_TIMEOUT

logger = logging.getLogger(__name__)

# The instruction sent alongside every image. The bot is "functional": users
# normally send no prompt, so this fixed instruction defines the whole job.
REMOVAL_INSTRUCTION = (
    "Remove every watermark, logo, text overlay, caption, label, timestamp, "
    "signature, and superimposed graphic from this image. Reconstruct the "
    "underlying image content naturally and seamlessly wherever something was "
    "removed, matching surrounding texture, lighting, and detail. Do not crop, "
    "rotate, resize, recolor, or stylize the image, and do not add anything new. "
    "Preserve the original composition and resolution. Return only the cleaned image."
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


async def remove_overlays(
    api_key: str,
    image_bytes: bytes,
    model: str,
    mime_type: str = "image/jpeg",
) -> bytes:
    """Send one image to an OpenRouter image-output model and return the cleaned bytes.

    Raises ProcessingError on any failure so the caller can decide how to report
    it and (for free-tier usage) refund the reserved slot.
    """
    payload = {
        "model": model,
        "modalities": ["image", "text"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": REMOVAL_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": _data_url(image_bytes, mime_type)}},
                ],
            }
        ],
    }
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

    return _extract_image(message)
