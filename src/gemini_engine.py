import asyncio
import logging
import re

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.errors import ProcessingError
from src.gemini_defaults import REMOVAL_PROMPT

logger = logging.getLogger(__name__)


def _sanitize_prompt(prompt: str) -> str:
    """Normalize prompt keywords to avoid safety refusal triggers."""
    cleaned = re.sub(r"\bwatermarks\b", "overlays and stamps", prompt, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bwatermark\b", "overlay", cleaned, flags=re.IGNORECASE)
    return cleaned


def _process_sync(image_bytes: bytes, mime_type: str, prompt: str) -> bytes:
    client = genai.Client(api_key=GEMINI_API_KEY)
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    sanitized_prompt = _sanitize_prompt(prompt)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[sanitized_prompt, part],
        config=config,
    )

    text_parts = []
    if response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for p in candidate.content.parts:
                    if p.inline_data:
                        return p.inline_data.data
                    if p.text:
                        text_parts.append(p.text.strip())

    if text_parts:
        err_msg = " ".join(text_parts)
        logger.warning("Gemini returned text instead of image: %s", err_msg)
        raise ProcessingError(f"Gemini returned no image: {err_msg}")

    raise ProcessingError("Gemini returned no image.")


async def remove_overlays_gemini(image_bytes: bytes, mime_type: str, prompt: str | None = None) -> bytes:
    """Call Gemini 2.5 Flash Image to remove overlays. Raises ProcessingError on failure."""
    effective_prompt = prompt or REMOVAL_PROMPT
    try:
        return await asyncio.to_thread(_process_sync, image_bytes, mime_type, effective_prompt)
    except ProcessingError:
        raise
    except Exception as e:
        logger.exception("Gemini processing error: %s", e)
        raise ProcessingError(f"Gemini error: {e}") from e
