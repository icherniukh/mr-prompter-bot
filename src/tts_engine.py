import io
import logging
import edge_tts

from src.config import TTS_VOICE

logger = logging.getLogger(__name__)


async def generate_speech(text: str, voice: str | None = None) -> bytes:
    """Generate audio bytes from text using Edge TTS.

    Returns raw audio bytes (MP3 format, suitable for Telegram voice notes).
    """
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise ValueError("Text for speech synthesis cannot be empty.")

    selected_voice = voice or TTS_VOICE
    communicate = edge_tts.Communicate(cleaned_text, selected_voice)
    buf = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            buf.write(chunk.get("data", b""))

    audio_bytes = buf.getvalue()
    if not audio_bytes:
        raise RuntimeError("No audio data returned by TTS engine.")

    return audio_bytes
