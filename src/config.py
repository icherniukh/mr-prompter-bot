import os
from dotenv import load_dotenv

from src.gemini_defaults import DEFAULT_GEMINI_IMAGE_MODEL

load_dotenv()

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"].strip()
GEMINI_API_KEY: str = (
    os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
).strip()
if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY.")

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_IMAGE_MODEL).strip()
SUPPORT_ACCOUNT: str = os.getenv("SUPPORT_ACCOUNT", "@kappa_alive").strip()
SUPPORT_CHAT_ID: str = os.getenv("SUPPORT_CHAT_ID", "").strip()
DEFAULT_TTS_VOICE = "en-US-ChristopherNeural"
TTS_VOICE: str = os.getenv("TTS_VOICE", DEFAULT_TTS_VOICE).strip()
