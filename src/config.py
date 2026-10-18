import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"

TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"].strip()
OPENROUTER_API_KEY: str = (
    os.getenv("OPENROUTER_API_KEY") or os.getenv("HOST_OPENROUTER_KEY") or ""
).strip()
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OpenRouter API key not found. Set OPENROUTER_API_KEY or HOST_OPENROUTER_KEY."
    )

OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL
).strip()
# Optional override; when unset/empty, the engine dynamically discovers active
# :free models from OpenRouter's /models catalog and caches the working one.
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "").strip()
