import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_KEY_FILE = Path("data/secret.key")


def _load_encryption_key() -> str:
    """Load the Fernet encryption key from data/secret.key.

    On first run the key is read from the ENCRYPTION_KEY env var, written to
    the key file (mode 0400), and the env var is no longer needed afterwards.
    Keeping the key file separate from .env means a leaked .env cannot decrypt
    the database.
    """
    if _KEY_FILE.exists():
        key = _KEY_FILE.read_text().strip()
        if key:
            return key
    key = os.environ.get("ENCRYPTION_KEY", "")
    if not key:
        raise RuntimeError(
            "Encryption key not found. Set ENCRYPTION_KEY in .env for the first run; "
            "it will be written to data/secret.key automatically."
        )
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _KEY_FILE.write_text(key)
    _KEY_FILE.chmod(0o400)
    return key


def _parse_shortlist(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
ENCRYPTION_KEY: str = _load_encryption_key()

# The host's shared OpenRouter key, used to process each user's first
# FREE_TIER_LIMIT images. Optional: if unset, the free tier is disabled and
# users are asked to bring their own key immediately.
HOST_OPENROUTER_KEY: str = os.getenv("HOST_OPENROUTER_KEY", "").strip()

OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Number of images a user may process on the host key before they must bring
# their own OpenRouter key. Counted per user, for the lifetime of their record.
FREE_TIER_LIMIT: int = int(os.getenv("FREE_TIER_LIMIT", "25"))

# Curated shortlist of OpenRouter models that can actually *return* an edited
# image (image-output / image-editing models). A plain text/vision model cannot
# return a cleaned image, so the shortlist must only contain image-output models.
# Override with a comma-separated MODEL_SHORTLIST env var.
_DEFAULT_SHORTLIST = "google/gemini-2.5-flash-image,google/gemini-2.5-flash-image-preview"
MODEL_SHORTLIST: list[str] = _parse_shortlist(os.getenv("MODEL_SHORTLIST", _DEFAULT_SHORTLIST))

DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", MODEL_SHORTLIST[0])

# Per-image processing timeout (seconds).
PROCESS_TIMEOUT: float = float(os.getenv("PROCESS_TIMEOUT", "90"))

# Max images accepted per user submission window (a soft guard, not the free cap).
MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "25"))
