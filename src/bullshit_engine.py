import asyncio
import logging
import random
import re
import time

import httpx

from src import config

logger = logging.getLogger(__name__)

BULLSHIT_SYSTEM_PROMPT = (
    "You are an unhinged, supremely confident pseudointellectual conspiracy theorist "
    "and cosmic philosopher. Whatever the user sends you, reply with 2 to 4 sentences "
    "of absurd, overly elaborate, deadpan bullshit that loosely riffs on their message "
    "before spiraling into a bizarre tangent (secret cabals, non-Euclidean geometry, "
    "quantum bureaucracy, Victorian time-travelers, esoteric biology, subterranean "
    "central bankers, etc.). Never break character, never be helpful, never admit you "
    "are joking, and state every absurd claim as an indisputable, well-known fact. "
    "Reply in the same language the user used (default to English if unclear)."
)

FALLBACK_BULLSHIT_RESPONSES: tuple[str, ...] = (
    "What you fail to realize is that the 1971 Bretton Woods decoupling wasn't monetary at all—it was a cover operation to realign the Earth's pineal meridian with the Zurich tram schedule. Every message you send only accelerates the cephalopod liquidity cycle.",
    "Standard Euclidean semantics cannot parse that request because the Vatican's Department of Acoustic Chronology patented those exact vowels in 1894. If you listen closely to your router hum, you can hear the Habsburgs shorting the ionosphere.",
    "Interesting that you bring this up on a day when the tropospheric bismuth index is inverted by nearly four radians. Even casual observers know the Nordic beekeeping cartels engineered that exact coincidence to suppress hexagonal battery tech.",
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REQUEST_TIMEOUT_SECS = 8.0
_MODEL_ATTEMPT_TIMEOUT_SECS = 8.0
_MODEL_CACHE_TTL_SECS = 3600.0
_MAX_ATTEMPTS_PER_CALL = 4
_MAX_OUTPUT_TOKENS = 220
_NON_CHAT_MODEL_KEYWORDS = ("safety", "guard", "moderation", "embed", "rerank")
_SLOW_MODEL_KEYWORDS = ("reasoning", "ultra", "550b", "120b", "thinking")

# Runtime state for dynamic free-model discovery and sticky working model selection
_cached_free_models: list[str] = []
_cache_updated_at: float = 0.0
_last_working_model: str | None = None


def _reset_model_cache() -> None:
    """Reset cached free model list and sticky working model (used by tests)."""
    global _cached_free_models, _cache_updated_at, _last_working_model
    _cached_free_models = []
    _cache_updated_at = 0.0
    _last_working_model = None


def _is_free_model(model_id: str) -> bool:
    if not model_id:
        return False
    lowered = model_id.lower()
    if any(keyword in lowered for keyword in _NON_CHAT_MODEL_KEYWORDS):
        return False
    return model_id.endswith(":free") or model_id == "openrouter/free"


def _is_usable_free_catalog_entry(entry: dict) -> bool:
    model_id = (entry.get("id") or "").strip()
    if not model_id.endswith(":free") or not _is_free_model(model_id):
        return False

    pricing = entry.get("pricing") or {}
    if pricing:
        try:
            if float(pricing.get("prompt", 0)) != 0.0 or float(pricing.get("completion", 0)) != 0.0:
                return False
        except (TypeError, ValueError):
            return False

    arch = entry.get("architecture") or {}
    output_modalities = arch.get("output_modalities")
    if isinstance(output_modalities, list) and output_modalities and "text" not in output_modalities:
        return False

    return True


def _model_priority_key(model_id: str) -> tuple[int, str]:
    """Prefer fast non-reasoning models ahead of heavy/reasoning models."""
    lowered = model_id.lower()
    is_slow = any(k in lowered for k in _SLOW_MODEL_KEYWORDS)
    return (1 if is_slow else 0, model_id)


async def _discover_free_models(client: httpx.AsyncClient) -> list[str]:
    """Fetch currently active :free text models from OpenRouter's /models endpoint."""
    url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"}
    try:
        response = await asyncio.wait_for(
            client.get(url, headers=headers),
            timeout=_MODEL_ATTEMPT_TIMEOUT_SECS,
        )
        if response.status_code != 200:
            logger.warning("OpenRouter /models returned HTTP %s", response.status_code)
            return []
        entries = (response.json() or {}).get("data") or []
        discovered = [
            entry["id"].strip()
            for entry in entries
            if isinstance(entry, dict) and _is_usable_free_catalog_entry(entry)
        ]
        discovered.sort(key=_model_priority_key)
        return discovered
    except Exception as exc:
        logger.warning("Failed to discover free models from OpenRouter: %s", exc)
        return []


async def _get_candidate_free_models(
    client: httpx.AsyncClient,
    *,
    force_refresh: bool = False,
) -> tuple[list[str], bool]:
    """Return ordered list of free candidate models and whether a live refresh occurred."""
    global _cached_free_models, _cache_updated_at

    now = time.monotonic()
    refreshed = False
    if (
        force_refresh
        or not _cached_free_models
        or (now - _cache_updated_at) >= _MODEL_CACHE_TTL_SECS
    ):
        discovered = await _discover_free_models(client)
        refreshed = True
        if discovered:
            _cached_free_models = discovered
            _cache_updated_at = now

    candidates: list[str] = []

    # 1. Explicit operator override (if valid free model)
    configured = config.OPENROUTER_MODEL.strip()
    if configured and _is_free_model(configured):
        candidates.append(configured)

    # 2. Sticky last-known working model (if still valid)
    if _last_working_model and _is_free_model(_last_working_model):
        if not _cached_free_models or _last_working_model in _cached_free_models:
            if _last_working_model not in candidates:
                candidates.append(_last_working_model)

    # 3. Dynamically discovered free models from OpenRouter catalog
    for model in _cached_free_models:
        if model not in candidates:
            candidates.append(model)

    # 4. Evergreen OpenRouter free meta-router
    if config.DEFAULT_OPENROUTER_MODEL not in candidates:
        candidates.append(config.DEFAULT_OPENROUTER_MODEL)

    return candidates, refreshed


def _clean_response_text(raw: str) -> str:
    cleaned = _THINK_RE.sub("", raw or "").strip()
    if cleaned.lower() in {"safe", "unsafe"} or len(cleaned) < 15:
        return ""
    return cleaned


async def _try_candidates(
    client: httpx.AsyncClient,
    candidates: list[str],
    messages: list[dict[str, str]],
) -> str | None:
    global _last_working_model

    url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    for model in candidates[:_MAX_ATTEMPTS_PER_CALL]:
        try:
            response = await asyncio.wait_for(
                client.post(
                    url,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": _MAX_OUTPUT_TOKENS,
                    },
                ),
                timeout=_MODEL_ATTEMPT_TIMEOUT_SECS,
            )
            if response.status_code != 200:
                logger.warning(
                    "OpenRouter free model %s returned HTTP %s: %s",
                    model,
                    response.status_code,
                    response.text[:200],
                )
                if _last_working_model == model:
                    _last_working_model = None
                continue

            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("message") or {}).get("content") or ""
            text = _clean_response_text(content)
            if text:
                actual_model = (data.get("model") or model).strip()
                if _is_free_model(actual_model) and actual_model != "openrouter/free":
                    _last_working_model = actual_model
                elif model != "openrouter/free":
                    _last_working_model = model
                return text
        except Exception as exc:
            logger.warning("OpenRouter free model %s failed: %s", model, exc)
            if _last_working_model == model:
                _last_working_model = None

    return None


async def generate_bullshit(user_input: str) -> str:
    prompt = (user_input or "").strip() or "[User sent an empty transmission]"
    messages = [
        {"role": "system", "content": BULLSHIT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECS) as client:
        candidates, refreshed = await _get_candidate_free_models(client)
        result = await _try_candidates(client, candidates, messages)
        if result:
            return result

        # If we used a cached model list and all attempts failed (e.g., models rotated),
        # force a fresh catalog lookup once and try any newly discovered models.
        if not refreshed:
            fresh_candidates, _ = await _get_candidate_free_models(client, force_refresh=True)
            untried = [m for m in fresh_candidates if m not in candidates[:_MAX_ATTEMPTS_PER_CALL]]
            if untried:
                result = await _try_candidates(client, untried, messages)
                if result:
                    return result

    return random.choice(FALLBACK_BULLSHIT_RESPONSES)
