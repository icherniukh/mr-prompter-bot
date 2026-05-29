import io
import logging

from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler

from src import database as db
from src.config import (
    DEFAULT_MODEL,
    FREE_TIER_LIMIT,
    HOST_OPENROUTER_KEY,
    MODEL_SHORTLIST,
)
from src.engine import ProcessingError, remove_overlays

logger = logging.getLogger(__name__)

# Conversation states for /setup
ASK_KEY, ASK_MODEL = range(2)


def _model_menu() -> str:
    lines = [f"{i + 1}. `{m}`" for i, m in enumerate(MODEL_SHORTLIST)]
    return "\n".join(lines)


# ── Static commands ────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 *Mr Prompter* — batch watermark & overlay remover.\n\n"
        "Just send me one or more images (as photos or files). I clean each one "
        "independently — removing watermarks, logos, text overlays, captions and "
        "labels — and send the cleaned versions back. No instructions needed.\n\n"
        f"🎁 Your first {FREE_TIER_LIMIT} images are on me. After that, run /setup "
        "to add your own OpenRouter key and keep going.\n\n"
        "Commands: /status · /setup · /model · /forget",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📖 *How to use*\n\n"
        "1. Send any number of images (photos or image files).\n"
        "2. I process each one independently and return the cleaned image.\n"
        "3. For best quality, send images *as files* (uncompressed).\n\n"
        f"You get {FREE_TIER_LIMIT} free images. After that use /setup to add your "
        "own OpenRouter key.\n\n"
        "/status — see how many free images remain\n"
        "/setup — add your OpenRouter API key\n"
        "/model — choose the AI model\n"
        "/forget — delete all your stored data",
        parse_mode="Markdown",
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await db.get_user(update.effective_user.id)
    free_used = user["free_used"] if user else 0
    has_key = bool(user and user["api_key"])
    model = (user and user["model"]) or DEFAULT_MODEL

    lines = [f"🤖 Model: `{model}`"]
    if has_key:
        lines.append("🔑 Using *your own* OpenRouter key — unlimited.")
    else:
        remaining = max(FREE_TIER_LIMIT - free_used, 0)
        lines.append(f"🎁 Free images remaining: *{remaining}* of {FREE_TIER_LIMIT}.")
        if remaining == 0:
            lines.append("Run /setup to add your own OpenRouter key and keep going.")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    deleted = await db.delete_user(update.effective_user.id)
    msg = (
        "🗑️ Deleted your stored key, model choice, and usage count."
        if deleted
        else "Nothing stored for you."
    )
    await update.message.reply_text(msg)


# ── /setup conversation ──────────────────────────────────────────────────────

async def start_setup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "🔑 Send me your *OpenRouter* API key (starts with `sk-or-`).\n\n"
        "Get one at https://openrouter.ai/keys — it's only used to process your "
        "images and is stored encrypted.\n\n"
        "Send /cancel to abort.",
        parse_mode="Markdown",
    )
    return ASK_KEY


async def receive_key(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    key = (update.message.text or "").strip()
    # Best-effort: delete the message containing the key so it doesn't linger.
    try:
        await update.message.delete()
    except Exception:
        pass

    if not key or " " in key:
        await update.message.reply_text(
            "That doesn't look like a key. Paste just the API key, or /cancel."
        )
        return ASK_KEY

    await db.set_api_key(update.effective_user.id, key)
    await update.message.reply_text(
        "✅ Key saved (encrypted).\n\nNow pick a model:\n"
        f"{_model_menu()}\n\nReply with the number, or /cancel to keep the default.",
        parse_mode="Markdown",
    )
    return ASK_MODEL


async def receive_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    choice = (update.message.text or "").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(MODEL_SHORTLIST)):
        await update.message.reply_text(
            f"Reply with a number between 1 and {len(MODEL_SHORTLIST)}, or /cancel."
        )
        return ASK_MODEL
    model = MODEL_SHORTLIST[int(choice) - 1]
    await db.set_model(update.effective_user.id, model)
    await update.message.reply_text(
        f"✅ All set — using `{model}`.\nSend me some images!",
        parse_mode="Markdown",
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Cancelled.")
    return ConversationHandler.END


# ── /model (standalone model picker) ─────────────────────────────────────────

async def choose_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Pick a model:\n"
        f"{_model_menu()}\n\nReply with the number, or /cancel.",
        parse_mode="Markdown",
    )
    return ASK_MODEL


# ── Image processing ─────────────────────────────────────────────────────────

def _pick_extension(mime_type: str) -> str:
    return "png" if "png" in mime_type else "jpg"


async def _download_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Return (bytes, mime_type) for the image in the message, or None."""
    msg = update.message
    if msg.photo:
        # Largest available size.
        tg_file = await msg.photo[-1].get_file()
        data = bytes(await tg_file.download_as_bytearray())
        return data, "image/jpeg"
    if msg.document and (msg.document.mime_type or "").startswith("image/"):
        tg_file = await msg.document.get_file()
        data = bytes(await tg_file.download_as_bytearray())
        return data, msg.document.mime_type
    return None


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    downloaded = await _download_image(update, context)
    if not downloaded:
        return
    image_bytes, mime_type = downloaded

    user = await db.get_user(user_id)
    own_key = user["api_key"] if user else None
    model = (user and user["model"]) or DEFAULT_MODEL

    # Decide which key to use and (for the free tier) reserve a slot.
    used_free_slot = False
    if own_key:
        api_key = own_key
    elif HOST_OPENROUTER_KEY and await db.claim_free_slot(user_id, FREE_TIER_LIMIT):
        api_key = HOST_OPENROUTER_KEY
        used_free_slot = True
    else:
        await update.message.reply_text(
            "🎁 You've used all your free images.\n\n"
            "Run /setup to add your own OpenRouter key and keep removing watermarks.",
        )
        return

    placeholder = await update.message.reply_text("🧼 Cleaning…")
    try:
        cleaned = await remove_overlays(api_key, image_bytes, model, mime_type)
    except ProcessingError as e:
        if used_free_slot:
            await db.release_free_slot(user_id)  # don't burn quota on failures
        await placeholder.edit_text(f"⚠️ Couldn't process that image: {e}")
        return
    except Exception as e:  # pragma: no cover - defensive
        if used_free_slot:
            await db.release_free_slot(user_id)
        logger.exception("Unexpected error processing image: %s", e)
        await placeholder.edit_text("⚠️ Unexpected error processing that image.")
        return

    ext = _pick_extension(mime_type)
    bio = io.BytesIO(cleaned)
    bio.name = f"cleaned.{ext}"
    # Sent as a document to avoid Telegram re-compressing the cleaned result.
    await update.message.reply_document(
        document=bio,
        filename=f"cleaned.{ext}",
        reply_to_message_id=update.message.message_id,
    )
    try:
        await placeholder.delete()
    except Exception:
        pass
