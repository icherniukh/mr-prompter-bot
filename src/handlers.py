import io
import logging
import zipfile

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
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
        "Commands: /status · /settings · /setup · /model · /forget",
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
        "/settings — output format and upscaling options\n"
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


# ── /settings (inline keyboard) ──────────────────────────────────────────────

_OUTPUT_LABELS = {"files": "Files", "zip": "ZIP archive", "photo": "Inline photos"}
_UPSCALE_LABELS = {"original": "Keep original", "low": "Upscale (low)"}


def _settings_keyboard(output_format: str, upscale: str) -> InlineKeyboardMarkup:
    def _btn(label: str, cb: str, current: str, val: str) -> InlineKeyboardButton:
        return InlineKeyboardButton(("✅ " if current == val else "") + label, callback_data=cb)

    return InlineKeyboardMarkup([
        [
            _btn("Files", "sf:files", output_format, "files"),
            _btn("ZIP", "sf:zip", output_format, "zip"),
            _btn("Photos", "sf:photo", output_format, "photo"),
        ],
        [
            _btn("Keep original", "su:original", upscale, "original"),
            _btn("Upscale (low)", "su:low", upscale, "low"),
        ],
    ])


def _settings_text(output_format: str, upscale: str) -> str:
    return (
        "⚙️ *Output Settings*\n\n"
        f"📤 Format: *{_OUTPUT_LABELS[output_format]}*\n"
        f"🔍 Upscaling: *{_UPSCALE_LABELS[upscale]}*\n\n"
        "Tap a button to change:"
    )


async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = await db.get_user(update.effective_user.id)
    output_format = (user and user.get("output_format")) or "files"
    upscale = (user and user.get("upscale")) or "original"
    await update.message.reply_text(
        _settings_text(output_format, upscale),
        parse_mode="Markdown",
        reply_markup=_settings_keyboard(output_format, upscale),
    )


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    data = query.data

    if data.startswith("sf:"):
        fmt = data[3:]
        if fmt not in _OUTPUT_LABELS:
            return
        await db.set_output_format(user_id, fmt)
    elif data.startswith("su:"):
        upscale_val = data[3:]
        if upscale_val not in _UPSCALE_LABELS:
            return
        await db.set_upscale(user_id, upscale_val)
    else:
        return

    user = await db.get_user(user_id)
    output_format = (user and user.get("output_format")) or "files"
    upscale = (user and user.get("upscale")) or "original"
    await query.edit_message_text(
        _settings_text(output_format, upscale),
        parse_mode="Markdown",
        reply_markup=_settings_keyboard(output_format, upscale),
    )


# ── /model (standalone model picker) ─────────────────────────────────────────

async def choose_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Pick a model:\n"
        f"{_model_menu()}\n\nReply with the number, or /cancel.",
        parse_mode="Markdown",
    )
    return ASK_MODEL


# ── ZIP flush job ────────────────────────────────────────────────────────────

async def _flush_zip_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    buf_key = context.job.data["buf_key"]
    chat_id = context.job.data["chat_id"]
    buf = context.bot_data.pop(buf_key, None)
    if not buf:
        return
    results = buf.get("results", [])
    if not results:
        return

    if len(results) == 1:
        img_bytes, ext = results[0]
        bio = io.BytesIO(img_bytes)
        bio.name = f"cleaned.{ext}"
        await context.bot.send_document(chat_id=chat_id, document=bio, filename=f"cleaned.{ext}")
        return

    zf_buf = io.BytesIO()
    with zipfile.ZipFile(zf_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (img_bytes, ext) in enumerate(results, 1):
            zf.writestr(f"cleaned_{i:03d}.{ext}", img_bytes)
    zf_buf.seek(0)
    zf_buf.name = "cleaned_images.zip"
    await context.bot.send_document(chat_id=chat_id, document=zf_buf, filename="cleaned_images.zip")


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
    output_format = (user and user.get("output_format")) or "files"

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

    try:
        await placeholder.delete()
    except Exception:
        pass

    ext = _pick_extension(mime_type)

    if output_format == "zip" and context is not None:
        group_id = getattr(update.message, "media_group_id", None) or str(update.message.message_id)
        buf_key = f"zip_{user_id}_{group_id}"
        buf = context.bot_data.setdefault(buf_key, {"results": [], "chat_id": user_id})
        buf["results"].append((cleaned, ext))
        job_name = f"flush_{buf_key}"
        for job in context.job_queue.get_jobs_by_name(job_name):
            job.schedule_removal()
        context.job_queue.run_once(
            _flush_zip_job,
            2.5,
            data={"buf_key": buf_key, "chat_id": user_id},
            name=job_name,
        )
    elif output_format == "photo":
        await update.message.reply_photo(
            photo=io.BytesIO(cleaned),
            reply_to_message_id=update.message.message_id,
        )
    else:
        bio = io.BytesIO(cleaned)
        bio.name = f"cleaned.{ext}"
        await update.message.reply_document(
            document=bio,
            filename=f"cleaned.{ext}",
            reply_to_message_id=update.message.message_id,
        )
