import asyncio
import io
import logging
from pathlib import Path
import random
import time
import zipfile

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaDocument, InputMediaPhoto, Message, Update
from telegram.ext import ContextTypes

from src import database as db
from src.config import GEMINI_MODEL, SUPPORT_ACCOUNT, SUPPORT_CHAT_ID, TTS_VOICE
from src.errors import ProcessingError

from src.gemini_engine import remove_overlays_gemini
from src.image_postprocess import upscale_to_full_hd_if_needed
from src.tts_engine import generate_speech
from src.voices import CURATED_VOICES, get_voice_by_id


logger = logging.getLogger(__name__)


# Seconds to wait after the first image in a group before processing the batch.
# Telegram delivers all album messages within ~500 ms; 1.5 s gives ample margin.
_BATCH_WAIT = 1.5
_MAX_CUSTOM_PROMPT_LENGTH = 2000
_PENDING_SETTING_KEY = "pending_setting"
_PUSH_COOLDOWN_SECS = 15 * 60
_PUSH_COOLDOWN_KEY = "push_last_{}"
_PUSH_ANIMATION_SLEEP_SECS = 0.3
_PUSH_MIN_ROLL = 1
_PUSH_TRAIL_EMOJIS = ["✨", "💫", "🌈", "⭐", "🫧", "⚡"]
_PUSH_START_ROLL_MAX = db.DEFAULT_ROLL_MAX
_DICE_FRAMES = [
    "🟥⬜⬜⬜⬜⬜  🐎",
    "🟥🟧⬜⬜⬜⬜  🐎",
    "🟥🟧🟨⬜⬜⬜  🐎",
    "🟥🟧🟨🟩⬜⬜  🐎💨",
    "🟥🟧🟨🟩🟦⬜  🐎💨",
    "🟥🟧🟨🟩🟦🟪  🐎💨✨",
]
_PUSH_LEVEL_UP_GIFS = [
    Path("assets/push_the_horses/bean-steady-on.gif"),
    Path("assets/push_the_horses/blackadder-schemes.gif"),
    Path("assets/push_the_horses/properly-chuffed.gif"),
    Path("assets/push_the_horses/national-incident.gif"),
]
_OUTPUT_LABELS = {
    "zip": "Zip",
    "files": "Files",
    "inline": "Inline",
}
_RESCALE_LABELS = {
    "none": "Off",
    "auto": "Auto HD",
}


def _pick_extension(mime_type: str) -> str:
    return "png" if "png" in mime_type else "jpg"


def _push_rank(roll_max: int) -> int:
    return max(1, roll_max - _PUSH_START_ROLL_MAX + 1)


def _style_roll_number(result: int, roll_max: int) -> str:
    percent = result / max(1, roll_max)
    raw = str(result)
    if percent <= 0.25:
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return raw.translate(superscript)
    if percent >= 0.9:
        bold = str.maketrans("0123456789", "𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵")
        return raw.translate(bold)
    return raw


def _push_number_effect(result: int, roll_max: int) -> str:
    if result == roll_max:
        return "🌈"
    if result >= max(_PUSH_MIN_ROLL, roll_max - 2):
        return "🔥"
    if result >= max(_PUSH_MIN_ROLL, int(roll_max * 0.85)):
        return "✨"
    return "🎲"


def _push_animation_text(frame: str, roll_max: int, marker: str, display: str) -> str:
    return f"{frame} {marker}\n{display}"


def _push_result_text(result: int, roll_max: int) -> str:
    styled = _style_roll_number(result, roll_max)
    effect = _push_number_effect(result, roll_max)
    if result == roll_max:
        next_rank = _push_rank(roll_max + 1)
        return f"{effect} {styled}\n🐎 {next_rank} lvl · 🎲 {_PUSH_MIN_ROLL}-{roll_max + 1}"
    return f"{effect} {styled}\n🐎 {_push_rank(roll_max)} lvl · 🎯 {roll_max}"


def _push_level_up_text(new_roll_max: int) -> str:
    new_rank = _push_rank(new_roll_max)
    return (
        f"🎉 LEVEL UP, YOU ABSOLUTE WEAPON\n"
        f"🐎 Rank {new_rank} unlocked\n"
        f"🎲 New range: {_PUSH_MIN_ROLL}-{new_roll_max}\n"
        "🇬🇧 Celebration briefing: deeply unnecessary, vaguely heroic."
    )


def _named_image(cleaned_bytes: bytes, mime_type: str, index: int | None = None) -> io.BytesIO:
    ext = _pick_extension(mime_type)
    name = f"cleaned_{index}.{ext}" if index is not None else f"cleaned.{ext}"
    bio = io.BytesIO(cleaned_bytes)
    bio.name = name
    return bio


def _zip_results(results: list[tuple[bytes, str]]) -> io.BytesIO:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, (cleaned_bytes, mime_type) in enumerate(results, start=1):
            ext = _pick_extension(mime_type)
            zf.writestr(f"cleaned_{index}.{ext}", cleaned_bytes)
    archive.seek(0)
    archive.name = "cleaned_images.zip"
    return archive


async def _download_from_message(msg: Message) -> tuple[bytes, str] | None:
    """Download image bytes from a photo or image document message."""
    if msg.photo:
        tg_file = await msg.photo[-1].get_file()
        return bytes(await tg_file.download_as_bytearray()), "image/jpeg"
    if msg.document and (msg.document.mime_type or "").startswith("image/"):
        tg_file = await msg.document.get_file()
        return bytes(await tg_file.download_as_bytearray()), msg.document.mime_type
    return None


async def _clean_image(image_bytes: bytes, mime_type: str, user_id: int) -> bytes:
    """Route one image through Gemini. Raises ProcessingError on failure."""
    settings = await db.get_user_settings(user_id)
    cleaned = await remove_overlays_gemini(image_bytes, mime_type, settings["custom_prompt"])
    if settings["rescale_mode"] == "auto":
        return await asyncio.to_thread(upscale_to_full_hd_if_needed, cleaned, mime_type)
    return cleaned


_VOICES_PER_PAGE = 5


def _settings_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Output format", callback_data="settings:output")],
        [InlineKeyboardButton("Upscaling", callback_data="settings:rescale")],
        [InlineKeyboardButton("Prompt config", callback_data="settings:prompt")],
        [InlineKeyboardButton("🎙️ Voice config", callback_data="settings:voice:0")],
        [InlineKeyboardButton("✕ Close", callback_data="settings:close")],
    ])


def _voice_keyboard(current_voice: str, page: int = 0) -> InlineKeyboardMarkup:
    total_voices = len(CURATED_VOICES)
    total_pages = (total_voices + _VOICES_PER_PAGE - 1) // _VOICES_PER_PAGE
    page = max(0, min(page, total_pages - 1))

    start = page * _VOICES_PER_PAGE
    end = min(start + _VOICES_PER_PAGE, total_voices)
    page_voices = CURATED_VOICES[start:end]

    buttons = []
    for v in page_voices:
        is_selected = (v.id == current_voice)
        prefix = "✅ " if is_selected else ""
        gender_icon = "👨" if v.gender == "Male" else "👩"
        voice_label = f"{prefix}{v.flag} {v.name} ({gender_icon} {v.country})"
        buttons.append([
            InlineKeyboardButton(voice_label, callback_data=f"settings:voice_set:{v.id}:{page}"),
            InlineKeyboardButton("▶ Sample", callback_data=f"settings:voice_prev:{v.id}:{page}"),
        ])

    # Navigation row
    nav_row = []
    if page > 0:
        nav_row.append(InlineKeyboardButton("◀ Prev", callback_data=f"settings:voice:{page - 1}"))
    else:
        nav_row.append(InlineKeyboardButton("·", callback_data="settings:noop"))

    nav_row.append(InlineKeyboardButton(f"{page + 1} / {total_pages}", callback_data="settings:noop"))

    if page < total_pages - 1:
        nav_row.append(InlineKeyboardButton("Next ▶", callback_data=f"settings:voice:{page + 1}"))
    else:
        nav_row.append(InlineKeyboardButton("·", callback_data="settings:noop"))

    buttons.append(nav_row)
    buttons.append([InlineKeyboardButton("← Back to Settings", callback_data="settings:main")])
    return InlineKeyboardMarkup(buttons)


def _prompt_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Set custom", callback_data="settings:prompt:set")],
        [InlineKeyboardButton("Reset", callback_data="settings:prompt:reset")],
        [InlineKeyboardButton("← Back", callback_data="settings:main")],
    ])


def _output_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Zip", callback_data="settings:output:zip")],
        [InlineKeyboardButton("Files", callback_data="settings:output:files")],
        [InlineKeyboardButton("Inline", callback_data="settings:output:inline")],
        [InlineKeyboardButton("← Back", callback_data="settings:main")],
    ])


def _rescale_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Off", callback_data="settings:rescale:none")],
        [InlineKeyboardButton("Auto HD", callback_data="settings:rescale:auto")],
        [InlineKeyboardButton("← Back", callback_data="settings:main")],
    ])


async def _settings_text(user_id: int) -> str:
    s = await db.get_user_settings(user_id)
    prompt_val = "custom" if s["custom_prompt"] else "default"
    voice_id = s.get("tts_voice") or TTS_VOICE
    voice_obj = get_voice_by_id(voice_id)
    voice_label = voice_obj.label if voice_obj else voice_id
    lines = [
        "Settings",
        "",
        f"Output: {_OUTPUT_LABELS[s['output_format']]}",
        f"Upscaling: {_RESCALE_LABELS[s['rescale_mode']]}",
        f"Prompt: {prompt_val}",
        f"Voice: {voice_label}",
    ]
    return "\n".join(lines)


async def _edit_or_reply(update: Update, text: str, reply_markup=None) -> None:
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup)
    else:
        await update.effective_chat.send_message(text, reply_markup=reply_markup)


async def _send_single_result(msg: Message, cleaned: bytes, mime_type: str, user_id: int) -> None:
    settings = await db.get_user_settings(user_id)
    output_format = settings["output_format"]
    if output_format == "zip":
        await msg.reply_document(
            document=_zip_results([(cleaned, mime_type)]),
            filename="cleaned_images.zip",
            reply_to_message_id=msg.message_id,
        )
        return

    bio = _named_image(cleaned, mime_type)
    if output_format == "inline":
        await msg.reply_photo(
            photo=bio,
            reply_to_message_id=msg.message_id,
        )
        return

    await msg.reply_document(
        document=bio,
        filename=bio.name,
        reply_to_message_id=msg.message_id,
    )


async def _send_batch_results(bot, chat_id: int, results: list[tuple[bytes, str]], user_id: int) -> None:
    settings = await db.get_user_settings(user_id)
    output_format = settings["output_format"]
    if output_format == "zip":
        await bot.send_document(chat_id, document=_zip_results(results), filename="cleaned_images.zip")
        return

    media_items = []
    for index, (cleaned_bytes, mime_type) in enumerate(results, start=1):
        bio = _named_image(cleaned_bytes, mime_type, index)
        if output_format == "inline":
            media_items.append(InputMediaPhoto(media=bio))
        else:
            media_items.append(InputMediaDocument(media=bio, filename=bio.name))

    # Telegram media groups are capped at 10 items.
    for chunk_start in range(0, len(media_items), 10):
        await bot.send_media_group(chat_id, media_items[chunk_start:chunk_start + 10])


# ── Static commands ────────────────────────────────────────────────────────

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _edit_or_reply(
        update,
        await _settings_text(update.effective_user.id),
        reply_markup=_settings_keyboard(),
    )


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = update.effective_user.id

    if data == "settings:noop":
        return
    if data == "settings:close":
        await query.message.delete()
        return
    if data == "settings:main":
        await settings(update, context)
        return
    if data == "settings:output":
        await query.edit_message_text("Output format", reply_markup=_output_keyboard())
        return
    if data.startswith("settings:output:"):
        await db.set_output_format(user_id, data.rsplit(":", 1)[1])
        await query.edit_message_text(await _settings_text(user_id), reply_markup=_settings_keyboard())
        return
    if data == "settings:rescale":
        await query.edit_message_text("Upscaling", reply_markup=_rescale_keyboard())
        return
    if data in {"settings:rescale:none", "settings:rescale:auto"}:
        await db.set_rescale_mode(user_id, data.rsplit(":", 1)[1])
        await query.edit_message_text(await _settings_text(user_id), reply_markup=_settings_keyboard())
        return
    if data == "settings:prompt":
        await query.edit_message_text("Prompt", reply_markup=_prompt_keyboard())
        return
    if data == "settings:prompt:set":
        context.user_data[_PENDING_SETTING_KEY] = "custom_prompt"
        await query.edit_message_text(
            f"Type your prompt. Max {_MAX_CUSTOM_PROMPT_LENGTH} chars."
        )
        return
    if data == "settings:prompt:reset":
        await db.clear_custom_prompt(user_id)
        await query.edit_message_text(await _settings_text(user_id), reply_markup=_settings_keyboard())
        return
    if data.startswith("settings:voice:"):
        page = int(data.split(":")[2])
        cur_voice = await db.get_tts_voice(user_id) or TTS_VOICE
        await query.edit_message_text(
            "Select Voice:\nTap a voice to choose it, or tap Sample to preview.",
            reply_markup=_voice_keyboard(cur_voice, page),
        )
        return
    if data.startswith("settings:voice_set:"):
        _, _, voice_id, page_str = data.split(":")
        page = int(page_str)
        await db.set_tts_voice(user_id, voice_id)
        await query.edit_message_text(
            "Select Voice:\nTap a voice to choose it, or tap Sample to preview.",
            reply_markup=_voice_keyboard(voice_id, page),
        )
        return
    if data.startswith("settings:voice_prev:"):
        _, _, voice_id, page_str = data.split(":")
        v = get_voice_by_id(voice_id)
        name = v.name if v else "this voice"
        country = v.country if v else ""
        if voice_id.startswith("uk-"):
            preview_text = f"Привіт! Мене звати {name}, і я можу озвучувати ваші повідомлення."
        else:
            preview_text = f"Hello! I am {name}, speaking with a {country} accent."
        try:
            audio_bytes = await generate_speech(preview_text, voice=voice_id)

            bio = io.BytesIO(audio_bytes)
            bio.name = f"sample_{voice_id}.mp3"
            await update.effective_message.reply_voice(
                voice=bio,
                caption=f"🎙️ Sample: {v.label if v else voice_id}",
            )
        except Exception as e:
            logger.exception("Failed to generate voice preview for %s: %s", voice_id, e)
        return


async def _forward_support_report(update: Update, context: ContextTypes.DEFAULT_TYPE, report_text: str) -> None:
    user = update.effective_user
    chat = update.effective_chat
    user_id = user.id if user else "Unknown"
    username = f"@{user.username}" if user and user.username else "No username"
    full_name = user.full_name if user else "Unknown"
    chat_id = chat.id if chat else "Unknown"

    support_msg = (
        f"📩 <b>New Support Report</b>\n"
        f"<b>From:</b> {full_name} ({username})\n"
        f"<b>User ID:</b> <code>{user_id}</code>\n"
        f"<b>Chat ID:</b> <code>{chat_id}</code>\n\n"
        f"<b>Message:</b>\n{report_text}"
    )

    try:
        Path("data/logs").mkdir(parents=True, exist_ok=True)
        with open("data/logs/support.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | User {user_id} ({username}): {report_text}\n")
    except Exception as e:
        logger.warning("Failed to write to support log: %s", e)

    target_chat = SUPPORT_CHAT_ID or SUPPORT_ACCOUNT
    sent = False
    if target_chat:
        try:
            await context.bot.send_message(
                chat_id=target_chat,
                text=support_msg,
                parse_mode="HTML",
            )
            sent = True
        except Exception as e:
            logger.warning("Could not forward support message to %s: %s", target_chat, e)

    if sent:
        await update.effective_message.reply_text(
            "✅ Thank you! Your report has been forwarded to support."
        )
    else:
        await update.effective_message.reply_text(
            "✅ Thank you! Your report has been received and logged."
        )


async def support(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        report_text = " ".join(context.args).strip()
        await _forward_support_report(update, context, report_text)
        return

    context.user_data[_PENDING_SETTING_KEY] = "support_issue"
    await update.effective_message.reply_text(
        "Please send a message describing the issue or feedback you'd like to report."
    )


async def _send_speech_voice(update: Update, text_to_speak: str) -> None:
    msg = update.effective_message
    user_id = update.effective_user.id if update.effective_user else 0
    if not text_to_speak:
        await msg.reply_text("Please provide some text for speech synthesis.")
        return

    if len(text_to_speak) > _MAX_CUSTOM_PROMPT_LENGTH:
        await msg.reply_text(f"Text is too long. Keep it under {_MAX_CUSTOM_PROMPT_LENGTH} characters.")
        return

    progress = await msg.reply_text("🎙️ Generating speech…")
    try:
        user_voice = (await db.get_tts_voice(user_id)) or TTS_VOICE
        audio_bytes = await generate_speech(text_to_speak, voice=user_voice)
        bio = io.BytesIO(audio_bytes)
        bio.name = "voice.mp3"
        await msg.reply_voice(voice=bio, reply_to_message_id=msg.message_id)
        try:
            await progress.delete()
        except Exception:
            pass
    except Exception as e:
        logger.exception("Failed to generate speech: %s", e)
        await progress.edit_text(f"⚠️ Speech generation failed: {e}")



async def speak(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # 1. If arguments provided: /speak hello world
    if context.args:
        text_to_speak = " ".join(context.args).strip()
        await _send_speech_voice(update, text_to_speak)
        return

    # 2. If replying to a text message:
    reply = update.effective_message.reply_to_message if update.effective_message else None
    if reply and reply.text:
        await _send_speech_voice(update, reply.text.strip())
        return

    # 3. Interactive prompt for the next message
    context.user_data[_PENDING_SETTING_KEY] = "speak_text"
    await update.effective_message.reply_text(
        "Please send the text you'd like me to speak in your next message. (Send /cancel to abort)"
    )



async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    pending = context.user_data.pop(_PENDING_SETTING_KEY, None)
    if pending:
        await update.effective_message.reply_text("Action cancelled.")
    else:
        await update.effective_message.reply_text("Nothing to cancel.")


async def handle_settings_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    pending = context.user_data.get(_PENDING_SETTING_KEY)
    if not pending:
        return

    user_id = update.effective_user.id
    text = (update.message.text or "").strip()
    if pending == "custom_prompt":
        if not text:
            await update.message.reply_text("Custom prompt cannot be empty. Open /settings to try again.")
            context.user_data.pop(_PENDING_SETTING_KEY, None)
            return
        if len(text) > _MAX_CUSTOM_PROMPT_LENGTH:
            await update.message.reply_text(
                f"Custom prompt is too long. Keep it under {_MAX_CUSTOM_PROMPT_LENGTH} characters."
            )
            return
        await db.set_custom_prompt(user_id, text)
        context.user_data.pop(_PENDING_SETTING_KEY, None)
        await update.message.reply_text(
            await _settings_text(user_id),
            reply_markup=_settings_keyboard(),
        )
        return

    if pending == "support_issue":
        if not text:
            await update.message.reply_text("Issue description cannot be empty. Send /support to try again.")
            context.user_data.pop(_PENDING_SETTING_KEY, None)
            return
        await _forward_support_report(update, context, text)
        context.user_data.pop(_PENDING_SETTING_KEY, None)
        return

    if pending == "speak_text":
        if not text:
            await update.message.reply_text("Text cannot be empty. Send /speak to try again.")
            context.user_data.pop(_PENDING_SETTING_KEY, None)
            return
        context.user_data.pop(_PENDING_SETTING_KEY, None)
        await _send_speech_voice(update, text)
        return





async def forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("Yes, delete", callback_data="forget:confirm"),
        InlineKeyboardButton("Cancel", callback_data="forget:cancel"),
    ]])
    await update.message.reply_text("Delete all your stored data?", reply_markup=keyboard)


async def forget_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if query.data == "forget:confirm":
        deleted = await db.delete_user(update.effective_user.id)
        await query.edit_message_text("Deleted." if deleted else "Nothing stored.")
    else:
        await query.message.delete()


async def push_the_horses(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    now = time.time()
    roll_max = await db.get_roll_max(user_id)

    cooldown_key = _PUSH_COOLDOWN_KEY.format(user_id)
    last = context.bot_data.get(cooldown_key, 0)
    if now - last > _PUSH_COOLDOWN_SECS:
        await update.message.reply_text(
            "Pushing the horses is strictly prohibited in this environment"
        )
    context.bot_data[cooldown_key] = now

    random.seed(int(now))
    result = random.randint(_PUSH_MIN_ROLL, roll_max)

    marker = random.choice(_PUSH_TRAIL_EMOJIS)
    msg = await update.message.reply_text(
        _push_animation_text(_DICE_FRAMES[0], roll_max, marker, "🎲")
    )
    for index, frame in enumerate(_DICE_FRAMES[1:], start=1):
        await asyncio.sleep(_PUSH_ANIMATION_SLEEP_SECS)
        try:
            reveal = "🎲"
            if index == len(_DICE_FRAMES) - 1:
                reveal = _style_roll_number(result, roll_max)
                reveal = f"{_push_number_effect(result, roll_max)} {reveal}"
            await msg.edit_text(_push_animation_text(frame, roll_max, random.choice(_PUSH_TRAIL_EMOJIS), reveal))
        except Exception:
            pass

    if result == roll_max:
        await db.increment_roll_max(user_id)
        new_roll_max = roll_max + 1
        await msg.edit_text(_push_level_up_text(new_roll_max))
        gif_path = random.choice(_PUSH_LEVEL_UP_GIFS)
        with gif_path.open("rb") as animation_file:
            await update.message.reply_animation(
                animation=animation_file,
                caption="Properly chuffed. Try not to make a national incident of it.",
                reply_to_message_id=update.message.message_id,
            )
        return
    await msg.edit_text(_push_result_text(result, roll_max))


# ── Image processing ──────────────────────────────────────────────────────────

async def _process_batch(bot, bot_data: dict, key: tuple, user_id: int, chat_id: int) -> None:
    """Process all buffered images for one media group after the collection window."""
    await asyncio.sleep(_BATCH_WAIT)
    messages: list[Message] = bot_data.pop(key, [])
    bot_data.pop(f"{key}_task", None)
    if not messages:
        return

    n = len(messages)
    progress = await bot.send_message(chat_id, f"🧼 Cleaning… 0/{n}")

    results: list[tuple[bytes, str] | None] = []
    for i, msg in enumerate(messages):
        downloaded = await _download_from_message(msg)
        if downloaded is None:
            results.append(None)
        else:
            image_bytes, mime_type = downloaded
            try:
                cleaned = await _clean_image(image_bytes, mime_type, user_id)
                results.append((cleaned, mime_type))
            except ProcessingError as e:
                logger.warning("Batch image %d/%d failed: %s", i + 1, n, e)
                results.append(None)
            except Exception as e:
                logger.exception("Unexpected error on batch image %d/%d: %s", i + 1, n, e)
                results.append(None)

        try:
            await progress.edit_text(f"🧼 Cleaning… {i + 1}/{n}")
        except Exception:
            pass

    successful = [result for result in results if result is not None]
    if successful:
        await _send_batch_results(bot, chat_id, successful, user_id)

    failed = n - len(successful)
    if failed:
        await bot.send_message(chat_id, f"⚠️ {failed}/{n} images failed to process.")

    try:
        await progress.delete()
    except Exception:
        pass


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    msg = update.message

    if msg.media_group_id:
        # Buffer all messages in this album; fire processing once all have arrived.
        key = (msg.chat_id, msg.media_group_id)
        if key not in context.bot_data:
            context.bot_data[key] = []
        context.bot_data[key].append(msg)

        # Only create one task per group — first message wins.
        task_key = f"{key}_task"
        if task_key not in context.bot_data:
            task = asyncio.create_task(
                _process_batch(context.bot, context.bot_data, key, user_id, msg.chat_id)
            )
            context.bot_data[task_key] = task
        return

    # Single image — process immediately with a live progress message.
    downloaded = await _download_from_message(msg)
    if not downloaded:
        return
    image_bytes, mime_type = downloaded

    progress = await msg.reply_text("🧼 Cleaning…")
    try:
        cleaned = await _clean_image(image_bytes, mime_type, user_id)
    except ProcessingError as e:
        await progress.edit_text(f"⚠️ Couldn't process that image: {e}")
        return
    except Exception as e:
        logger.exception("Unexpected error processing image: %s", e)
        await progress.edit_text("⚠️ Unexpected error processing that image.")
        return

    await _send_single_result(msg, cleaned, mime_type, user_id)
    try:
        await progress.delete()
    except Exception:
        pass
