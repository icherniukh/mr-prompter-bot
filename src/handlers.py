import asyncio
import logging
from pathlib import Path
import random
import time

from telegram import Message, Update
from telegram.ext import ContextTypes

from src import database as db
from src.bullshit_engine import generate_bullshit

logger = logging.getLogger(__name__)

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


def _extract_user_input(msg: Message) -> str:
    text = (getattr(msg, "text", None) or "").strip()
    if text:
        return text
    caption = (getattr(msg, "caption", None) or "").strip()
    if caption:
        return f"[User sent media with caption: {caption}]"
    if getattr(msg, "photo", None) or getattr(msg, "document", None):
        return "[User sent an image or document]"
    if getattr(msg, "voice", None) or getattr(msg, "audio", None):
        return "[User sent a voice or audio message]"
    if getattr(msg, "sticker", None):
        return "[User sent a sticker]"
    return "[User sent an unintelligible transmission]"


async def handle_bullshit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message or update.message
    if msg is None:
        return
    user_input = _extract_user_input(msg)
    reply = await generate_bullshit(user_input)
    await msg.reply_text(reply)


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
