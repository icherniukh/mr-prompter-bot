from pathlib import Path

import pytest

from src import database as db
from src import handlers


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path) / "bot.db")
    await db.init_db()
    return db


class FakeMessage:
    def __init__(self, photo_bytes=None, text=None, caption=None):
        self.photo = [photo_bytes] if photo_bytes else []
        self.document = None
        self.voice = None
        self.audio = None
        self.sticker = None
        self.message_id = 1
        self.chat_id = 1
        self.text = text
        self.caption = caption
        self.replies = []
        self.animations_sent = []
        self.placeholders = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)
        placeholder = _Placeholder()
        self.placeholders.append(placeholder)
        return placeholder

    async def reply_animation(self, animation, **kwargs):
        animation_name = getattr(animation, "name", animation)
        self.animations_sent.append({
            "animation": animation_name,
            "caption": kwargs.get("caption"),
            "reply_to_message_id": kwargs.get("reply_to_message_id"),
        })


class _Placeholder:
    def __init__(self):
        self.edits = []

    async def edit_text(self, text, **kwargs):
        self.edits.append(text)


class FakeUser:
    def __init__(self, uid):
        self.id = uid


class FakeUpdate:
    def __init__(self, uid, photo_bytes=None, text=None, caption=None):
        self.effective_user = FakeUser(uid)
        self.message = FakeMessage(photo_bytes=photo_bytes, text=text, caption=caption)
        self.effective_message = self.message


class FakeContext:
    def __init__(self):
        self.user_data = {}
        self.bot_data = {}


async def test_handle_bullshit_replies_to_text(monkeypatch):
    captured = {}

    async def fake_generate(user_input):
        captured["input"] = user_input
        return "Absolute cosmic nonsense."

    monkeypatch.setattr(handlers, "generate_bullshit", fake_generate)

    upd = FakeUpdate(uid=7, text="What is the weather?")
    await handlers.handle_bullshit(upd, FakeContext())

    assert captured["input"] == "What is the weather?"
    assert upd.message.replies == ["Absolute cosmic nonsense."]


async def test_handle_bullshit_replies_to_photo(monkeypatch):
    captured = {}

    async def fake_generate(user_input):
        captured["input"] = user_input
        return "Photo conspiracy nonsense."

    monkeypatch.setattr(handlers, "generate_bullshit", fake_generate)

    upd = FakeUpdate(uid=7, photo_bytes=b"img")
    await handlers.handle_bullshit(upd, FakeContext())

    assert captured["input"] == "[User sent an image or document]"
    assert upd.message.replies == ["Photo conspiracy nonsense."]


async def test_push_the_horses_initial_warning_and_normal_roll(fresh_db, monkeypatch):
    sleep_calls = []
    seed_calls = []
    trail_markers = []

    async def fake_sleep(secs):
        sleep_calls.append(secs)
        return None

    monkeypatch.setattr(handlers.time, "time", lambda: 10_000.0)
    monkeypatch.setattr(handlers.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(handlers.random, "seed", lambda seed: seed_calls.append(seed))
    monkeypatch.setattr(handlers.random, "randint", lambda _start, _end: 7)
    monkeypatch.setattr(handlers.random, "choice", lambda seq: trail_markers.append(seq[0]) or seq[0])

    upd = FakeUpdate(uid=55, text="/push_the_horses")
    ctx = FakeContext()

    await handlers.push_the_horses(upd, ctx)

    assert upd.message.replies == [
        "Pushing the horses is strictly prohibited in this environment",
        handlers._push_animation_text(handlers._DICE_FRAMES[0], 10, handlers._PUSH_TRAIL_EMOJIS[0], "🎲"),
    ]
    assert ctx.bot_data[handlers._PUSH_COOLDOWN_KEY.format(55)] == 10_000.0
    assert upd.message.placeholders[1].edits == [
        *[
            handlers._push_animation_text(
                frame,
                10,
                handlers._PUSH_TRAIL_EMOJIS[0],
                "🎲" if index < len(handlers._DICE_FRAMES) - 1 else "🎲 7",
            )
            for index, frame in enumerate(handlers._DICE_FRAMES[1:], start=1)
        ],
        handlers._push_result_text(7, 10),
    ]
    assert "✨" in upd.message.replies[1]
    assert upd.message.placeholders[1].edits[-1] == "🎲 7\n🐎 1 lvl · 🎯 10"
    assert len(handlers._DICE_FRAMES) == 6
    assert len(sleep_calls) == len(handlers._DICE_FRAMES) - 1
    assert sum(sleep_calls) == pytest.approx((len(handlers._DICE_FRAMES) - 1) * handlers._PUSH_ANIMATION_SLEEP_SECS)
    assert all(call > 0 for call in sleep_calls)
    assert seed_calls == [10_000]
    assert len(trail_markers) == len(handlers._DICE_FRAMES)
    assert await fresh_db.get_roll_max(55) == 10


async def test_push_the_horses_cooldown_skips_warning_but_still_rolls(fresh_db, monkeypatch):
    sleep_calls = []

    async def fake_sleep(secs):
        sleep_calls.append(secs)
        return None

    monkeypatch.setattr(handlers.time, "time", lambda: 20_000.0)
    monkeypatch.setattr(handlers.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(handlers.random, "seed", lambda _seed: None)
    monkeypatch.setattr(handlers.random, "randint", lambda _start, _end: 9)
    monkeypatch.setattr(handlers.random, "choice", lambda seq: seq[-1])

    upd = FakeUpdate(uid=55, text="/push_the_horses")
    ctx = FakeContext()
    ctx.bot_data[handlers._PUSH_COOLDOWN_KEY.format(55)] = 19_999.0

    await handlers.push_the_horses(upd, ctx)

    assert upd.message.replies == [handlers._push_animation_text(handlers._DICE_FRAMES[0], 10, handlers._PUSH_TRAIL_EMOJIS[-1], "🎲")]
    assert ctx.bot_data[handlers._PUSH_COOLDOWN_KEY.format(55)] == 20_000.0
    assert upd.message.placeholders[0].edits[-1] == handlers._push_result_text(9, 10)
    assert len(sleep_calls) == len(handlers._DICE_FRAMES) - 1
    assert sum(sleep_calls) == pytest.approx((len(handlers._DICE_FRAMES) - 1) * handlers._PUSH_ANIMATION_SLEEP_SECS)
    assert upd.message.placeholders[0].edits[-1] == "🔥 𝟵\n🐎 1 lvl · 🎯 10"


async def test_push_the_horses_jackpot_increments_roll_max(fresh_db, monkeypatch):
    sleep_calls = []
    random_choices = []

    async def fake_sleep(secs):
        sleep_calls.append(secs)
        return None

    monkeypatch.setattr(handlers.time, "time", lambda: 30_000.0)
    monkeypatch.setattr(handlers.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(handlers.random, "seed", lambda _seed: None)
    monkeypatch.setattr(handlers.random, "randint", lambda _start, roll_max: roll_max)
    monkeypatch.setattr(handlers.random, "choice", lambda seq: random_choices.append(seq[1]) or seq[1])

    upd = FakeUpdate(uid=77, text="/push_the_horses")

    assert await fresh_db.get_roll_max(77) == 10

    await handlers.push_the_horses(upd, FakeContext())

    assert upd.message.placeholders[1].edits[-1] == handlers._push_level_up_text(11)
    assert upd.message.placeholders[1].edits[-1] == (
        "🎉 LEVEL UP, YOU ABSOLUTE WEAPON\n"
        "🐎 Rank 2 unlocked\n"
        "🎲 New range: 1-11\n"
        "🇬🇧 Celebration briefing: deeply unnecessary, vaguely heroic."
    )
    assert sum(sleep_calls) == pytest.approx((len(handlers._DICE_FRAMES) - 1) * handlers._PUSH_ANIMATION_SLEEP_SECS)
    assert upd.message.animations_sent == [{
        "animation": str(handlers._PUSH_LEVEL_UP_GIFS[1]),
        "caption": "Properly chuffed. Try not to make a national incident of it.",
        "reply_to_message_id": 1,
    }]
    assert random_choices[-1] == handlers._PUSH_LEVEL_UP_GIFS[1]
    assert await fresh_db.get_roll_max(77) == 11
