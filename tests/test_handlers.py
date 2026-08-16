from pathlib import Path
import io
import zipfile

import pytest

from src import database as db
from src import handlers
from src.errors import ProcessingError


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path) / "bot.db")
    await db.init_db()
    return db


class FakeFile:
    def __init__(self, data):
        self._data = data

    async def download_as_bytearray(self):
        return bytearray(self._data)


class FakePhotoSize:
    def __init__(self, data):
        self._data = data

    async def get_file(self):
        return FakeFile(self._data)


class FakeMessage:
    def __init__(self, photo_bytes=None, text=None, media_group_id=None):
        self.photo = [FakePhotoSize(photo_bytes)] if photo_bytes else []
        self.document = None
        self.message_id = 1
        self.chat_id = 1
        self.media_group_id = media_group_id
        self.text = text
        self.replies = []
        self.reply_markups = []
        self.documents_sent = []
        self.photos_sent = []
        self.voices_sent = []
        self.animations_sent = []
        self.placeholders = []
        self.reply_to_message = None

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)
        self.reply_markups.append(kwargs.get("reply_markup"))
        placeholder = _Placeholder()
        self.placeholders.append(placeholder)
        return placeholder

    async def reply_document(self, document, **kwargs):
        self.documents_sent.append(document.read())

    async def reply_photo(self, photo, **kwargs):
        self.photos_sent.append(photo.read())

    async def reply_voice(self, voice, **kwargs):
        self.voices_sent.append(voice.read() if hasattr(voice, "read") else voice)

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

    async def delete(self):
        pass


class FakeUser:
    def __init__(self, uid, username="testuser", full_name="Test User"):
        self.id = uid
        self.username = username
        self.full_name = full_name


class FakeChat:
    def __init__(self, message: "FakeMessage", chat_id: int = 1):
        self.id = chat_id
        self._message = message

    async def send_message(self, text, **kwargs):
        self._message.replies.append(text)
        self._message.reply_markups.append(kwargs.get("reply_markup"))
        placeholder = _Placeholder()
        self._message.placeholders.append(placeholder)
        return placeholder


class FakeUpdate:
    def __init__(self, uid, photo_bytes=None, text=None, media_group_id=None, username="testuser", full_name="Test User", chat_id=1):
        self.effective_user = FakeUser(uid, username, full_name)
        self.message = FakeMessage(
            photo_bytes=photo_bytes,
            text=text,
            media_group_id=media_group_id,
        )
        self.message.chat_id = chat_id
        self.effective_message = self.message
        self.callback_query = None
        self.effective_chat = FakeChat(self.message, chat_id=chat_id)


class FakeCallbackQuery:
    def __init__(self, data):
        self.data = data
        self.answers = 0
        self.edits = []
        self.reply_markups = []

    async def answer(self):
        self.answers += 1

    async def edit_message_text(self, text, **kwargs):
        self.edits.append(text)
        self.reply_markups.append(kwargs.get("reply_markup"))


class FakeCallbackUpdate:
    def __init__(self, uid, data):
        self.effective_user = FakeUser(uid)
        self.message = None
        self.effective_message = None
        self.callback_query = FakeCallbackQuery(data)
        self.effective_chat = None


class FakeBot:
    def __init__(self):
        self.sent_messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent_messages.append({"chat_id": chat_id, "text": text, "kwargs": kwargs})


class FakeContext:
    def __init__(self, args=None):
        self.user_data = {}
        self.bot_data = {}
        self.args = args or []
        self.bot = FakeBot()



async def test_settings_shows_default_prompt_state(fresh_db):
    upd = FakeUpdate(uid=42, text="/settings")
    await handlers.settings(upd, None)
    text = upd.message.replies[-1]
    assert "Output: Files" in text
    assert "Upscaling: Off" in text
    assert "Prompt: default" in text
    assert upd.message.reply_markups[-1] is not None


async def test_settings_prompt_input_saves_user_setting(fresh_db):
    ctx = FakeContext()
    cb = FakeCallbackUpdate(uid=42, data="settings:prompt:set")
    await handlers.settings_callback(cb, ctx)
    assert ctx.user_data["pending_setting"] == "custom_prompt"

    upd = FakeUpdate(uid=42, text="keep all signs untouched")
    await handlers.handle_settings_text(upd, ctx)
    assert await fresh_db.get_custom_prompt(42) == "keep all signs untouched"
    assert "Prompt: custom" in upd.message.replies[-1]
    assert "pending_setting" not in ctx.user_data


async def test_settings_prompt_input_rejects_too_long_prompt(fresh_db):
    ctx = FakeContext()
    ctx.user_data["pending_setting"] = "custom_prompt"
    upd = FakeUpdate(uid=42, text="x" * 2001)
    await handlers.handle_settings_text(upd, ctx)
    assert await fresh_db.get_custom_prompt(42) is None
    assert "too long" in upd.message.replies[-1]
    assert ctx.user_data["pending_setting"] == "custom_prompt"


async def test_settings_prompt_reset_clears_saved_setting(fresh_db):
    await fresh_db.set_custom_prompt(42, "custom")
    upd = FakeCallbackUpdate(uid=42, data="settings:prompt:reset")
    await handlers.settings_callback(upd, FakeContext())
    assert await fresh_db.get_custom_prompt(42) is None
    assert "Prompt: default" in upd.callback_query.edits[-1]


async def test_settings_output_format_callback_persists(fresh_db):
    upd = FakeCallbackUpdate(uid=42, data="settings:output:zip")
    await handlers.settings_callback(upd, FakeContext())
    assert (await fresh_db.get_user_settings(42))["output_format"] == "zip"
    assert "Output: Zip" in upd.callback_query.edits[-1]


async def test_settings_rescale_auto_callback_persists(fresh_db):
    upd = FakeCallbackUpdate(uid=42, data="settings:rescale:auto")
    await handlers.settings_callback(upd, FakeContext())
    assert (await fresh_db.get_user_settings(42))["rescale_mode"] == "auto"
    assert "Upscaling: Auto HD" in upd.callback_query.edits[-1]


async def test_handle_image_uses_custom_prompt(fresh_db, monkeypatch):
    await fresh_db.set_custom_prompt(7, "preserve building names")

    captured = {}

    async def fake_remove(image_bytes, mime_type, prompt):
        captured["image_bytes"] = image_bytes
        captured["mime_type"] = mime_type
        captured["prompt"] = prompt
        return b"CLEANED"

    monkeypatch.setattr(handlers, "remove_overlays_gemini", fake_remove)

    upd = FakeUpdate(uid=7, photo_bytes=b"img")
    await handlers.handle_image(upd, None)
    assert upd.message.documents_sent == [b"CLEANED"]
    assert captured["prompt"] == "preserve building names"


async def test_handle_image_without_custom_prompt_uses_default(fresh_db, monkeypatch):
    captured = {}

    async def fake_remove(image_bytes, mime_type, prompt):
        captured["prompt"] = prompt
        return b"CLEANED"

    monkeypatch.setattr(handlers, "remove_overlays_gemini", fake_remove)

    upd = FakeUpdate(uid=7, photo_bytes=b"img")
    await handlers.handle_image(upd, None)
    assert upd.message.documents_sent == [b"CLEANED"]
    assert captured["prompt"] is None


async def test_handle_image_auto_rescale_runs_local_postprocess(fresh_db, monkeypatch):
    await fresh_db.set_custom_prompt(7, "remove watermark")
    await fresh_db.set_rescale_mode(7, "auto")

    captured = {}

    async def fake_remove(image_bytes, mime_type, prompt):
        captured["prompt"] = prompt
        return b"CLEANED"

    def fake_upscale(image_bytes, mime_type):
        captured["upscale_input"] = (image_bytes, mime_type)
        return b"UPSCALED"

    monkeypatch.setattr(handlers, "remove_overlays_gemini", fake_remove)
    monkeypatch.setattr(handlers, "upscale_to_full_hd_if_needed", fake_upscale)

    upd = FakeUpdate(uid=7, photo_bytes=b"img")
    await handlers.handle_image(upd, None)
    assert captured["prompt"] == "remove watermark"
    assert captured["upscale_input"] == (b"CLEANED", "image/jpeg")
    assert upd.message.documents_sent == [b"UPSCALED"]


async def test_handle_image_sends_inline_photo_when_configured(fresh_db, monkeypatch):
    await fresh_db.set_output_format(7, "inline")

    async def fake_remove(image_bytes, mime_type, prompt):
        return b"CLEANED"

    monkeypatch.setattr(handlers, "remove_overlays_gemini", fake_remove)

    upd = FakeUpdate(uid=7, photo_bytes=b"img")
    await handlers.handle_image(upd, None)
    assert upd.message.photos_sent == [b"CLEANED"]
    assert upd.message.documents_sent == []


async def test_handle_image_sends_zip_when_configured(fresh_db, monkeypatch):
    await fresh_db.set_output_format(7, "zip")

    async def fake_remove(image_bytes, mime_type, prompt):
        return b"CLEANED"

    monkeypatch.setattr(handlers, "remove_overlays_gemini", fake_remove)

    upd = FakeUpdate(uid=7, photo_bytes=b"img")
    await handlers.handle_image(upd, None)
    assert len(upd.message.documents_sent) == 1
    with zipfile.ZipFile(io.BytesIO(upd.message.documents_sent[0])) as zf:
        assert zf.read("cleaned_1.jpg") == b"CLEANED"


async def test_processing_error_is_reported_to_user(fresh_db, monkeypatch):
    async def boom(image_bytes, mime_type, prompt):
        raise ProcessingError("Gemini returned no image.")

    monkeypatch.setattr(handlers, "remove_overlays_gemini", boom)
    upd = FakeUpdate(uid=5, photo_bytes=b"img")
    await handlers.handle_image(upd, None)
    assert upd.message.documents_sent == []
    assert upd.message.placeholders[-1].edits[-1] == "⚠️ Couldn't process that image: Gemini returned no image."


# ── Batch independence / no-collage tests ────────────────────────────────────


async def test_multiple_images_are_processed_as_separate_independent_calls(fresh_db, monkeypatch):
    """When a user sends several images (as Telegram delivers them one message at a time),
    each image must trigger its own call to Gemini.

    The model must never see a 'collage' or multi-image request for one logical user batch.
    """
    calls = []

    async def fake_remove(image_bytes, mime_type, prompt):
        calls.append((image_bytes, mime_type, prompt))
        return b"CLEANED-" + image_bytes[:4]

    monkeypatch.setattr(handlers, "remove_overlays_gemini", fake_remove)

    images = [b"image-one-data", b"image-two-data", b"image-three-data"]

    for img in images:
        upd = FakeUpdate(uid=99, photo_bytes=img)
        await handlers.handle_image(upd, None)
        assert upd.message.documents_sent == [b"CLEANED-" + img[:4]]

    # Exactly 3 independent calls, each with its own distinct image bytes
    assert len(calls) == 3
    assert calls[0][0] == b"image-one-data"
    assert calls[1][0] == b"image-two-data"
    assert calls[2][0] == b"image-three-data"


async def test_even_concurrent_messages_result_in_separate_model_calls(fresh_db, monkeypatch):
    """Even under concurrent message handling, we still get one Gemini call per image.
    """
    import asyncio

    call_count = 0

    async def slow_fake_remove(image_bytes, mime_type, prompt):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)  # simulate real latency
        return b"CLEANED"

    monkeypatch.setattr(handlers, "remove_overlays_gemini", slow_fake_remove)

    # Simulate 5 images arriving "at the same time"
    updates = [FakeUpdate(uid=123, photo_bytes=f"img-{i}".encode()) for i in range(5)]

    await asyncio.gather(*(handlers.handle_image(u, None) for u in updates))

    assert call_count == 5, "Each image message must produce exactly one Gemini call"


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


async def test_support_prompts_for_issue_when_no_args():
    upd = FakeUpdate(uid=123, text="/support")
    ctx = FakeContext()

    await handlers.support(upd, ctx)

    assert ctx.user_data[handlers._PENDING_SETTING_KEY] == "support_issue"
    assert "Please send a message describing the issue" in upd.message.replies[-1]
    assert "@kappa_alive" not in upd.message.replies[-1]


async def test_support_with_args_forwards_immediately(monkeypatch):
    upd = FakeUpdate(uid=123, username="alice", full_name="Alice Smith", chat_id=999, text="/support image failed to process")
    ctx = FakeContext(args=["image", "failed", "to", "process"])

    await handlers.support(upd, ctx)

    assert len(ctx.bot.sent_messages) == 1
    sent = ctx.bot.sent_messages[0]
    assert sent["chat_id"] == "@kappa_alive"
    assert "Alice Smith (@alice)" in sent["text"]
    assert "<b>User ID:</b> <code>123</code>" in sent["text"]
    assert "<b>Chat ID:</b> <code>999</code>" in sent["text"]
    assert "image failed to process" in sent["text"]
    assert "forwarded to support" in upd.message.replies[-1]
    assert "@kappa_alive" not in upd.message.replies[-1]


async def test_support_followup_text_forwards_and_clears_pending():
    upd = FakeUpdate(uid=456, username="bob", full_name="Bob Jones", chat_id=888, text="Here is my issue details")
    ctx = FakeContext()
    ctx.user_data[handlers._PENDING_SETTING_KEY] = "support_issue"

    await handlers.handle_settings_text(upd, ctx)

    assert handlers._PENDING_SETTING_KEY not in ctx.user_data
    assert len(ctx.bot.sent_messages) == 1
    sent = ctx.bot.sent_messages[0]
    assert sent["chat_id"] == "@kappa_alive"
    assert "Bob Jones (@bob)" in sent["text"]
    assert "<b>User ID:</b> <code>456</code>" in sent["text"]
    assert "<b>Chat ID:</b> <code>888</code>" in sent["text"]
    assert "Here is my issue details" in sent["text"]
    assert "forwarded to support" in upd.message.replies[-1]
    assert "@kappa_alive" not in upd.message.replies[-1]


async def test_support_fallback_when_sending_fails(monkeypatch):
    upd = FakeUpdate(uid=789, username="charlie", full_name="Charlie", chat_id=777, text="/support bug report")
    ctx = FakeContext(args=["bug", "report"])

    async def fake_send_message(*args, **kwargs):
        raise RuntimeError("Chat not found")

    monkeypatch.setattr(ctx.bot, "send_message", fake_send_message)

    await handlers.support(upd, ctx)

    assert "Your report has been received and logged" in upd.message.replies[-1]
    assert "@kappa_alive" not in upd.message.replies[-1]


async def test_speak_with_args_generates_and_sends_voice(fresh_db, monkeypatch):
    async def fake_generate_speech(text, voice=None):
        assert text == "Hello from speak command"
        return b"SPEECH_AUDIO_BYTES"

    monkeypatch.setattr(handlers, "generate_speech", fake_generate_speech)

    upd = FakeUpdate(uid=11, text="/speak Hello from speak command")
    ctx = FakeContext(args=["Hello", "from", "speak", "command"])

    await handlers.speak(upd, ctx)

    assert upd.message.voices_sent == [b"SPEECH_AUDIO_BYTES"]


async def test_speak_reply_to_message_generates_voice(fresh_db, monkeypatch):
    async def fake_generate_speech(text, voice=None):
        assert text == "Replied message text"
        return b"REPLIED_AUDIO_BYTES"

    monkeypatch.setattr(handlers, "generate_speech", fake_generate_speech)

    upd = FakeUpdate(uid=11, text="/speak")
    upd.message.reply_to_message = FakeMessage(text="Replied message text")
    ctx = FakeContext()

    await handlers.speak(upd, ctx)

    assert upd.message.voices_sent == [b"REPLIED_AUDIO_BYTES"]


async def test_speak_prompts_when_no_args_and_no_reply():
    upd = FakeUpdate(uid=11, text="/speak")
    ctx = FakeContext()

    await handlers.speak(upd, ctx)

    assert ctx.user_data[handlers._PENDING_SETTING_KEY] == "speak_text"
    assert "Please send the text you'd like me to speak in your next message" in upd.message.replies[-1]



async def test_speak_followup_text_generates_voice(fresh_db, monkeypatch):
    async def fake_generate_speech(text, voice=None):
        assert text == "Follow up text message"
        return b"FOLLOWUP_AUDIO_BYTES"

    monkeypatch.setattr(handlers, "generate_speech", fake_generate_speech)

    upd = FakeUpdate(uid=11, text="Follow up text message")
    ctx = FakeContext()
    ctx.user_data[handlers._PENDING_SETTING_KEY] = "speak_text"

    await handlers.handle_settings_text(upd, ctx)

    assert handlers._PENDING_SETTING_KEY not in ctx.user_data
    assert upd.message.voices_sent == [b"FOLLOWUP_AUDIO_BYTES"]


async def test_speak_failure_handles_error_gracefully(fresh_db, monkeypatch):
    async def fake_generate_speech(text, voice=None):
        raise RuntimeError("TTS service down")

    monkeypatch.setattr(handlers, "generate_speech", fake_generate_speech)

    upd = FakeUpdate(uid=11, text="/speak Hello")
    ctx = FakeContext(args=["Hello"])

    await handlers.speak(upd, ctx)

    assert any("Speech generation failed" in edit for edit in upd.message.placeholders[0].edits)




async def test_cancel_clears_pending_action():
    upd = FakeUpdate(uid=11, text="/cancel")
    ctx = FakeContext()
    ctx.user_data[handlers._PENDING_SETTING_KEY] = "custom_prompt"

    await handlers.cancel(upd, ctx)

    assert handlers._PENDING_SETTING_KEY not in ctx.user_data
    assert "Action cancelled." in upd.message.replies[-1]


async def test_cancel_when_nothing_pending():
    upd = FakeUpdate(uid=11, text="/cancel")
    ctx = FakeContext()

    await handlers.cancel(upd, ctx)

    assert "Nothing to cancel." in upd.message.replies[-1]


async def test_settings_voice_navigation_and_selection(fresh_db):
    ctx = FakeContext()
    cb_nav = FakeCallbackUpdate(uid=42, data="settings:voice:1")
    await handlers.settings_callback(cb_nav, ctx)
    assert "Select Voice" in cb_nav.callback_query.edits[-1]

    cb_set = FakeCallbackUpdate(uid=42, data="settings:voice_set:en-GB-RyanNeural:1")
    await handlers.settings_callback(cb_set, ctx)
    assert await fresh_db.get_tts_voice(42) == "en-GB-RyanNeural"


async def test_settings_voice_preview(fresh_db, monkeypatch):
    called = []

    async def fake_generate_speech(text, voice=None):
        called.append((text, voice))
        return b"PREVIEW_AUDIO_BYTES"

    monkeypatch.setattr(handlers, "generate_speech", fake_generate_speech)

    upd = FakeUpdate(uid=42, text="")
    cb = FakeCallbackUpdate(uid=42, data="settings:voice_prev:en-US-JennyNeural:0")
    cb.effective_message = upd.message

    await handlers.settings_callback(cb, FakeContext())

    assert len(called) == 1
    assert called[0][1] == "en-US-JennyNeural"
    assert upd.message.voices_sent == [b"PREVIEW_AUDIO_BYTES"]



async def test_settings_ukrainian_voice_preview(fresh_db, monkeypatch):
    called = []

    async def fake_generate_speech(text, voice=None):
        called.append((text, voice))
        return b"UKRAINIAN_PREVIEW_AUDIO"

    monkeypatch.setattr(handlers, "generate_speech", fake_generate_speech)

    upd = FakeUpdate(uid=42, text="")
    cb = FakeCallbackUpdate(uid=42, data="settings:voice_prev:uk-UA-OstapNeural:0")
    cb.effective_message = upd.message

    await handlers.settings_callback(cb, FakeContext())

    assert len(called) == 1
    assert called[0][1] == "uk-UA-OstapNeural"
    assert "Привіт! Мене звати Остап" in called[0][0]
    assert upd.message.voices_sent == [b"UKRAINIAN_PREVIEW_AUDIO"]



async def test_speak_uses_user_configured_voice(fresh_db, monkeypatch):
    called = []

    async def fake_generate_speech(text, voice=None):
        called.append((text, voice))
        return b"VOICE_BYTES"

    monkeypatch.setattr(handlers, "generate_speech", fake_generate_speech)
    await fresh_db.set_tts_voice(42, "en-GB-RyanNeural")

    upd = FakeUpdate(uid=42, text="/speak Brilliant performance")
    ctx = FakeContext(args=["Brilliant", "performance"])

    await handlers.speak(upd, ctx)

    assert len(called) == 1
    assert called[0][0] == "Brilliant performance"
    assert called[0][1] == "en-GB-RyanNeural"
    assert upd.message.voices_sent == [b"VOICE_BYTES"]
