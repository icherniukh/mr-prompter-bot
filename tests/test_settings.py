from datetime import date
from pathlib import Path

import pytest

from src import database as db, handlers


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path) / "bot.db")
    await db.init_db()
    return db


# ── DB persistence ─────────────────────────────────────────────────────────────

async def test_output_format_default(fresh_db):
    await fresh_db.set_api_key(1, "sk-or-x")
    user = await fresh_db.get_user(1)
    assert user["output_format"] == "files"
    assert user["upscale"] == "original"


async def test_set_output_format(fresh_db):
    await fresh_db.set_output_format(1, "zip")
    assert (await fresh_db.get_user(1))["output_format"] == "zip"


async def test_set_output_format_photo(fresh_db):
    await fresh_db.set_output_format(1, "photo")
    assert (await fresh_db.get_user(1))["output_format"] == "photo"


async def test_set_output_format_roundtrips(fresh_db):
    await fresh_db.set_output_format(1, "zip")
    await fresh_db.set_output_format(1, "files")
    assert (await fresh_db.get_user(1))["output_format"] == "files"


# ── Settings callback ──────────────────────────────────────────────────────────

class _FakeCQ:
    def __init__(self, data: str):
        self.data = data
        self.answered_text: str | None = None
        self.edited_text: str | None = None

    async def answer(self, text: str | None = None, show_alert: bool = False):
        self.answered_text = text

    async def edit_message_text(self, text: str, **kwargs):
        self.edited_text = text


class _FakeUser:
    def __init__(self, uid: int):
        self.id = uid


class _FakeUpdate:
    def __init__(self, data: str, uid: int):
        self.callback_query = _FakeCQ(data)
        self.effective_user = _FakeUser(uid)


async def test_callback_changes_output_format(fresh_db):
    upd = _FakeUpdate("sf:zip", uid=1)
    await handlers.settings_callback(upd, None)
    assert (await fresh_db.get_user(1))["output_format"] == "zip"
    assert upd.callback_query.edited_text is not None


async def test_callback_rejects_unknown_format(fresh_db):
    upd = _FakeUpdate("sf:unknown", uid=1)
    await handlers.settings_callback(upd, None)
    assert await fresh_db.get_user(1) is None  # no DB write


async def test_callback_upscale_coming_soon_shows_toast(fresh_db):
    upd = _FakeUpdate("soon", uid=1)
    await handlers.settings_callback(upd, None)
    assert "soon" in (upd.callback_query.answered_text or "").lower()


async def test_callback_upscale_coming_soon_does_not_edit_message(fresh_db):
    upd = _FakeUpdate("soon", uid=1)
    await handlers.settings_callback(upd, None)
    assert upd.callback_query.edited_text is None


async def test_callback_upscale_coming_soon_does_not_save_preference(fresh_db):
    upd = _FakeUpdate("soon", uid=1)
    await handlers.settings_callback(upd, None)
    user = await fresh_db.get_user(1)
    assert user is None or user["upscale"] == "original"


async def test_callback_su_original_saves_and_refreshes(fresh_db):
    upd = _FakeUpdate("su:original", uid=1)
    await handlers.settings_callback(upd, None)
    assert (await fresh_db.get_user(1))["upscale"] == "original"
    assert upd.callback_query.edited_text is not None


# ── ZIP progressive naming ─────────────────────────────────────────────────────

_TODAY = date(2026, 5, 30)
_TOMORROW = date(2026, 5, 31)


def test_first_zip_is_base_name():
    bd = {}
    assert handlers._zip_filename(bd, user_id=1, today=_TODAY) == "cleaned_may30.zip"


def test_second_zip_gets_suffix_1():
    bd = {}
    handlers._zip_filename(bd, user_id=1, today=_TODAY)
    assert handlers._zip_filename(bd, user_id=1, today=_TODAY) == "cleaned_may30_1.zip"


def test_third_zip_gets_suffix_2():
    bd = {}
    for _ in range(2):
        handlers._zip_filename(bd, user_id=1, today=_TODAY)
    assert handlers._zip_filename(bd, user_id=1, today=_TODAY) == "cleaned_may30_2.zip"


def test_counter_is_per_user():
    bd = {}
    assert handlers._zip_filename(bd, user_id=1, today=_TODAY) == "cleaned_may30.zip"
    assert handlers._zip_filename(bd, user_id=2, today=_TODAY) == "cleaned_may30.zip"


def test_counter_resets_each_day():
    bd = {}
    handlers._zip_filename(bd, user_id=1, today=_TODAY)
    handlers._zip_filename(bd, user_id=1, today=_TODAY)
    assert handlers._zip_filename(bd, user_id=1, today=_TOMORROW) == "cleaned_may31.zip"


def test_filename_month_abbreviation():
    bd = {}
    assert handlers._zip_filename(bd, user_id=1, today=date(2026, 1, 5)) == "cleaned_jan05.zip"
    assert handlers._zip_filename(bd, user_id=1, today=date(2026, 12, 31)) == "cleaned_dec31.zip"
