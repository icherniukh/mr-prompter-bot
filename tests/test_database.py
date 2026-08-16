from pathlib import Path

import aiosqlite
import pytest

from src import database as db


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path) / "bot.db")
    await db.init_db()
    return db


async def test_unknown_user_is_none(fresh_db):
    assert await fresh_db.get_custom_prompt(123) is None
    assert await fresh_db.get_tts_voice(123) is None
    assert await fresh_db.get_user_settings(123) == {
        "custom_prompt": None,
        "output_format": "files",
        "rescale_mode": "none",
        "rescale_width": None,
        "rescale_height": None,
        "tts_voice": None,
    }
    assert await fresh_db.get_roll_max(123) == 10


async def test_tts_voice_roundtrips(fresh_db):
    await fresh_db.set_tts_voice(1, "en-GB-RyanNeural")
    assert await fresh_db.get_tts_voice(1) == "en-GB-RyanNeural"
    settings = await fresh_db.get_user_settings(1)
    assert settings["tts_voice"] == "en-GB-RyanNeural"



async def test_set_custom_prompt_roundtrips(fresh_db):
    await fresh_db.set_custom_prompt(1, "remove billboard watermark only")
    assert await fresh_db.get_custom_prompt(1) == "remove billboard watermark only"


async def test_clear_custom_prompt(fresh_db):
    await fresh_db.set_custom_prompt(1, "custom")
    await fresh_db.clear_custom_prompt(1)
    assert await fresh_db.get_custom_prompt(1) is None


async def test_output_format_roundtrips(fresh_db):
    await fresh_db.set_output_format(1, "zip")
    settings = await fresh_db.get_user_settings(1)
    assert settings["output_format"] == "zip"


async def test_rescale_mode_roundtrips(fresh_db):
    await fresh_db.set_rescale_mode(1, "auto")
    settings = await fresh_db.get_user_settings(1)
    assert settings["rescale_mode"] == "auto"
    assert settings["rescale_width"] is None
    assert settings["rescale_height"] is None


async def test_init_db_adds_settings_columns_for_legacy_table(tmp_path, monkeypatch):
    db_path = Path(tmp_path) / "bot.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()

    await db.init_db()

    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute("PRAGMA table_info(users)") as cur:
            columns = [row[1] async for row in cur]
    assert "custom_prompt" in columns
    assert "output_format" in columns
    assert "rescale_mode" in columns
    assert "rescale_width" in columns
    assert "rescale_height" in columns
    assert "roll_max" in columns
    assert await db.get_roll_max(1) == 10


async def test_init_db_preserves_existing_roll_max_values(tmp_path, monkeypatch):
    db_path = Path(tmp_path) / "bot.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                roll_max INTEGER NOT NULL DEFAULT 10,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.execute("INSERT INTO users (user_id, roll_max) VALUES (1, 10), (2, 11), (3, 60)")
        await conn.commit()

    await db.init_db()

    assert await db.get_roll_max(1) == 10
    assert await db.get_roll_max(2) == 11
    assert await db.get_roll_max(3) == 60


async def test_delete_user(fresh_db):
    await fresh_db.set_custom_prompt(1, "custom")
    assert await fresh_db.delete_user(1) is True
    assert await fresh_db.get_custom_prompt(1) is None
    assert await fresh_db.delete_user(1) is False
