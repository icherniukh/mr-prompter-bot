from pathlib import Path

import aiosqlite
import pytest

from src import database as db


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path) / "bot.db")
    await db.init_db()
    return db


async def test_unknown_user_roll_max_default(fresh_db):
    assert await fresh_db.get_roll_max(123) == 10


async def test_increment_roll_max_roundtrips(fresh_db):
    await fresh_db.increment_roll_max(42)
    assert await fresh_db.get_roll_max(42) == 11
    await fresh_db.increment_roll_max(42)
    assert await fresh_db.get_roll_max(42) == 12


async def test_init_db_adds_roll_max_and_strips_legacy_columns(tmp_path, monkeypatch):
    db_path = Path(tmp_path) / "bot.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                custom_prompt TEXT,
                output_format TEXT NOT NULL DEFAULT 'files',
                tts_voice TEXT,
                roll_max INTEGER NOT NULL DEFAULT 10,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.execute(
            "INSERT INTO users (user_id, custom_prompt, roll_max) VALUES (1, 'old', 25)"
        )
        await conn.commit()

    await db.init_db()

    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute("PRAGMA table_info(users)") as cur:
            columns = [row[1] async for row in cur]
    assert set(columns) == {"user_id", "roll_max", "created_at", "updated_at"}
    assert await db.get_roll_max(1) == 25


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
