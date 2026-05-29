import asyncio
from pathlib import Path

import pytest

from src import database as db


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path) / "bot.db")
    await db.init_db()
    return db


async def test_unknown_user_is_none(fresh_db):
    assert await fresh_db.get_user(123) is None


async def test_set_key_roundtrips_decrypted(fresh_db):
    await fresh_db.set_api_key(1, "sk-or-secret")
    user = await fresh_db.get_user(1)
    assert user["api_key"] == "sk-or-secret"
    assert user["free_used"] == 0
    assert user["model"] is None


async def test_key_is_encrypted_at_rest(fresh_db):
    await fresh_db.set_api_key(1, "sk-or-secret")
    import aiosqlite

    async with aiosqlite.connect(fresh_db.DB_PATH) as conn:
        async with conn.execute("SELECT api_key FROM users WHERE user_id=1") as cur:
            (stored,) = await cur.fetchone()
    assert "sk-or-secret" not in stored


async def test_set_model(fresh_db):
    await fresh_db.set_model(1, "google/gemini-2.5-flash-image")
    user = await fresh_db.get_user(1)
    assert user["model"] == "google/gemini-2.5-flash-image"


async def test_claim_free_slot_respects_limit(fresh_db):
    granted = [await fresh_db.claim_free_slot(7, limit=3) for _ in range(5)]
    assert granted == [True, True, True, False, False]
    user = await fresh_db.get_user(7)
    assert user["free_used"] == 3


async def test_claim_free_slot_is_atomic_under_concurrency(fresh_db):
    results = await asyncio.gather(
        *[fresh_db.claim_free_slot(9, limit=10) for _ in range(50)]
    )
    # Never grant more than the limit, even with everything racing at once.
    assert sum(results) == 10
    user = await fresh_db.get_user(9)
    assert user["free_used"] == 10


async def test_release_refunds_slot(fresh_db):
    await fresh_db.claim_free_slot(1, limit=2)
    await fresh_db.claim_free_slot(1, limit=2)
    assert (await fresh_db.get_user(1))["free_used"] == 2
    await fresh_db.release_free_slot(1)
    assert (await fresh_db.get_user(1))["free_used"] == 1
    # Can claim again after a refund.
    assert await fresh_db.claim_free_slot(1, limit=2) is True


async def test_release_never_goes_negative(fresh_db):
    await fresh_db.set_api_key(1, "k")
    await fresh_db.release_free_slot(1)
    assert (await fresh_db.get_user(1))["free_used"] == 0


async def test_delete_user(fresh_db):
    await fresh_db.set_api_key(1, "k")
    assert await fresh_db.delete_user(1) is True
    assert await fresh_db.get_user(1) is None
    assert await fresh_db.delete_user(1) is False
