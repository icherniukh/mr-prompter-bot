import os
import aiosqlite
from pathlib import Path
from cryptography.fernet import Fernet
from src.config import ENCRYPTION_KEY

DB_PATH = Path("data/bot.db")

_fernet = Fernet(ENCRYPTION_KEY.encode())


def _enforce_db_permissions() -> None:
    """Ensure the database file is readable only by its owner (0600)."""
    try:
        if DB_PATH.exists():
            current = DB_PATH.stat().st_mode & 0o777
            if current != 0o600:
                os.chmod(DB_PATH, 0o600)
    except OSError:
        pass


def _encrypt(text: str) -> str:
    return _fernet.encrypt(text.encode()).decode()


def _decrypt(text: str) -> str:
    return _fernet.decrypt(text.encode()).decode()


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                api_key TEXT,
                model TEXT,
                free_used INTEGER NOT NULL DEFAULT 0,
                output_format TEXT NOT NULL DEFAULT 'files',
                upscale TEXT NOT NULL DEFAULT 'original',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migrations for existing databases.
        for migration in [
            "ALTER TABLE users ADD COLUMN output_format TEXT NOT NULL DEFAULT 'files'",
            "ALTER TABLE users ADD COLUMN upscale TEXT NOT NULL DEFAULT 'original'",
        ]:
            try:
                await db.execute(migration)
            except Exception:
                pass  # column already exists
        await db.commit()
    _enforce_db_permissions()


async def _ensure_user(db: aiosqlite.Connection, user_id: int) -> None:
    await db.execute(
        "INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,)
    )


async def get_user(user_id: int) -> dict | None:
    """Return the user's record (with decrypted key), or None if not present."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT user_id, api_key, model, free_used, output_format, upscale "
            "FROM users WHERE user_id=?",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    user_id, enc_key, model, free_used, output_format, upscale = row
    return {
        "user_id": user_id,
        "api_key": _decrypt(enc_key) if enc_key else None,
        "model": model,
        "free_used": free_used,
        "output_format": output_format or "files",
        "upscale": upscale or "original",
    }


async def set_api_key(user_id: int, api_key: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_user(db, user_id)
        await db.execute(
            "UPDATE users SET api_key=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
            (_encrypt(api_key), user_id),
        )
        await db.commit()


async def set_output_format(user_id: int, fmt: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_user(db, user_id)
        await db.execute(
            "UPDATE users SET output_format=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
            (fmt, user_id),
        )
        await db.commit()


async def set_upscale(user_id: int, upscale: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_user(db, user_id)
        await db.execute(
            "UPDATE users SET upscale=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
            (upscale, user_id),
        )
        await db.commit()


async def set_model(user_id: int, model: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_user(db, user_id)
        await db.execute(
            "UPDATE users SET model=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
            (model, user_id),
        )
        await db.commit()


async def claim_free_slot(user_id: int, limit: int) -> bool:
    """Atomically reserve one free-tier image slot for the user.

    Returns True if a slot was granted (free_used was below the limit and is
    now incremented), False if the user has exhausted their free quota. The
    UPDATE ... WHERE free_used < limit guard makes this safe under the
    concurrency of a batch processed in parallel — no slot is ever double-spent.
    Call release_free_slot() if the work the slot was reserved for fails.
    """
    async with aiosqlite.connect(DB_PATH, isolation_level=None) as db:
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("BEGIN IMMEDIATE")
        try:
            await _ensure_user(db, user_id)
            cur = await db.execute(
                "UPDATE users SET free_used = free_used + 1, updated_at=CURRENT_TIMESTAMP "
                "WHERE user_id=? AND free_used < ?",
                (user_id, limit),
            )
            granted = cur.rowcount > 0
            await db.execute("COMMIT")
        except Exception:
            await db.execute("ROLLBACK")
            raise
    return granted


async def release_free_slot(user_id: int) -> None:
    """Refund a previously reserved free slot (e.g. after a failed image)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET free_used = MAX(free_used - 1, 0), updated_at=CURRENT_TIMESTAMP "
            "WHERE user_id=?",
            (user_id,),
        )
        await db.commit()


async def delete_user(user_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        await db.commit()
        return cur.rowcount > 0
