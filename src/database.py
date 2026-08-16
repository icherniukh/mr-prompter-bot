import os
import aiosqlite
from pathlib import Path

DB_PATH = Path("data/bot.db")
DEFAULT_OUTPUT_FORMAT = "files"
DEFAULT_RESCALE_MODE = "none"
DEFAULT_ROLL_MAX = 10
VALID_OUTPUT_FORMATS = {"zip", "files", "inline"}
VALID_RESCALE_MODES = {"none", "auto"}


def _enforce_db_permissions() -> None:
    """Ensure the database file is readable only by its owner (0600)."""
    try:
        if DB_PATH.exists():
            current = DB_PATH.stat().st_mode & 0o777
            if current != 0o600:
                os.chmod(DB_PATH, 0o600)
    except OSError:
        pass

async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                custom_prompt TEXT,
                output_format TEXT NOT NULL DEFAULT 'files',
                rescale_mode TEXT NOT NULL DEFAULT 'none',
                rescale_width INTEGER,
                rescale_height INTEGER,
                roll_max INTEGER NOT NULL DEFAULT 10,
                tts_voice TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )

        """)
        await _add_column_if_missing(db, "custom_prompt", "TEXT")
        await _add_column_if_missing(db, "output_format", "TEXT NOT NULL DEFAULT 'files'")
        await _add_column_if_missing(db, "rescale_mode", "TEXT NOT NULL DEFAULT 'none'")
        await _add_column_if_missing(db, "rescale_width", "INTEGER")
        await _add_column_if_missing(db, "rescale_height", "INTEGER")
        await _add_column_if_missing(db, "roll_max", f"INTEGER NOT NULL DEFAULT {DEFAULT_ROLL_MAX}")
        await _add_column_if_missing(db, "tts_voice", "TEXT")
        await db.execute(
            "UPDATE users SET roll_max=? WHERE roll_max IS NULL",
            (DEFAULT_ROLL_MAX,),
        )
        await db.commit()
    _enforce_db_permissions()


async def _add_column_if_missing(db: aiosqlite.Connection, name: str, definition: str) -> None:
    async with db.execute("PRAGMA table_info(users)") as cur:
        existing = {row[1] async for row in cur}
    if name not in existing:
        await db.execute(f"ALTER TABLE users ADD COLUMN {name} {definition}")


async def _ensure_user(db: aiosqlite.Connection, user_id: int) -> None:
    await db.execute(
        "INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,)
    )

async def get_custom_prompt(user_id: int) -> str | None:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT custom_prompt FROM users WHERE user_id=?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else None


async def set_custom_prompt(user_id: int, prompt: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_user(db, user_id)
        await db.execute(
            "UPDATE users SET custom_prompt=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
            (prompt, user_id),
        )
        await db.commit()


async def get_tts_voice(user_id: int) -> str | None:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT tts_voice FROM users WHERE user_id=?", (user_id,)
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else None


async def set_tts_voice(user_id: int, voice: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_user(db, user_id)
        await db.execute(
            "UPDATE users SET tts_voice=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
            (voice, user_id),
        )
        await db.commit()


async def get_user_settings(user_id: int) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """
            SELECT custom_prompt, output_format, rescale_mode, rescale_width, rescale_height, tts_voice
            FROM users
            WHERE user_id=?
            """,
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return {
            "custom_prompt": None,
            "output_format": DEFAULT_OUTPUT_FORMAT,
            "rescale_mode": DEFAULT_RESCALE_MODE,
            "rescale_width": None,
            "rescale_height": None,
            "tts_voice": None,
        }

    output_format = row[1] if row[1] in VALID_OUTPUT_FORMATS else DEFAULT_OUTPUT_FORMAT
    rescale_mode = row[2] if row[2] in VALID_RESCALE_MODES else DEFAULT_RESCALE_MODE
    return {
        "custom_prompt": row[0],
        "output_format": output_format,
        "rescale_mode": rescale_mode,
        "rescale_width": row[3],
        "rescale_height": row[4],
        "tts_voice": row[5],
    }



async def set_output_format(user_id: int, output_format: str) -> None:
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}")
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_user(db, user_id)
        await db.execute(
            "UPDATE users SET output_format=?, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
            (output_format, user_id),
        )
        await db.commit()


async def set_rescale_mode(
    user_id: int,
    mode: str,
    width: int | None = None,
    height: int | None = None,
) -> None:
    if mode not in VALID_RESCALE_MODES:
        raise ValueError(f"Unsupported rescale mode: {mode}")
    width = None
    height = None

    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_user(db, user_id)
        await db.execute(
            """
            UPDATE users
            SET rescale_mode=?, rescale_width=?, rescale_height=?, updated_at=CURRENT_TIMESTAMP
            WHERE user_id=?
            """,
            (mode, width, height, user_id),
        )
        await db.commit()


async def clear_custom_prompt(user_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE users SET custom_prompt=NULL, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
            (user_id,),
        )
        await db.commit()


async def get_roll_max(user_id: int) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT roll_max FROM users WHERE user_id=?", (user_id,)) as cur:
            row = await cur.fetchone()
    return row[0] if row and row[0] is not None else DEFAULT_ROLL_MAX


async def increment_roll_max(user_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await _ensure_user(db, user_id)
        await db.execute(
            "UPDATE users SET roll_max = roll_max + 1, updated_at=CURRENT_TIMESTAMP WHERE user_id=?",
            (user_id,),
        )
        await db.commit()


async def delete_user(user_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        await db.commit()
        return cur.rowcount > 0
