import os
from pathlib import Path
import aiosqlite

DB_PATH = Path("data/bot.db")
DEFAULT_ROLL_MAX = 10
_EXPECTED_COLUMNS = {"user_id", "roll_max", "created_at", "updated_at"}


def _enforce_db_permissions() -> None:
    """Ensure the database file is readable only by its owner (0600)."""
    try:
        if DB_PATH.exists():
            current = DB_PATH.stat().st_mode & 0o777
            if current != 0o600:
                os.chmod(DB_PATH, 0o600)
    except OSError:
        pass


async def _add_column_if_missing(db: aiosqlite.Connection, name: str, definition: str) -> None:
    async with db.execute("PRAGMA table_info(users)") as cur:
        existing = {row[1] async for row in cur}
    if name not in existing:
        await db.execute(f"ALTER TABLE users ADD COLUMN {name} {definition}")


async def _rebuild_clean_users_table_if_needed(db: aiosqlite.Connection) -> None:
    async with db.execute("PRAGMA table_info(users)") as cur:
        existing = {row[1] async for row in cur}
    if existing <= _EXPECTED_COLUMNS:
        return

    await db.execute(f"""
        CREATE TABLE users_new (
            user_id INTEGER PRIMARY KEY,
            roll_max INTEGER NOT NULL DEFAULT {DEFAULT_ROLL_MAX},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await db.execute(
        """
        INSERT INTO users_new (user_id, roll_max, created_at, updated_at)
        SELECT user_id, COALESCE(roll_max, ?), created_at, updated_at
        FROM users
        """,
        (DEFAULT_ROLL_MAX,),
    )
    await db.execute("DROP TABLE users")
    await db.execute("ALTER TABLE users_new RENAME TO users")


async def _ensure_user(db: aiosqlite.Connection, user_id: int) -> None:
    await db.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))


async def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                roll_max INTEGER NOT NULL DEFAULT {DEFAULT_ROLL_MAX},
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await _add_column_if_missing(db, "roll_max", f"INTEGER NOT NULL DEFAULT {DEFAULT_ROLL_MAX}")
        await db.execute(
            "UPDATE users SET roll_max=? WHERE roll_max IS NULL",
            (DEFAULT_ROLL_MAX,),
        )
        await _rebuild_clean_users_table_if_needed(db)
        await db.commit()
    _enforce_db_permissions()


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
