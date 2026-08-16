import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from telegram import BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from src.config import TELEGRAM_BOT_TOKEN
from src.database import init_db
from src import handlers

# Ensure logs directory exists
Path("data/logs").mkdir(parents=True, exist_ok=True)

# Main logger config
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Persistent error telemetry: append errors to a permanent file for later analysis
error_log_path = "data/logs/errors.log"
error_handler = RotatingFileHandler(
    error_log_path,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5,
    encoding="utf-8",
)
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s\n"
    "Exception: %(exc_info)s\n"
    "----------------------------------------\n"
)
error_handler.setFormatter(error_formatter)

# Attach to root logger so all errors are captured
logging.getLogger().addHandler(error_handler)

logger = logging.getLogger(__name__)
logger.info(f"Persistent error logging enabled → {error_log_path}")

# Redact common API-key-looking strings from logs.
_API_KEY_RE = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9_\-]{20,})")


class _RedactApiKeys(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _API_KEY_RE.sub("[REDACTED]", str(record.msg))
        if record.args:
            record.args = tuple(
                _API_KEY_RE.sub("[REDACTED]", str(a)) if isinstance(a, str) else a
                for a in record.args
            )
        return True


logging.getLogger().addFilter(_RedactApiKeys())


async def _post_init(app) -> None:
    await init_db()
    await app.bot.set_my_commands([
        BotCommand("settings", "Settings"),
        BotCommand("speak", "Text-to-speech voice message"),
        BotCommand("cancel", "Cancel current action"),
        BotCommand("support", "Report an issue / feedback"),
        BotCommand("push_the_horses", "Push the horses"),
        BotCommand("forget", "Delete my data"),
    ])


def main() -> None:
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(_post_init)
        .job_queue(None)
        .build()
    )

    app.add_handler(CommandHandler("settings", handlers.settings, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("speak", handlers.speak, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("cancel", handlers.cancel, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("support", handlers.support, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("push_the_horses", handlers.push_the_horses, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("forget", handlers.forget, filters.ChatType.PRIVATE))
    app.add_handler(CallbackQueryHandler(handlers.settings_callback, pattern=r"^settings:"))
    app.add_handler(CallbackQueryHandler(handlers.forget_callback, pattern=r"^forget:"))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE, handlers.handle_settings_text)
    )




    image_filter = filters.PHOTO | filters.Document.IMAGE
    app.add_handler(MessageHandler(image_filter & filters.ChatType.PRIVATE, handlers.handle_image))

    print("Mr Prompter watermark-removal bot started…")
    app.run_polling()


if __name__ == "__main__":
    main()
