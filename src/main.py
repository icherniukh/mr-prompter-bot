import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from telegram import BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    filters,
)

from src.config import HOST_OPENROUTER_KEY, TELEGRAM_BOT_TOKEN
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

# Redact anything that looks like an API key from logs (OpenRouter keys are sk-or-…).
_API_KEY_RE = re.compile(r"(sk-or-[A-Za-z0-9_\-]{10,}|sk-[A-Za-z0-9_\-]{20,})")


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
        BotCommand("start", "What this bot does"),
        BotCommand("status", "Free images remaining / current model"),
        BotCommand("setup", "Add your own OpenRouter API key"),
        BotCommand("model", "Choose the AI model"),
        BotCommand("forget", "Delete all your stored data"),
        BotCommand("cancel", "Cancel the current operation"),
    ])
    if not HOST_OPENROUTER_KEY:
        logging.getLogger(__name__).warning(
            "HOST_OPENROUTER_KEY is not set — the free tier is disabled; users must "
            "run /setup with their own key before processing any images."
        )


def build_setup_wizard() -> ConversationHandler:
    return ConversationHandler(
        entry_points=[
            CommandHandler("setup", handlers.start_setup, filters.ChatType.PRIVATE),
            CommandHandler("model", handlers.choose_model, filters.ChatType.PRIVATE),
        ],
        states={
            handlers.ASK_KEY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.receive_key)
            ],
            handlers.ASK_MODEL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.receive_model)
            ],
        },
        fallbacks=[CommandHandler("cancel", handlers.cancel)],
        conversation_timeout=300,
    )


def main() -> None:
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(_post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", handlers.start, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("help", handlers.help_command, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("status", handlers.status, filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("forget", handlers.forget, filters.ChatType.PRIVATE))
    app.add_handler(build_setup_wizard())

    # Images arrive as photos or as image documents. Each message is processed
    # independently, so albums/batches just produce one cleaned image per input.
    image_filter = filters.PHOTO | filters.Document.IMAGE
    app.add_handler(MessageHandler(image_filter & filters.ChatType.PRIVATE, handlers.handle_image))

    print("Mr Prompter watermark-removal bot started…")
    app.run_polling()


if __name__ == "__main__":
    main()
