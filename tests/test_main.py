from types import SimpleNamespace
from unittest.mock import AsyncMock

from telegram import BotCommand
from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler

from src import handlers
from src import main as main_mod


class FakeApplication:
    def __init__(self):
        self.handlers = []
        self.run_polling_called = False

    def add_handler(self, handler):
        self.handlers.append(handler)

    def run_polling(self):
        self.run_polling_called = True


class FakeApplicationBuilder:
    def __init__(self, app):
        self.app = app
        self.token_arg = None
        self.post_init_arg = None
        self.job_queue_arg = object()
        self.build_called = False

    def token(self, value):
        self.token_arg = value
        return self

    def post_init(self, callback):
        self.post_init_arg = callback
        return self

    def job_queue(self, value):
        self.job_queue_arg = value
        return self

    def build(self):
        self.build_called = True
        return self.app


async def test_post_init_sets_commands_and_initializes_db(monkeypatch):
    init_db = AsyncMock()
    set_my_commands = AsyncMock()
    app = SimpleNamespace(bot=SimpleNamespace(set_my_commands=set_my_commands))

    monkeypatch.setattr(main_mod, "init_db", init_db)

    await main_mod._post_init(app)

    init_db.assert_awaited_once_with()
    set_my_commands.assert_awaited_once()
    commands = set_my_commands.await_args.args[0]

    assert commands == [
        BotCommand("settings", "Settings"),
        BotCommand("speak", "Text-to-speech voice message"),
        BotCommand("cancel", "Cancel current action"),
        BotCommand("support", "Report an issue / feedback"),
        BotCommand("push_the_horses", "Push the horses"),
        BotCommand("forget", "Delete my data"),
    ]


def test_main_builds_application_registers_handlers_and_runs_polling(monkeypatch):
    app = FakeApplication()
    builder = FakeApplicationBuilder(app)

    monkeypatch.setattr(main_mod, "ApplicationBuilder", lambda: builder)
    monkeypatch.setattr(main_mod, "TELEGRAM_BOT_TOKEN", "test:token")

    main_mod.main()

    assert builder.token_arg == "test:token"
    assert builder.post_init_arg is main_mod._post_init
    assert builder.job_queue_arg is None
    assert builder.build_called is True
    assert app.run_polling_called is True

    assert len(app.handlers) == 10

    settings_handler, speak_handler, cancel_handler, support_handler, horses_handler, forget_handler = app.handlers[:6]
    settings_callback, forget_callback = app.handlers[6:8]
    settings_text_handler, image_handler = app.handlers[8:]

    assert isinstance(settings_handler, CommandHandler)
    assert settings_handler.commands == frozenset({"settings"})
    assert settings_handler.callback is handlers.settings

    assert isinstance(speak_handler, CommandHandler)
    assert speak_handler.commands == frozenset({"speak"})
    assert speak_handler.callback is handlers.speak

    assert isinstance(cancel_handler, CommandHandler)
    assert cancel_handler.commands == frozenset({"cancel"})
    assert cancel_handler.callback is handlers.cancel

    assert isinstance(support_handler, CommandHandler)
    assert support_handler.commands == frozenset({"support"})
    assert support_handler.callback is handlers.support



    assert isinstance(horses_handler, CommandHandler)
    assert horses_handler.commands == frozenset({"push_the_horses"})
    assert horses_handler.callback is handlers.push_the_horses


    assert isinstance(forget_handler, CommandHandler)
    assert forget_handler.commands == frozenset({"forget"})
    assert forget_handler.callback is handlers.forget

    assert isinstance(settings_callback, CallbackQueryHandler)
    assert settings_callback.callback is handlers.settings_callback
    assert settings_callback.pattern.pattern == r"^settings:"

    assert isinstance(forget_callback, CallbackQueryHandler)
    assert forget_callback.callback is handlers.forget_callback
    assert forget_callback.pattern.pattern == r"^forget:"

    assert isinstance(settings_text_handler, MessageHandler)
    assert settings_text_handler.callback is handlers.handle_settings_text
    assert "filters.TEXT" in repr(settings_text_handler.filters)
    assert "filters.ChatType.PRIVATE" in repr(settings_text_handler.filters)
    assert "filters.COMMAND" in repr(settings_text_handler.filters)

    assert isinstance(image_handler, MessageHandler)
    assert image_handler.callback is handlers.handle_image
    assert "filters.PHOTO" in repr(image_handler.filters)
    assert "image/" in repr(image_handler.filters)
    assert "filters.ChatType.PRIVATE" in repr(image_handler.filters)
