from types import SimpleNamespace
from unittest.mock import AsyncMock

from telegram import BotCommand
from telegram.ext import CommandHandler, MessageHandler

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


async def test_post_init_sets_clean_commands_and_initializes_db(monkeypatch):
    init_db = AsyncMock()
    set_my_commands = AsyncMock()
    app = SimpleNamespace(bot=SimpleNamespace(set_my_commands=set_my_commands))

    monkeypatch.setattr(main_mod, "init_db", init_db)

    await main_mod._post_init(app)

    init_db.assert_awaited_once_with()
    set_my_commands.assert_awaited_once()
    commands = set_my_commands.await_args.args[0]

    assert commands == [
        BotCommand("push_the_horses", "Push the horses"),
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

    assert len(app.handlers) == 2

    horses_handler, bullshit_handler = app.handlers

    assert isinstance(horses_handler, CommandHandler)
    assert horses_handler.commands == frozenset({"push_the_horses"})
    assert horses_handler.callback is handlers.push_the_horses

    assert isinstance(bullshit_handler, MessageHandler)
    assert bullshit_handler.callback is handlers.handle_bullshit
    assert "filters.ChatType.PRIVATE" in repr(bullshit_handler.filters)
