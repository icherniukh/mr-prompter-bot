import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test:token")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gemini_free: tests that make real calls to Gemini free tier (expensive, rate limited)"
    )
