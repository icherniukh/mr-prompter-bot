import os
from cryptography.fernet import Fernet

# Must be set before any src imports resolve config.py
os.environ.setdefault("ENCRYPTION_KEY", Fernet.generate_key().decode())
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test:token")
os.environ.setdefault("HOST_OPENROUTER_KEY", "sk-or-test-hostkey")
