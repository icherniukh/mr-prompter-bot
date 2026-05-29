from pathlib import Path

import pytest

from src import database as db
from src import handlers
from src.engine import ProcessingError


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", Path(tmp_path) / "bot.db")
    await db.init_db()
    return db


class FakeFile:
    def __init__(self, data):
        self._data = data

    async def download_as_bytearray(self):
        return bytearray(self._data)


class FakePhotoSize:
    def __init__(self, data):
        self._data = data

    async def get_file(self):
        return FakeFile(self._data)


class FakeMessage:
    def __init__(self, photo_bytes=None):
        self.photo = [FakePhotoSize(photo_bytes)] if photo_bytes else []
        self.document = None
        self.message_id = 1
        self.replies = []
        self.documents_sent = []

    async def reply_text(self, text, **kwargs):
        reply = FakeMessage()
        reply.text = text
        self.replies.append(text)
        return _Placeholder()

    async def reply_document(self, document, **kwargs):
        self.documents_sent.append(document.read())


class _Placeholder:
    def __init__(self):
        self.edits = []

    async def edit_text(self, text, **kwargs):
        self.edits.append(text)

    async def delete(self):
        pass


class FakeUser:
    def __init__(self, uid):
        self.id = uid


class FakeUpdate:
    def __init__(self, uid, photo_bytes):
        self.effective_user = FakeUser(uid)
        self.message = FakeMessage(photo_bytes=photo_bytes)


async def test_free_tier_processes_then_blocks(fresh_db, monkeypatch):
    monkeypatch.setattr(handlers, "HOST_OPENROUTER_KEY", "sk-or-host")
    monkeypatch.setattr(handlers, "FREE_TIER_LIMIT", 2)

    used_keys = []

    async def fake_remove(api_key, image_bytes, model, mime_type):
        used_keys.append(api_key)
        return b"CLEANED"

    monkeypatch.setattr(handlers, "remove_overlays", fake_remove)

    # First two images: processed on the host key.
    for _ in range(2):
        upd = FakeUpdate(uid=42, photo_bytes=b"img")
        await handlers.handle_image(upd, None)
        assert upd.message.documents_sent == [b"CLEANED"]

    assert used_keys == ["sk-or-host", "sk-or-host"]
    assert (await fresh_db.get_user(42))["free_used"] == 2

    # Third image: quota exhausted, no processing, prompted to /setup.
    upd = FakeUpdate(uid=42, photo_bytes=b"img")
    await handlers.handle_image(upd, None)
    assert upd.message.documents_sent == []
    assert any("/setup" in r for r in upd.message.replies)
    assert len(used_keys) == 2


async def test_own_key_is_unlimited(fresh_db, monkeypatch):
    monkeypatch.setattr(handlers, "HOST_OPENROUTER_KEY", "sk-or-host")
    monkeypatch.setattr(handlers, "FREE_TIER_LIMIT", 1)
    await fresh_db.set_api_key(7, "sk-or-mine")

    used_keys = []

    async def fake_remove(api_key, image_bytes, model, mime_type):
        used_keys.append(api_key)
        return b"CLEANED"

    monkeypatch.setattr(handlers, "remove_overlays", fake_remove)

    for _ in range(3):
        upd = FakeUpdate(uid=7, photo_bytes=b"img")
        await handlers.handle_image(upd, None)
        assert upd.message.documents_sent == [b"CLEANED"]

    assert used_keys == ["sk-or-mine"] * 3
    # Own key never touches the free counter.
    assert (await fresh_db.get_user(7))["free_used"] == 0


async def test_failure_refunds_free_slot(fresh_db, monkeypatch):
    monkeypatch.setattr(handlers, "HOST_OPENROUTER_KEY", "sk-or-host")
    monkeypatch.setattr(handlers, "FREE_TIER_LIMIT", 2)

    async def boom(api_key, image_bytes, model, mime_type):
        raise ProcessingError("model down")

    monkeypatch.setattr(handlers, "remove_overlays", boom)

    upd = FakeUpdate(uid=5, photo_bytes=b"img")
    await handlers.handle_image(upd, None)
    assert upd.message.documents_sent == []
    # Slot refunded so a failure doesn't cost the user.
    assert (await fresh_db.get_user(5))["free_used"] == 0


async def test_no_host_key_blocks_free_users(fresh_db, monkeypatch):
    monkeypatch.setattr(handlers, "HOST_OPENROUTER_KEY", "")
    monkeypatch.setattr(handlers, "FREE_TIER_LIMIT", 5)

    called = []

    async def fake_remove(*a, **k):
        called.append(1)
        return b"x"

    monkeypatch.setattr(handlers, "remove_overlays", fake_remove)

    upd = FakeUpdate(uid=1, photo_bytes=b"img")
    await handlers.handle_image(upd, None)
    assert called == []
    assert any("/setup" in r for r in upd.message.replies)
