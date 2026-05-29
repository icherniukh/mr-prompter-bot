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


# ── Batch independence / no-collage tests ────────────────────────────────────


async def test_multiple_images_are_processed_as_separate_independent_calls(fresh_db, monkeypatch):
    """When a user sends several images (as Telegram delivers them one message at a time),
    each image must trigger its own call to remove_overlays.

    The model must never see a 'collage' or multi-image request for one logical user batch.
    """
    monkeypatch.setattr(handlers, "HOST_OPENROUTER_KEY", "sk-or-host")
    monkeypatch.setattr(handlers, "FREE_TIER_LIMIT", 10)

    calls = []  # (image_bytes, model)

    async def fake_remove(api_key, image_bytes, model, mime_type):
        calls.append((image_bytes, model))
        return b"CLEANED-" + image_bytes[:4]

    monkeypatch.setattr(handlers, "remove_overlays", fake_remove)

    images = [b"image-one-data", b"image-two-data", b"image-three-data"]

    for img in images:
        upd = FakeUpdate(uid=99, photo_bytes=img)
        await handlers.handle_image(upd, None)
        assert upd.message.documents_sent == [b"CLEANED-" + img[:4]]

    # Exactly 3 independent calls, each with its own distinct image bytes
    assert len(calls) == 3
    assert calls[0][0] == b"image-one-data"
    assert calls[1][0] == b"image-two-data"
    assert calls[2][0] == b"image-three-data"

    # All used the same model (no weird cross-image payload construction)
    assert all(c[1] == "google/gemini-3.1-flash-image-preview" for c in calls)


async def test_even_concurrent_messages_result_in_separate_model_calls(fresh_db, monkeypatch):
    """Even under concurrent message handling (realistic for albums or rapid sends),
    we still get one remove_overlays call per image, never a combined collage.
    """
    import asyncio

    monkeypatch.setattr(handlers, "HOST_OPENROUTER_KEY", "sk-or-host")
    monkeypatch.setattr(handlers, "FREE_TIER_LIMIT", 10)

    call_count = 0

    async def slow_fake_remove(api_key, image_bytes, model, mime_type):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)  # simulate real latency
        return b"CLEANED"

    monkeypatch.setattr(handlers, "remove_overlays", slow_fake_remove)

    # Simulate 5 images arriving "at the same time"
    updates = [FakeUpdate(uid=123, photo_bytes=f"img-{i}".encode()) for i in range(5)]

    await asyncio.gather(*(handlers.handle_image(u, None) for u in updates))

    assert call_count == 5, "Each image message must produce exactly one model call"
