import pytest
from unittest.mock import AsyncMock, patch

from src import tts_engine


async def test_generate_speech_empty_text():
    with pytest.raises(ValueError, match="cannot be empty"):
        await tts_engine.generate_speech("")


async def test_generate_speech_success():
    fake_chunks = [
        {"type": "audio", "data": b"AUDIO_PART_1_"},
        {"type": "audio", "data": b"AUDIO_PART_2"},
        {"type": "other", "data": b"ignored"},
    ]

    class FakeCommunicate:
        def __init__(self, text, voice):
            self.text = text
            self.voice = voice

        async def stream(self):
            for c in fake_chunks:
                yield c

    with patch("src.tts_engine.edge_tts.Communicate", FakeCommunicate):
        out = await tts_engine.generate_speech("Hello world", voice="test-voice")
        assert out == b"AUDIO_PART_1_AUDIO_PART_2"


async def test_generate_speech_no_data_raises():
    class FakeEmptyCommunicate:
        def __init__(self, text, voice):
            pass

        async def stream(self):
            if False:
                yield {}

    with patch("src.tts_engine.edge_tts.Communicate", FakeEmptyCommunicate):
        with pytest.raises(RuntimeError, match="No audio data returned"):
            await tts_engine.generate_speech("Hello world")
