# tests/test_transcriber.py
import pytest
import asyncio
import numpy as np
from jet.audio.system.transcribe_system_audio import AudioTranscriber


class TestAudioTranscriber:
    def test_chunk_samples(self):
        tr = AudioTranscriber(sample_rate=8000, chunk_duration=0.5)
        assert tr.chunk_samples == 4000

    def test_mock_transcription(monkeypatch):
        tr = AudioTranscriber()
        fake_array = np.zeros(tr.chunk_samples, dtype=np.int16)
        tr.frames = [fake_array]

        async def fake_transcribe():
            return "hello"
        tr.input_device = None
        tr.model.transcribe = lambda *args, **kwargs: (
            [type("x", (), {"text": "hello"})()], None)
        result = asyncio.get_event_loop().run_until_complete(tr.capture_and_transcribe())
        assert result == "hello"
