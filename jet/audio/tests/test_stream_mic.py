import pytest
import numpy as np
from typing import Generator
from jet.audio.stream_mic import stream_non_silent_audio
from jet.audio.record_mic import SAMPLE_RATE, CHANNELS, DTYPE


@pytest.fixture
def mock_stream(monkeypatch):
    """Mock sounddevice InputStream for testing."""
    class MockStream:
        def __init__(self, samplerate, channels, dtype, blocksize):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.blocksize = blocksize
            self._started = False
            self._call_count = 0

        def start(self):
            self._started = True

        def stop(self):
            self._started = False

        def close(self):
            pass

        def read(self, frames):
            self._call_count += 1
            # Simulate non-silent audio for 6 internal chunks, then silent
            if self._call_count <= 6:
                return np.full((frames, self.channels), self._call_count * 100, dtype=self.dtype), False
            return np.zeros((frames, self.channels), dtype=self.dtype), False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    monkeypatch.setattr("sounddevice.InputStream", MockStream)


def test_chunk_duration_equals_min_chunk_duration(mock_stream):
    """Test behavior when chunk_duration equals min_chunk_duration."""
    # Given
    chunk_duration = 1.0
    min_chunk_duration = 1.0
    silence_threshold = 0.01
    silence_duration = 0.1
    expected_chunk_size = int(SAMPLE_RATE * chunk_duration)
    expected_chunks = 6
    expected_data = [
        np.full((expected_chunk_size, CHANNELS), i * 100, dtype=DTYPE)
        for i in range(1, expected_chunks + 1)
    ]

    # When
    generator = stream_non_silent_audio(
        silence_threshold=silence_threshold,
        chunk_duration=chunk_duration,
        silence_duration=silence_duration,
        min_chunk_duration=min_chunk_duration
    )
    chunks = list(generator)[:expected_chunks]

    # Then
