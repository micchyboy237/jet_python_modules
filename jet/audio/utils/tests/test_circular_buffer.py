# test_circular_buffer.py
import numpy as np
import pytest
from jet.audio.utils.circular_buffer import CircularAudioBuffer


@pytest.fixture
def buffer_10s():
    """Provides a fresh buffer limited to 10 seconds of audio."""
    return CircularAudioBuffer(max_duration_sec=10.0, sample_rate=16000)


class TestCircularAudioBuffer:
    def test_initialization(self):
        buf = CircularAudioBuffer(max_duration_sec=5.0, sample_rate=16000)
        assert buf.max_duration_sec == 5.0
        assert buf.max_samples == 80_000  # 5s * 16kHz
        assert buf.total_samples == 0
        assert len(buf.segments) == 0

    def test_add_valid_segment(self, buffer_10s):
        audio = np.zeros(16000, dtype=np.int16)  # Exactly 1 second
        buffer_10s.add_audio_segment(audio, ja_text="こんにちは", en_text="Hello")
        assert buffer_10s.total_samples == 16000
        assert len(buffer_10s.segments) == 1

    def test_add_empty_segment_is_ignored(self, buffer_10s):
        audio = np.array([], dtype=np.int16)
        buffer_10s.add_audio_segment(audio, ja_text="skip")
        assert buffer_10s.total_samples == 0
        assert len(buffer_10s.segments) == 0

    def test_add_invalid_dtype_raises(self, buffer_10s):
        audio = np.zeros(100, dtype=np.float32)
        with pytest.raises(TypeError, match="expects np.int16"):
            buffer_10s.add_audio_segment(audio)

    def test_add_multi_channel_raises(self, buffer_10s):
        audio = np.zeros((100, 2), dtype=np.int16)  # Stereo
        with pytest.raises(ValueError, match="1D"):
            buffer_10s.add_audio_segment(audio)

    def test_add_too_long_segment_raises(self, buffer_10s):
        # 11 seconds > 10 second limit
        audio = np.zeros(11 * 16000, dtype=np.int16)
        with pytest.raises(ValueError, match="exceeds max_duration_sec"):
            buffer_10s.add_audio_segment(audio)

    def test_pruning_removes_oldest_segment(self, buffer_10s):
        # Add two 6-second chunks. Total = 12s > 10s limit. First should be dropped.
        chunk = np.zeros(6 * 16000, dtype=np.int16)
        buffer_10s.add_audio_segment(chunk, id="first")
        buffer_10s.add_audio_segment(chunk, id="second")

        # Only the second chunk remains
        assert len(buffer_10s.segments) == 1
        assert buffer_10s.segments[0][1].get("id") == "second"
        assert buffer_10s.total_samples == 6 * 16000

    def test_get_full_audio_empty(self, buffer_10s):
        result = buffer_10s.get_full_audio()
        assert result.dtype == np.int16
        assert len(result) == 0

    def test_get_full_audio_concatenates(self, buffer_10s):
        chunk1 = np.array([1, 2], dtype=np.int16)
        chunk2 = np.array([3, 4], dtype=np.int16)
        buffer_10s.add_audio_segment(chunk1)
        buffer_10s.add_audio_segment(chunk2)
        result = buffer_10s.get_full_audio()
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4], dtype=np.int16))

    def test_get_total_duration(self, buffer_10s):
        audio = np.zeros(32000, dtype=np.int16)  # 2 seconds
        buffer_10s.add_audio_segment(audio)
        assert buffer_10s.get_total_duration() == pytest.approx(2.0)

    def test_get_history_no_metadata(self, buffer_10s):
        audio = np.zeros(1000, dtype=np.int16)
        buffer_10s.add_audio_segment(audio)  # No meta passed
        history = buffer_10s.get_history()
        assert history == []

    def test_get_history_filters_incomplete_pairs(self, buffer_10s):
        audio = np.zeros(1000, dtype=np.int16)
        buffer_10s.add_audio_segment(audio, ja_text="only ja")
        buffer_10s.add_audio_segment(audio, en_text="only en")
        buffer_10s.add_audio_segment(audio, ja_text="ja", en_text="en")

        history = buffer_10s.get_history()
        # Only the third segment has both JA and EN
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "ja"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "en"

    def test_get_history_respects_max_segments(self, buffer_10s):
        audio = np.zeros(1000, dtype=np.int16)
        for i in range(5):
            buffer_10s.add_audio_segment(audio, ja_text=f"ja_{i}", en_text=f"en_{i}")

        history = buffer_10s.get_history(max_segments=2)
        # max_segments=2 means last 2 segments -> 4 history items (user+assistant each)
        assert len(history) == 4
        assert history[0]["content"] == "ja_3"
        assert history[-1]["content"] == "en_4"

    def test_reset_clears_state(self, buffer_10s):
        audio = np.zeros(1000, dtype=np.int16)
        buffer_10s.add_audio_segment(audio, ja_text="test", en_text="test")
        buffer_10s.reset()

        assert buffer_10s.total_samples == 0
        assert len(buffer_10s.segments) == 0
        assert buffer_10s.get_total_duration() == 0.0
        assert buffer_10s.get_history() == []
        assert len(buffer_10s.get_full_audio()) == 0
