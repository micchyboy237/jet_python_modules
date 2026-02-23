import numpy as np
import pytest
from jet.audio.audio_waveform import AudioStreamingBuffer


class TestAudioStreamingBuffer:
    def test_empty_buffer(self):
        buf = AudioStreamingBuffer(5)
        result = buf.to_array()
        expected = np.array([], dtype=np.float32)
        assert np.array_equal(result, expected)

    def test_append_within_limit(self):
        buf = AudioStreamingBuffer(5)
        buf.append(np.array([1, 2, 3], dtype=float))

        result = buf.to_array()
        expected = np.array([1, 2, 3], dtype=np.float32)
        assert np.array_equal(result, expected)

    def test_overflow_keeps_latest(self):
        buf = AudioStreamingBuffer(3)
        buf.append(np.array([1, 2, 3, 4, 5], dtype=float))

        result = buf.to_array()
        expected = np.array([3, 4, 5], dtype=np.float32)
        assert np.array_equal(result, expected)

    def test_multiple_appends(self):
        buf = AudioStreamingBuffer(4)
        buf.append(np.array([1, 2], dtype=float))
        buf.append(np.array([3, 4, 5], dtype=float))

        result = buf.to_array()
        expected = np.array([2, 3, 4, 5], dtype=np.float32)
        assert np.array_equal(result, expected)

    def test_invalid_buffer_size(self):
        with pytest.raises(ValueError):
            AudioStreamingBuffer(0)

    def test_invalid_samples_dimension(self):
        buf = AudioStreamingBuffer(5)
        with pytest.raises(ValueError):
            buf.append(np.array([[1, 2], [3, 4]]))
