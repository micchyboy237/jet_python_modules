import numpy as np
import pytest
from jet.audio.audio_search import find_audio_offset


class TestAudioOffset:
    SAMPLE_RATE = 1000

    def test_exact_match_at_start(self):
        # Given a deterministic non-periodic signal
        long_signal = np.concatenate(
            [np.linspace(0, 1, 1000), np.random.default_rng(42).normal(0, 0.1, 2000)]
        )

        short_signal = long_signal[:500]

        # When searching for the short signal
        result = find_audio_offset(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
        )

        # Then it should match at index 0
        expected_start = 0
        expected_end = 500

        assert result is not None
        assert result["start_sample"] == expected_start
        assert result["end_sample"] == expected_end
        assert result["start_time"] == 0.0
        assert result["confidence"] == pytest.approx(1.0, abs=1e-6)

    def test_exact_match_with_offset(self):
        # Given silence + unique signal + silence
        rng = np.random.default_rng(123)

        silence = np.zeros(1000)
        unique = rng.normal(0, 1, 500)
        long_signal = np.concatenate([silence, unique, silence])

        short_signal = unique.copy()

        # When searching
        result = find_audio_offset(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
        )

        # Then it should detect correct offset
        expected_start = 1000
        expected_end = 1500

        assert result is not None
        assert result["start_sample"] == expected_start
        assert result["end_sample"] == expected_end
        assert result["start_time"] == 1.0
        assert result["end_time"] == 1.5
        assert result["confidence"] == pytest.approx(1.0, abs=1e-6)

    def test_returns_none_if_not_found(self):
        # Given two unrelated signals
        rng = np.random.default_rng(999)

        long_signal = rng.normal(0, 1, 3000)
        short_signal = rng.normal(0, 1, 500)

        # When searching with high confidence threshold
        result = find_audio_offset(
            long_signal=long_signal,
            short_signal=short_signal,
            sample_rate=self.SAMPLE_RATE,
            confidence_threshold=0.95,
        )

        # Then result should be None
        expected = None
        assert result is expected
