import pytest
from typing import List, Tuple

from jet.audio.speech.silero.speech_utils import get_speech_waves
from jet.audio.speech.silero.speech_types import SpeechSegment


class TestGetSpeechWaves:
    """BDD-style tests for get_speech_waves ensuring complete rise → sustained → fall patterns."""

    @pytest.fixture
    def base_segment(self) -> SpeechSegment:
        """Common segment fixture: 10-second segment starting at 0.0s (16000 Hz)."""
        return SpeechSegment(
            num=1,
            start=0.0,          # seconds
            end=10.0,           # seconds
            prob=0.75,
            duration=10.0,
            frames_length=312,  # ~10s at 512-sample windows (16000 Hz)
            frame_start=0,
            frame_end=311,
            segment_probs=[],
        )

    def test_single_complete_wave_in_middle(self, base_segment: SpeechSegment):
        """
        Given: A segment with one clear wave that rises above threshold, stays high, then falls.
        When:  get_speech_waves is called with threshold=0.7
        Then:  Exactly one complete wave is returned with correct start/end times.
        """
        # 312 frames ≈ 10 seconds at 512-sample windows / 16000 Hz ≈ 32ms per frame
        probs: List[float] = [0.1] * 100 + [0.85] * 100 + [0.2] * 112  # rise at ~3.2s, fall at ~6.4s

        expected_waves: List[Tuple[float, float]] = [(3.2, 6.4)]  # 100 frames = 3.2s

        result = get_speech_waves(base_segment, probs, threshold=0.7, sampling_rate=16000)

        assert result == expected_waves

    def test_multiple_complete_waves(self, base_segment: SpeechSegment):
        """
        Given: A segment containing two distinct complete waves separated by low probability.
        When:  get_speech_waves is called
        Then:  Both complete waves are detected independently.
        """
        probs: List[float] = (
            [0.1] * 50
            + [0.9] * 60   # wave 1: ~1.6s → ~3.52s
            + [0.2] * 40
            + [0.8] * 80   # wave 2: ~5.28s → ~7.84s
            + [0.15] * 82
        )

        expected_waves: List[Tuple[float, float]] = [(1.6, 3.52), (4.8, 7.36)]

        result = get_speech_waves(base_segment, probs, threshold=0.7, sampling_rate=16000)

        assert result == expected_waves

    def test_no_wave_when_only_rise_no_fall(self, base_segment: SpeechSegment):
        """
        Given: Probabilities that rise above threshold but never fall back below (ends high).
        When:  get_speech_waves is called
        Then:  No complete wave is returned because the pattern is incomplete.
        """
        probs: List[float] = [0.1] * 100 + [0.95] * 212  # rises at ~3.2s and stays high

        expected_waves: List[Tuple[float, float]] = []

        result = get_speech_waves(base_segment, probs, threshold=0.7, sampling_rate=16000)

        assert result == expected_waves

    def test_no_wave_when_only_fall_no_rise(self, base_segment: SpeechSegment):
        """
        Given: Segment starts already above threshold and then falls (no leading rise edge).
        When:  get_speech_waves is called
        Then:  No complete wave is returned (missing rise).
        """
        probs: List[float] = [0.9] * 150 + [0.2] * 162  # starts high, falls at ~4.8s

        expected_waves: List[Tuple[float, float]] = []

        result = get_speech_waves(base_segment, probs, threshold=0.7, sampling_rate=16000)

        assert result == expected_waves

    def test_wave_touching_threshold_edge_cases(self, base_segment: SpeechSegment):
        """
        Given: Wave that just touches threshold on rise and fall.
        When:  get_speech_waves is called with threshold=0.7
        Then:  Wave is still considered complete if it crosses ≥0.7 at least once.
        """
        probs: List[float] = (
            [0.69] * 80
            + [0.70, 0.85, 0.82, 0.71]  # brief peak
            + [0.68] * 228
        )

        expected_waves: List[Tuple[float, float]] = [(2.56, 2.688)]

        result = get_speech_waves(base_segment, probs, threshold=0.7, sampling_rate=16000)

        assert result == pytest.approx(expected_waves, abs=0.01)

    def test_empty_or_all_low_probabilities(self, base_segment: SpeechSegment):
        """
        Given: All probabilities below threshold or empty list.
        When:  get_speech_waves is called
        Then:  Empty list is returned.
        """
        probs_low: List[float] = [0.3] * 312
        probs_empty: List[float] = []

        expected: List[Tuple[float, float]] = []

        assert get_speech_waves(base_segment, probs_low, threshold=0.7) == expected
        assert get_speech_waves(base_segment, probs_empty, threshold=0.7) == expected