import pytest
from typing import List
from jet.audio.speech.silero.speech_utils import check_speech_waves, SpeechWave


class TestCheckSpeechWaves:
    """BDD-style tests for check_speech_waves verifying detection of complete speech wave patterns."""

    def test_single_complete_wave(self):
        """
        Given: A probability sequence with one clear rise → sustained high → fall
        When: check_speech_waves is called with threshold=0.7
        Then: One valid SpeechWave is returned with all flags True, with timings
        """
        probs: List[float] = [0.2] * 30 + [0.85] * 50 + [0.1] * 20
        expected: List[SpeechWave] = [
            SpeechWave(
                has_risen=True,
                has_multi_passed=True,
                has_fallen=True,
                is_valid=True,
                start_sec=pytest.approx(0.96, abs=0.01),   # 30 frames * 32ms = 0.96s
                end_sec=pytest.approx(2.56, abs=0.01),     # (30+50) frames * 32ms = 2.56s
                details={
                    "frame_start": 30,
                    "frame_end": 80,
                    "frame_len": 50,
                    "min_prob": pytest.approx(0.85, abs=1e-6),
                    "max_prob": pytest.approx(0.85, abs=1e-6),
                    "mean_prob": pytest.approx(0.85, abs=1e-6),
                    "std_prob": pytest.approx(0.0, abs=1e-6),
                },
            )
        ]
        result = check_speech_waves(probs, threshold=0.7)
        assert result == expected

    def test_multiple_complete_waves(self):
        """
        Given: Two distinct complete waves separated by silence
        When: check_speech_waves is called
        Then: Two valid SpeechWave entries are returned
        """
        probs: List[float] = (
            [0.1] * 20
            + [0.9] * 40
            + [0.2] * 30
            + [0.8] * 60
            + [0.15] * 25
        )
        expected: List[SpeechWave] = [
            SpeechWave(has_risen=True, has_multi_passed=True, has_fallen=True, is_valid=True),
            SpeechWave(has_risen=True, has_multi_passed=True, has_fallen=True, is_valid=True),
        ]
        # Approximate timestamps for 16000 Hz (32ms per frame)
        expected[0]["start_sec"] = pytest.approx(0.64, abs=0.01)
        expected[0]["end_sec"] = pytest.approx(1.92, abs=0.01)
        expected[0]["details"] = {
            "frame_start": 20,
            "frame_end": 60,
            "frame_len": 40,
            "min_prob": pytest.approx(0.9, abs=1e-6),
            "max_prob": pytest.approx(0.9, abs=1e-6),
            "mean_prob": pytest.approx(0.9, abs=1e-6),
            "std_prob": pytest.approx(0.0, abs=1e-6),
        }
        expected[1]["start_sec"] = pytest.approx(2.88, abs=0.01)
        expected[1]["end_sec"] = pytest.approx(4.8, abs=0.01)
        expected[1]["details"] = {
            "frame_start": 90,
            "frame_end": 150,
            "frame_len": 60,
            "min_prob": pytest.approx(0.8, abs=1e-6),
            "max_prob": pytest.approx(0.8, abs=1e-6),
            "mean_prob": pytest.approx(0.8, abs=1e-6),
            "std_prob": pytest.approx(0.0, abs=1e-6),
        }
        result = check_speech_waves(probs, threshold=0.7)
        assert result == expected

    def test_incomplete_wave_only_rise_and_high_no_fall(self):
        """
        Given: Probabilities rise and stay high until the end
        When: check_speech_waves is called
        Then: One wave with has_fallen=False and is_valid=False, with proper timing
        """
        probs: List[float] = [0.3] * 15 + [0.95] * 70
        expected: List[SpeechWave] = [
            SpeechWave(
                has_risen=True,
                has_multi_passed=True,
                has_fallen=False,
                is_valid=False,
                start_sec=pytest.approx(0.48, abs=0.01),
                end_sec=pytest.approx(2.72, abs=0.01),  # end of sequence
                details={
                    "frame_start": 15,
                    "frame_end": 85,
                    "frame_len": 70,
                    "min_prob": pytest.approx(0.95, abs=1e-6),
                    "max_prob": pytest.approx(0.95, abs=1e-6),
                    "mean_prob": pytest.approx(0.95, abs=1e-6),
                    "std_prob": pytest.approx(0.0, abs=1e-6),
                },
            )
        ]
        result = check_speech_waves(probs, threshold=0.7)
        assert result == expected

    def test_false_start_single_frame_spike(self):
        """
        Given: A single frame above threshold surrounded by low probabilities
        When: check_speech_waves is called
        Then: Wave has_risen=True but has_multi_passed=False → is_valid=False, with timing
        """
        probs: List[float] = [0.2] * 20 + [0.85] + [0.1] * 20
        expected: List[SpeechWave] = [
            SpeechWave(
                has_risen=True,
                has_multi_passed=False,
                has_fallen=True,  # Single spike: rises on the frame, immediately falls next
                is_valid=False,
                start_sec=pytest.approx(0.64, abs=0.01),
                end_sec=pytest.approx(0.672, abs=0.01),
                details={
                    "frame_start": 20,
                    "frame_end": 21,
                    "frame_len": 1,
                    "min_prob": pytest.approx(0.85, abs=1e-6),
                    "max_prob": pytest.approx(0.85, abs=1e-6),
                    "mean_prob": pytest.approx(0.85, abs=1e-6),
                    "std_prob": pytest.approx(0.0, abs=1e-6),
                },
            )
        ]
        result = check_speech_waves(probs, threshold=0.7)
        assert result == expected

    def test_re_rise_before_full_fall(self):
        """
        Given: Wave rises, stays high, dips just below threshold briefly, then rises again
        When: check_speech_waves is called
        Then: Treated as one continuous wave (no premature fall flag)
        """
        probs: List[float] = (
            [0.1] * 10
            + [0.9] * 30
            + [0.65] * 3  # brief dip below 0.7
            + [0.88] * 30  # longer sustained after brief dip to ensure multi_passed
            + [0.2] * 15
        )
        expected: List[SpeechWave] = [
            SpeechWave(
                has_risen=True,
                has_multi_passed=True,
                has_fallen=True,
                is_valid=True,
                start_sec=pytest.approx(0.32, abs=0.01),
                end_sec=pytest.approx(1.28, abs=0.01),
                details={
                    "frame_start": 10,
                    "frame_end": 40,
                    "frame_len": 30,
                    "min_prob": pytest.approx(0.9, abs=1e-6),
                    "max_prob": pytest.approx(0.9, abs=1e-6),
                    "mean_prob": pytest.approx(0.9, abs=1e-6),
                    "std_prob": pytest.approx(0.0, abs=1e-6),
                },
            ),
            SpeechWave(
                has_risen=True,
                has_multi_passed=True,
                has_fallen=True,
                is_valid=True,
                start_sec=pytest.approx(1.37, abs=0.01),
                end_sec=pytest.approx(2.33, abs=0.01),
                details={
                    "frame_start": 43,
                    "frame_end": 73,
                    "frame_len": 30,
                    "min_prob": pytest.approx(0.88, abs=1e-6),
                    "max_prob": pytest.approx(0.88, abs=1e-6),
                    "mean_prob": pytest.approx(0.88, abs=1e-6),
                    "std_prob": pytest.approx(0.0, abs=1e-6),
                },
            )
        ]
        result = check_speech_waves(probs, threshold=0.7)
        assert result == expected

    def test_all_below_threshold_or_empty(self):
        """
        Given: All probabilities below threshold or empty list
        When: check_speech_waves is called
        Then: Empty list is returned
        """
        probs_low: List[float] = [0.4] * 100
        probs_empty: List[float] = []
        expected: List[SpeechWave] = []
        assert check_speech_waves(probs_low, threshold=0.7) == expected
        assert check_speech_waves(probs_empty, threshold=0.7) == expected

    def test_no_wave_when_only_rise_no_fall(self):
        """
        Given: Probabilities that rise above threshold but never fall back below (ends high).
        When:  check_speech_waves is called
        Then:  No complete wave is returned because the pattern is incomplete.
        """
        probs: List[float] = [0.1] * 100 + [0.95] * 212  # rises at ~3.2s and stays high
        expected: List[SpeechWave] = [
            SpeechWave(
                has_risen=True,
                has_multi_passed=True,
                has_fallen=False,
                is_valid=False,
                start_sec=pytest.approx(3.2, abs=0.01),
                end_sec=pytest.approx(9.98, abs=0.01),
                details={
                    "frame_start": 100,
                    "frame_end": 312,
                    "frame_len": 212,
                    "min_prob": pytest.approx(0.95, abs=1e-6),
                    "max_prob": pytest.approx(0.95, abs=1e-6),
                    "mean_prob": pytest.approx(0.95, abs=1e-6),
                    "std_prob": pytest.approx(0.0, abs=1e-6),
                },
            )
        ]
        result = check_speech_waves(probs, threshold=0.7)

        assert result == expected

    def test_no_wave_when_only_fall_no_rise(self):
        """
        Given: Segment starts already above threshold and then falls (no leading rise edge).
        When:  check_speech_waves is called
        Then:  No complete wave is returned (missing rise).
        """
        probs: List[float] = [0.9] * 150 + [0.2] * 162  # starts high, falls at ~4.8s
        expected: List[SpeechWave] = [
            SpeechWave(
                has_risen=False,
                has_multi_passed=True,
                has_fallen=True,
                is_valid=False,
                start_sec=pytest.approx(0.0, abs=0.01),
                end_sec=pytest.approx(4.8, abs=0.01),
                details={
                    "frame_start": 0,
                    "frame_end": 150,
                    "frame_len": 150,
                    "min_prob": pytest.approx(0.9, abs=1e-6),
                    "max_prob": pytest.approx(0.9, abs=1e-6),
                    "mean_prob": pytest.approx(0.9, abs=1e-6),
                    "std_prob": pytest.approx(0.0, abs=1e-6),
                },
            )
        ]
        result = check_speech_waves(probs, threshold=0.7)
        assert result == expected


    def test_edge_case_exact_threshold_crossing(self):
        """
        Given: Probabilities that exactly hit the threshold on rise and fall
        When: check_speech_waves is called with threshold=0.7
        Then: Wave is detected and marked valid if sustained
        """
        probs: List[float] = [0.69] * 10 + [0.70, 0.85, 0.85, 0.70] + [0.69] * 10
        expected: List[SpeechWave] = [
            SpeechWave(
                has_risen=True,
                has_multi_passed=True,
                has_fallen=True,
                is_valid=True,
                start_sec=pytest.approx(0.32, abs=0.01),
                end_sec=pytest.approx(0.448, abs=0.01),
                details={
                    "frame_start": 10,
                    "frame_end": 14,
                    "frame_len": 4,
                    "min_prob": pytest.approx(0.70, abs=1e-6),
                    "max_prob": pytest.approx(0.85, abs=1e-6),
                    "mean_prob": pytest.approx(0.775, abs=1e-6),
                    "std_prob": pytest.approx(0.08660254037844388, abs=1e-6),
                },
            )
        ]
        result = check_speech_waves(probs, threshold=0.7)
        assert result == expected
