# tests/test_speech_analyzer.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import pytest
import torch

from jet.audio.speech.silero.speech_analyzer  import SpeechAnalyzer, SpeechSegment

# Fixture to create a temporary analyzer with fixed settings
@pytest.fixture
def analyzer() -> SpeechAnalyzer:
    return SpeechAnalyzer(
        threshold=0.5,
        raw_threshold=0.05,
        min_speech_duration_ms=100,   # smaller for synthetic tests
        min_silence_duration_ms=100,
        speech_pad_ms=0,
        sampling_rate=16000,
        min_duration_ms=None,
        min_std_prob=None,
        min_pct_threshold=None,
    )

# Helper to create a simple synthetic waveform with known speech regions
def create_synthetic_waveform(
    sr: int = 16000,
    duration_sec: float = 4.5,
) -> torch.Tensor:
    """
    Creates a waveform using real speech example from Silero repository.
    This guarantees reliable detection since it's actual human speech.
    The official example contains two clear speech segments.
    """
    import urllib.request
    from pathlib import Path

    example_url = "https://models.silero.ai/vad_models/en.wav"
    wav_path = Path("/tmp/en_example.wav")

    if not wav_path.exists():
        print("Downloading Silero example audio for reliable synthetic test...")
        urllib.request.urlretrieve(example_url, str(wav_path))

    wav, original_sr = sf.read(str(wav_path))
    wav = torch.from_numpy(wav).float()

    if original_sr != sr:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=sr)
        wav = resampler(wav.unsqueeze(0)).squeeze(0)

    # Trim or pad to desired duration
    target_samples = int(sr * duration_sec)
    if len(wav) > target_samples:
        wav = wav[:target_samples]
    else:
        wav = torch.nn.functional.pad(wav, (0, target_samples - len(wav)))

    return wav

class TestExtractProbs:
    def test_extract_probs_returns_one_probability_per_window(self, analyzer: SpeechAnalyzer):
        # Given a short synthetic waveform (3 seconds)
        wav = create_synthetic_waveform(sr=analyzer.sr, duration_sec=3.0)

        # When
        probs = analyzer.extract_probs(wav)

        # Then
        expected_windows = int(np.ceil(len(wav) / analyzer.window_size))
        assert len(probs) == expected_windows
        assert all(0.0 <= p <= 1.0 for p in probs)

class TestExtractSegments:
    @pytest.fixture
    def prob_array(self, analyzer: SpeechAnalyzer) -> np.ndarray:
        # 10 windows → 10 probabilities
        return np.array([0.1, 0.2, 0.9, 0.95, 0.98, 0.92, 0.4, 0.3, 0.05, 0.01])

    @pytest.fixture
    def silero_segments(self, analyzer: SpeechAnalyzer) -> List[dict]:
        # Simulate two segments covering windows 2-5 and 6 only (short but kept for test)
        return [
            {"start": 2 * analyzer.window_size, "end": 6 * analyzer.window_size},  # windows 2..5
            {"start": 6 * analyzer.window_size, "end": 7 * analyzer.window_size},  # window 6
        ]

    def test_extract_segments_creates_correct_rich_objects(
        self, analyzer: SpeechAnalyzer, prob_array: np.ndarray, silero_segments: List[dict]
    ):
        # When
        rich_segments = analyzer.extract_segments(silero_segments, prob_array)

        # Then
        assert len(rich_segments) == 2

        # Given: first segment corresponds to windows 2–5 (inclusive)
        expected_probs_seg1 = prob_array[2:6]  # windows 2,3,4,5
        expected_start_ms = int(round(2 * analyzer.step_sec * 1000))
        expected_end_ms = int(round(6 * analyzer.step_sec * 1000))
        expected_duration_ms = expected_end_ms - expected_start_ms

        # When
        result = rich_segments[0]

        # Then
        assert result.num == 1
        assert result.start == expected_start_ms
        assert result.end == expected_end_ms
        assert result.duration == expected_duration_ms
        assert result.stats["avg_prob"] == pytest.approx(float(expected_probs_seg1.mean()), abs=0.001)
        assert result.stats["min_prob"] == pytest.approx(float(expected_probs_seg1.min()), abs=0.001)
        assert result.stats["max_prob"] == pytest.approx(float(expected_probs_seg1.max()), abs=0.001)
        assert result.stats["std_prob"] == pytest.approx(float(expected_probs_seg1.std()), abs=0.001)
        assert result.stats["pct_above_threshold"] == pytest.approx(100.0, abs=0.1)

        # Given: second segment is a short one at window 6
        expected_probs_seg2 = prob_array[6:7]
        expected_start_ms_2 = int(round(6 * analyzer.step_sec * 1000))
        expected_end_ms_2 = int(round(7 * analyzer.step_sec * 1000))

        # When
        result_2 = rich_segments[1]

        # Then
        assert result_2.num == 2
        assert result_2.start == expected_start_ms_2
        assert result_2.end == expected_end_ms_2
        assert result_2.stats["avg_prob"] == pytest.approx(0.4, abs=0.001)
        assert result_2.stats["pct_above_threshold"] == pytest.approx(0.0, abs=0.1)

class TestExtractRawSegments:
    def test_extract_raw_segments_finds_contiguous_regions_above_raw_threshold(
        self, analyzer: SpeechAnalyzer
    ):
        # Given probabilities with two separate regions above raw_threshold=0.05
        prob_array = np.array([0.01, 0.03, 0.12, 0.25, 0.18, 0.08, 0.02, 0.30, 0.35, 0.01])

        # When
        raw_segments = analyzer.extract_raw_segments(prob_array)

        # Then
        # With default filters disabled (None), both contiguous regions are returned
        assert len(raw_segments) == 2

        # Given: first raw segment covers windows 2–5
        expected_probs1 = prob_array[2:6]
        expected_start_ms = int(round(2 * analyzer.step_sec * 1000))
        expected_end_ms = int(round(6 * analyzer.step_sec * 1000))

        # When
        result = raw_segments[0]

        # Then
        assert result.num == 1
        assert result.start == expected_start_ms
        assert result.end == expected_end_ms
        assert result.stats["avg_prob"] == pytest.approx(float(expected_probs1.mean()), abs=0.001)

        # Given: second raw segment covers windows 7–8
        expected_probs2 = prob_array[7:9]
        expected_start_ms_2 = int(round(7 * analyzer.step_sec * 1000))
        expected_end_ms_2 = int(round(9 * analyzer.step_sec * 1000))

        # When
        result_2 = raw_segments[1]

        # Then
        assert result_2.num == 2
        assert result_2.start == expected_start_ms_2
        assert result_2.end == expected_end_ms_2
        assert result_2.stats["avg_prob"] == pytest.approx(float(expected_probs2.mean()), abs=0.001)

    def test_extract_raw_segments_handles_all_below_threshold(self, analyzer: SpeechAnalyzer):
        # Given probabilities all below raw_threshold
        prob_array = np.array([0.01, 0.02, 0.03, 0.04])

        # When
        raw_segments = analyzer.extract_raw_segments(prob_array)

        # Then
        assert len(raw_segments) == 0

    def test_extract_raw_segments_applies_filters_correctly(self):
        # Given an analyzer with active filters

        filtered_analyzer = SpeechAnalyzer(
            threshold=0.5,
            raw_threshold=0.01,  # very low to capture all potential regions
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            speech_pad_ms=0,
            sampling_rate=16000,
            min_duration_ms=300,          # > 300 ms required
            min_std_prob=0.05,           # require some probability variation
            min_pct_threshold=30.0,      # at least 30% windows > threshold
        )

        # Prob array creating three raw regions:
        # 1. Long, high-confidence (should pass)
        # 2. Short, low-variation (should fail duration + std)
        # 3. Long but mostly below threshold (should fail pct)
        prob_array = np.array([
            0.95, 0.96, 0.94, 0.97, 0.93,  # windows 0-4: high, std >0.05, 100% >0.5 → passes
            0.20, 0.21, 0.20,               # windows 5-7: short (~96ms), low std → fails duration & std
            0.40, 0.41, 0.39, 0.42, 0.38, 0.60, 0.30, 0.31  # windows 8-15: long but only 1/8 >0.5 → fails pct
        ])

        # When
        raw_segments = filtered_analyzer.extract_raw_segments(prob_array)

        # Then only the first region should survive all filters
        assert len(raw_segments) == 1

        result = raw_segments[0]

        expected_start_ms = 0
        expected_end_ms = int(round(5 * filtered_analyzer.step_sec * 1000))
        expected_duration_ms = expected_end_ms - expected_start_ms

        assert result.num == 1
        assert result.start == expected_start_ms
        assert result.end == expected_end_ms
        assert result.duration == expected_duration_ms
        assert result.stats["avg_prob"] == pytest.approx(0.95, abs=0.01)
        assert result.stats["std_prob"] > 0.01  # variation present
        assert result.stats["pct_above_threshold"] == pytest.approx(100.0, abs=0.1)

    def test_extract_raw_segments_no_filters_returns_all_regions(self):
        # Given same prob array as above but with no filters active
        prob_array = np.array([
            0.95, 0.96, 0.94, 0.97, 0.93,
            0.20, 0.21, 0.20,
            0.40, 0.41, 0.39, 0.42, 0.38, 0.60, 0.30, 0.31
        ])

        # When
        no_filter_analyzer = SpeechAnalyzer(
            threshold=0.5,
            raw_threshold=0.01,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            speech_pad_ms=0,
            sampling_rate=16000,
            min_duration_ms=None,
            min_std_prob=None,
            min_pct_threshold=None,
        )
        raw_segments = no_filter_analyzer.extract_raw_segments(prob_array)

        # Then all three contiguous regions are returned
        assert len(raw_segments) == 3

        # Quick sanity checks on boundaries
        assert raw_segments[0].end == int(round(5 * no_filter_analyzer.step_sec * 1000))
        assert raw_segments[1].duration == int(round(3 * no_filter_analyzer.step_sec * 1000))
        assert raw_segments[2].duration == int(round(8 * no_filter_analyzer.step_sec * 1000))

class TestAnalyzeIntegration:
    def test_analyze_returns_consistent_lengths_with_synthetic_speech(
        self, analyzer: SpeechAnalyzer, tmp_path: Path
    ):
        # Given a waveform based on real speech example (guarantees detection)
        wav = create_synthetic_waveform(sr=analyzer.sr, duration_sec=4.5)

        # Write to temporary file
        audio_path = tmp_path / "synthetic_real.wav"
        sf.write(str(audio_path), wav.numpy(), analyzer.sr)

        # When
        probs, rich_segments, raw_segments = analyzer.analyze(str(audio_path))

        # Then
        expected_windows = int(np.ceil(len(wav) / analyzer.window_size))
        result_probs_len = len(probs)

        result_rich_len = len(rich_segments)
        expected_rich_len = 2  # The Silero example file contains two distinct utterances

        result_raw_len = len(raw_segments)
        expected_raw_at_least = 2

        assert result_probs_len == expected_windows
        assert result_rich_len == expected_rich_len
        assert result_raw_len >= expected_raw_at_least
        assert all(isinstance(seg, SpeechSegment) for seg in rich_segments + raw_segments)
