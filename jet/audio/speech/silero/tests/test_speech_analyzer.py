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

        seg1 = rich_segments[0]
        expected_probs_seg1 = prob_array[2:6]  # windows 2,3,4,5
        assert seg1.start_sec == pytest.approx(2 * analyzer.step_sec, abs=0.001)
        assert seg1.end_sec == pytest.approx(6 * analyzer.step_sec, abs=0.001)
        assert seg1.duration_sec == pytest.approx(4 * analyzer.step_sec, abs=0.001)
        assert seg1.avg_probability == pytest.approx(float(expected_probs_seg1.mean()), abs=0.001)
        assert seg1.min_probability == pytest.approx(float(expected_probs_seg1.min()), abs=0.001)
        assert seg1.max_probability == pytest.approx(float(expected_probs_seg1.max()), abs=0.001)
        assert seg1.percent_above_threshold == pytest.approx(100.0, abs=0.1)  # all > 0.5

        seg2 = rich_segments[1]
        expected_probs_seg2 = prob_array[6:7]
        assert seg2.avg_probability == pytest.approx(0.4, abs=0.001)
        assert seg2.percent_above_threshold == pytest.approx(0.0, abs=0.1)

class TestExtractRawSegments:
    def test_extract_raw_segments_finds_contiguous_regions_above_raw_threshold(
        self, analyzer: SpeechAnalyzer
    ):
        # Given probabilities with two separate regions above raw_threshold=0.05
        prob_array = np.array([0.01, 0.03, 0.12, 0.25, 0.18, 0.08, 0.02, 0.30, 0.35, 0.01])

        # When
        raw_segments = analyzer.extract_raw_segments(prob_array)

        # Then
        assert len(raw_segments) == 2

        # First raw region: windows 2-5
        seg1 = raw_segments[0]
        expected_probs1 = prob_array[2:6]
        assert seg1.start_sec == pytest.approx(2 * analyzer.step_sec, abs=0.001)
        assert seg1.end_sec == pytest.approx(6 * analyzer.step_sec, abs=0.001)
        assert seg1.avg_probability == pytest.approx(float(expected_probs1.mean()), abs=0.001)

        # Second raw region: windows 7-8
        seg2 = raw_segments[1]
        expected_probs2 = prob_array[7:9]
        assert seg2.start_sec == pytest.approx(7 * analyzer.step_sec, abs=0.001)
        assert seg2.end_sec == pytest.approx(9 * analyzer.step_sec, abs=0.001)
        assert seg2.avg_probability == pytest.approx(float(expected_probs2.mean()), abs=0.001)

    def test_extract_raw_segments_handles_all_below_threshold(self, analyzer: SpeechAnalyzer):
        prob_array = np.array([0.01, 0.02, 0.03, 0.04])

        raw_segments = analyzer.extract_raw_segments(prob_array)

        assert len(raw_segments) == 0

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
