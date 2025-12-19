import pytest
import numpy as np
from jet.audio.insights.audioflux_audio_insights_extractor import AudioFluxInsightsExtractor
from rich.console import Console
from typing import Dict

console = Console()

@pytest.fixture
def extractor() -> AudioFluxInsightsExtractor:
    """Fixture for extractor instance."""
    return AudioFluxInsightsExtractor(metrics=['spectral_centroid', 'mfcc', 'rms'])

@pytest.fixture
def sample_audio_path() -> str:
    """Path to a sample audio file (replace with your own WAV path)."""
    # e.g., a 5s guitar chord WAV
    audio_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_live_subtitles/full_recording.wav"
    return audio_path

def test_load_audio(extractor: AudioFluxInsightsExtractor, sample_audio_path: str):
    """Given a valid audio file, when loading, then return mono array and sample rate."""
    # Given
    expected_sr = 16000  # Matches actual sample file rate

    # When
    audio_arr, sr = extractor.load_audio(sample_audio_path)

    # Then
    assert audio_arr.ndim == 1, "Audio should be mono"
    assert sr == expected_sr, f"Sample rate mismatch: got {sr}, expected {expected_sr}"
    assert len(audio_arr) > 0, "Audio array is empty"

def test_extract_insights(extractor: AudioFluxInsightsExtractor, sample_audio_path: str):
    """Given an audio file, when extracting insights, then return dict with correct types/shapes."""
    # Given
    expected_metrics = ['spectral_centroid', 'mfcc', 'rms']

    # When
    insights: Dict = extractor.extract_insights(sample_audio_path, cc_num=13, radix2_exp=12)

    # Then
    assert set(insights.keys()) == set(expected_metrics), "Missing or extra metrics"
    assert isinstance(insights['spectral_centroid'], (float, np.float64)), "Centroid should be scalar mean"
    assert insights['mfcc'].shape == (13,), "MFCC mean should be (cc_num,)"
    assert isinstance(insights['rms'], (float, np.float64)), "RMS should be scalar mean"
    assert all(isinstance(v, (np.ndarray, float, np.float64)) for v in insights.values())

def test_unsupported_metric(extractor: AudioFluxInsightsExtractor, sample_audio_path: str):
    """Given an unsupported metric, when extracting, then raise ValueError."""
    # Given
    bad_extractor = AudioFluxInsightsExtractor(metrics=['invalid_metric'])

    # When/Then
    with pytest.raises(ValueError, match="Unsupported metric"):
        bad_extractor.extract_insights(sample_audio_path)

def test_pitch_extraction(sample_audio_path: str):
    """Given a pitched audio, when extracting pitch, then return mean > 0."""
    # Given
    extractor_with_pitch = AudioFluxInsightsExtractor(metrics=['pitch'])

    # When
    insights: Dict = extractor_with_pitch.extract_insights(sample_audio_path)

    # Then
    pitch_mean = insights['pitch']
    assert pitch_mean > 0, "Pitch mean should be positive for pitched audio (adjust sample if silent)"

# Cleanup not needed as no files are written