from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
from rich.console import Console

from jet.audio.insights.audio_insights_extractor import extract_audio_insights_and_plots


console = Console()


@pytest.fixture
def sample_audio(tmp_path: Path) -> str:
    """Generate a temporary 3-second sine wave audio file at 22050 Hz."""
    import soundfile as sf

    sr = 22050
    duration = 3.0
    freq = 440.0  # A4 note
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t)

    audio_path = tmp_path / "sine_440.wav"
    sf.write(audio_path, y, sr)
    return str(audio_path)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory."""
    dir_path = tmp_path / "plots"
    dir_path.mkdir()
    return dir_path


class TestExtractAudioInsightsAndPlots:
    def test_returns_expected_insights_structure(self, sample_audio: str, output_dir: Path):
        # Given: a clean temporary output directory and a simple sine wave
        # When: we run the extraction function with default parameters
        insights: Dict[str, Any] = extract_audio_insights_and_plots(
            audio_path=sample_audio,
            output_dir=output_dir,
            n_fft=2048,
            hop_length=512,
        )

        # Then: the returned dictionary should contain all expected keys with correct types
        expected_keys = {
            "duration_sec": float,
            "sample_rate_hz": int,
            "mean_rms_energy": float,
            "mean_zcr": float,
            "mean_spectral_centroid_hz": float,
            "mean_spectral_bandwidth_hz": float,
            "mean_spectral_rolloff_hz": float,
            "mean_spectral_flatness": float,
        }

        assert set(insights.keys()) == set(expected_keys.keys())

        for key, expected_type in expected_keys.items():
            assert isinstance(insights[key], expected_type), f"{key} should be {expected_type}"

        # Additional realistic checks for the sine wave
        assert insights["duration_sec"] == pytest.approx(3.0, abs=0.1)
        assert insights["sample_rate_hz"] == 22050
        assert 0.0 < insights["mean_rms_energy"] < 1.0
        assert insights["mean_zcr"] == pytest.approx(0.08, abs=0.05)  # ~440 Hz tone → low ZCR
        assert 400.0 < insights["mean_spectral_centroid_hz"] < 500.0  # close to fundamental

    def test_creates_expected_plot_files(self, sample_audio: str, output_dir: Path):
        # Given: an empty temporary output directory
        initial_files = set(output_dir.iterdir())

        # When: running the function
        extract_audio_insights_and_plots(
            audio_path=sample_audio,
            output_dir=output_dir,
        )

        # Then: exactly 6 plot files should be created with correct naming
        created_files = set(output_dir.iterdir()) - initial_files
        expected_filenames = {
            "01_waveform_rms_zcr.png",
            "02_spectrogram.png",
            "03_mel_spectrogram.png",
            "04_chromagram.png",
            "05_mfcc.png",
            "06_spectral_descriptors.png",
        }

        actual_filenames = {f.name for f in created_files}
        assert actual_filenames == expected_filenames

        for file_path in created_files:
            assert file_path.stat().st_size > 1000  # basic sanity: not empty

    def test_custom_parameters_are_respected(self, sample_audio: str, output_dir: Path):
        # Given: specific non-default STFT parameters
        custom_hop = 256
        custom_n_fft = 1024

        # When: calling with custom parameters
        insights = extract_audio_insights_and_plots(
            audio_path=sample_audio,
            output_dir=output_dir,
            n_fft=custom_n_fft,
            hop_length=custom_hop,
            n_mels=64,
            n_mfcc=13,
        )

        # Then: insights should still be valid (function completes without error)
        # and plots are generated (implicitly tested by previous test)
        assert isinstance(insights, dict)
        assert "mean_spectral_centroid_hz" in insights