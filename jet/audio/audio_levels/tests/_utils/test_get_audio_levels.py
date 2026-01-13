# tests/test_get_audio_levels.py
import numpy as np
import pytest
from scipy.io import wavfile
from pathlib import Path

from jet.audio.audio_levels.utils import (
    get_audio_levels,
)


def create_test_wav(
    tmp_path: Path,
    samples: np.ndarray,
    sample_rate: int = 44100,
    dtype: type[np.generic] = np.float32,
    filename: str = "test.wav"
) -> Path:
    """Helper: create temporary WAV file for testing file-based input"""
    path = tmp_path / filename
    
    # wavfile.write expects int16/int32/float32/float64
    if dtype == np.float32:
        wavfile.write(path, sample_rate, samples)
    elif dtype == np.int16:
        wavfile.write(path, sample_rate, (samples * 32767).astype(np.int16))
    else:
        raise ValueError(f"Unsupported dtype for test file: {dtype}")
        
    return path


class TestGetAudioLevels:

    # ── Basic correctness with numpy array input ──────────────────────────────

    @pytest.mark.parametrize("amplitude, expected_dbfs", [
        (1.0,       0.0),                    # full scale sine
        (0.70710678, -3.0),                  # ≈ -3 dBFS (1/√2)
        (0.5,       -6.020599913279624),     # ≈ -6 dB
        (0.1,       -20.0),
        (0.0316227766, -30.0),
        (0.0,       float('-inf')),          # silence
    ], ids=["0dB", "-3dB", "-6dB", "-20dB", "-30dB", "silence"])
    def test_known_sine_amplitudes(self, amplitude, expected_dbfs):
        """Given sine wave with known amplitude, when getting levels, then RMS and dBFS are correct"""
        # Given - 1 second 440 Hz sine wave
        sample_rate = 44100
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        samples = amplitude * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        expected_rms = amplitude / np.sqrt(2) if amplitude > 0 else 0.0

        # When
        result = get_audio_levels(samples)

        # Then
        assert result["sample_rate"] == sample_rate
        assert result["duration_seconds"] == pytest.approx(1.0, abs=0.001)

        assert result["rms_linear"] == pytest.approx(expected_rms, abs=1e-7)
        
        if expected_dbfs == float('-inf'):
            assert result["dbfs"] == float('-inf')
        else:
            assert result["dbfs"] == pytest.approx(expected_dbfs, abs=0.01)


    def test_zero_duration(self):
        """Given empty array, when getting levels, then returns safe defaults"""
        # Given
        empty = np.array([], dtype=np.float32)

        # When
        result = get_audio_levels(empty)

        # Then
        assert result["rms_linear"] == 0.0
        assert result["dbfs"] == float('-inf')
        assert result["duration_seconds"] == 0.0
        assert result["sample_rate"] == 44100  # fallback value


    # ── File-based input tests ────────────────────────────────────────────────

    def test_from_wav_file_sine_0dbfs(self, tmp_path):
        """Given full-scale sine WAV file, when loading via path, then levels are correct"""
        # Given
        sample_rate = 44100
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # amplitude 1.0 → 0 dBFS
        wav_path = create_test_wav(tmp_path, samples, sample_rate)

        # When
        result = get_audio_levels(str(wav_path))

        # Then
        assert result["rms_linear"] == pytest.approx(1.0 / np.sqrt(2), abs=1e-6)
        assert result["dbfs"] == pytest.approx(0.0, abs=0.01)
        assert result["sample_rate"] == sample_rate
        assert result["duration_seconds"] == pytest.approx(1.0, abs=0.001)


    def test_from_int16_wav_file(self, tmp_path):
        """Given 16-bit integer WAV file, when loading, then correctly normalized"""
        # Given - -6 dBFS sine stored as int16
        sample_rate = 44100
        t = np.linspace(0, 1, sample_rate, endpoint=False)
        float_samples = 0.5 * np.sin(2 * np.pi * 440 * t)
        int16_samples = (float_samples * 32767).astype(np.int16)
        wav_path = tmp_path / "int16.wav"
        wavfile.write(wav_path, sample_rate, int16_samples)
        # When
        result = get_audio_levels(wav_path)
        expected_rms = 0.5 / np.sqrt(2)   # ≈ 0.35355339
        assert result["rms_linear"] == pytest.approx(expected_rms, abs=1e-6)
        assert result["dbfs"] == pytest.approx(-6.02, abs=0.05)


    # ── Edge cases ─────────────────────────────────────────────────────────────

    def test_very_short_audio(self, tmp_path):
        """Given very short audio (< 50 ms), when getting levels, then duration is correct"""
        # Given - 20 ms of loud audio
        sample_rate = 44100
        short_length = int(sample_rate * 0.02)  # 20 ms
        samples = np.ones(short_length, dtype=np.float32) * 0.9
        
        wav_path = create_test_wav(tmp_path, samples, sample_rate, filename="short.wav")

        # When
        result = get_audio_levels(wav_path)

        # Then
        assert result["duration_seconds"] == pytest.approx(0.02, abs=0.001)
        assert result["rms_linear"] == pytest.approx(0.9, abs=1e-6)