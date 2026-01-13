# tests/test_has_sound.py
import numpy as np
from pathlib import Path
from scipy.io import wavfile

from jet.audio.audio_levels.utils import has_sound


def create_test_wav(
    tmp_path: Path,
    samples: np.ndarray,
    sample_rate: int = 44100,
    name: str = "test.wav"
) -> Path:
    """Helper to create temporary WAV file for testing file-based input"""
    path = tmp_path / name
    
    if samples.dtype in (np.float32, np.float64):
        wavfile.write(path, sample_rate, samples)
    elif samples.dtype == np.int16:
        wavfile.write(path, sample_rate, samples)
    else:
        # Convert to int16 for simplicity in most test cases
        wavfile.write(path, sample_rate, (samples * 32767).astype(np.int16))
    
    return path


class TestHasSound:
    """Tests for the high-level has_sound() function"""

    # ── Direct array input tests ──────────────────────────────────────────────

    def test_loud_sine_should_be_detected(self):
        """Given loud sine wave, when checking has_sound, then returns True"""
        # Given
        sample_rate = 44100
        duration_sec = 1.5
        n_samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        samples = 0.8 * np.sin(2 * np.pi * 440 * t).astype(np.float32)  # ≈ -1.94 dBFS

        # When
        result = has_sound(samples, threshold_db=-40.0)

        # Then
        assert result is True, "Loud sine wave should be detected as sound"


    def test_silence_should_not_be_detected(self):
        """Given pure digital silence, when checking has_sound, then returns False"""
        # Given
        silence = np.zeros(88200, dtype=np.float32)  # 2 seconds of silence

        # When
        result = has_sound(silence, threshold_db=-50.0)

        # Then
        assert result is False, "Digital silence should not be detected"


    def test_very_quiet_long_audio_below_threshold(self):
        """Given very quiet but long audio, when threshold is higher, then False"""
        # Given - ≈ -55 dBFS sine wave
        sample_rate = 44100
        samples = (0.001778 * np.sin(np.linspace(0, 2 * np.pi * 5, sample_rate * 3))).astype(np.float32)

        # When
        result_strict = has_sound(samples, threshold_db=-50.0)
        result_lenient = has_sound(samples, threshold_db=-60.0)

        # Then
        assert result_strict is False, "Should be below -50 dB threshold"
        assert result_lenient is True, "Should be detected with -60 dB threshold"


    def test_very_short_loud_audio_is_rejected(self):
        """Given loud but very short audio, when min_duration_sec is set, then False"""
        # Given - 15 ms of very loud audio
        sample_rate = 44100
        short_duration = 0.015
        n_short = int(sample_rate * short_duration)
        short_loud = np.ones(n_short, dtype=np.float32) * 0.95

        # When
        result_default = has_sound(short_loud)  # default min_duration_sec=0.02
        result_custom = has_sound(short_loud, min_duration_sec=0.005)

        # Then
        assert result_default is False, "Too short with default threshold"
        assert result_custom is True, "Should pass with very low min_duration_sec"


    def test_empty_or_zero_length_input(self):
        """Given empty array, when checking has_sound, then returns False"""
        # Given
        empty = np.array([], dtype=np.float32)

        # When
        result = has_sound(empty)

        # Then
        assert result is False, "Empty array should return False"


    # ── File-based input tests ────────────────────────────────────────────────

    def test_from_wav_file_loud(self, sine_1sec_440hz, tmp_wav_file):
        """Given valid loud WAV file, when checking has_sound via path, then True"""
        # Given
        samples = 0.8 * sine_1sec_440hz  # Use the shared sine fixture for loud audio
        wav_path = tmp_wav_file(samples=samples, name="loud.wav")

        # When
        result = has_sound(wav_path, threshold_db=-45.0)

        # Then
        assert result is True


    def test_from_int16_wav_file_quiet(self, tmp_path):
        """Given 16-bit quiet WAV file, normalization should be respected"""
        # Given - ≈ -52 dBFS in int16
        sample_rate = 48000
        samples_float = 0.0025 * np.sin(np.linspace(0, 2 * np.pi * 8, sample_rate * 2))
        samples_int16 = (samples_float * 32767).astype(np.int16)
        wav_path = create_test_wav(tmp_path, samples_int16, sample_rate)

        # When
        result_strict = has_sound(wav_path, threshold_db=-45.0)
        result_lenient = has_sound(wav_path, threshold_db=-60.0)

        # Then
        assert result_strict is False
        assert result_lenient is True


    def test_has_sound_with_custom_load_options(self, tmp_path):
        """Given file + custom load options (normalize=False), threshold behaves correctly"""
        # Given - file with low level int16 data (not normalized)
        sample_rate = 44100
        low_level = np.ones(44100 * 2, dtype=np.int16) * 500  # very low compared to 32767
        wav_path = create_test_wav(tmp_path, low_level, sample_rate)

        # When - without normalization → RMS will be very low
        result_normalized = has_sound(wav_path, threshold_db=-50.0)  # default normalize=True
        result_raw = has_sound(wav_path, threshold_db=-50.0, normalize=False)

        # Then
        assert result_normalized is True, "With normalization should be detected"
        assert result_raw is False, "Without normalization should be too quiet"