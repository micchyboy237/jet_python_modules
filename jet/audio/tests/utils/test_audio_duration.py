import io
import math
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import soundfile as sf
from jet.audio.audio_duration import (
    get_audio_duration,
    get_audio_duration_friendly,
    seconds_to_hms,
)

try:
    import torch
except ImportError:
    torch = None


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def sample_rate() -> int:
    return 16000


@pytest.fixture
def one_second_wave(sample_rate: int) -> np.ndarray:
    """
    1 second sine wave at 440Hz
    """
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def temp_audio_file(
    tmp_path: Path,
    one_second_wave: np.ndarray,
    sample_rate: int,
) -> Generator[Path, None, None]:
    """
    Creates temporary WAV file and cleans up automatically.
    """
    file_path = tmp_path / "test.wav"
    sf.write(file_path, one_second_wave, sample_rate)
    yield file_path
    # tmp_path auto-cleans


@pytest.fixture
def audio_bytes(one_second_wave: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, one_second_wave, sample_rate, format="WAV")
    return buffer.getvalue()


# ─────────────────────────────────────────────
# Path-based tests
# ─────────────────────────────────────────────


class TestGetAudioDurationFromPath:
    def test_should_return_exact_duration_for_valid_file(
        self,
        temp_audio_file: Path,
    ):
        # Given
        expected = 1.0

        # When
        result = get_audio_duration(temp_audio_file)

        # Then
        assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-3)

    def test_should_raise_file_not_found_for_missing_file(self):
        # Given
        missing = Path("non_existent.wav")

        # When / Then
        with pytest.raises(FileNotFoundError):
            get_audio_duration(missing)


# ─────────────────────────────────────────────
# Bytes tests
# ─────────────────────────────────────────────


class TestGetAudioDurationFromBytes:
    def test_should_return_exact_duration_for_valid_bytes(
        self,
        audio_bytes: bytes,
    ):
        # Given
        expected = 1.0

        # When
        result = get_audio_duration(audio_bytes)

        # Then
        assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-3)

    def test_should_raise_value_error_for_invalid_bytes(self):
        # Given
        invalid_bytes = b"not audio data"

        # When / Then
        with pytest.raises(ValueError):
            get_audio_duration(invalid_bytes)


# ─────────────────────────────────────────────
# NumPy tests
# ─────────────────────────────────────────────


class TestGetAudioDurationFromNumpy:
    def test_should_return_duration_for_numpy_array(
        self,
        one_second_wave: np.ndarray,
        sample_rate: int,
    ):
        # Given
        expected = 1.0

        # When
        result = get_audio_duration(one_second_wave, sample_rate=sample_rate)

        # Then
        assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-6)

    def test_should_raise_value_error_if_missing_sample_rate(
        self,
        one_second_wave: np.ndarray,
    ):
        # When / Then
        with pytest.raises(ValueError):
            get_audio_duration(one_second_wave)


# ─────────────────────────────────────────────
# Torch tests
# ─────────────────────────────────────────────


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestGetAudioDurationFromTorch:
    def test_should_return_duration_for_torch_tensor(
        self,
        one_second_wave: np.ndarray,
        sample_rate: int,
    ):
        # Given
        tensor = torch.tensor(one_second_wave)
        expected = 1.0

        # When
        result = get_audio_duration(tensor, sample_rate=sample_rate)

        # Then
        assert math.isclose(result, expected, rel_tol=0, abs_tol=1e-6)

    def test_should_raise_value_error_if_missing_sample_rate(
        self,
        one_second_wave: np.ndarray,
    ):
        # Given
        tensor = torch.tensor(one_second_wave)

        # When / Then
        with pytest.raises(ValueError):
            get_audio_duration(tensor)


# ─────────────────────────────────────────────
# Friendly formatting tests
# ─────────────────────────────────────────────


class TestFriendlyFormatting:
    def test_should_format_seconds_under_one_hour(self):
        # Given
        seconds = 65
        expected = "1:05"

        # When
        result = seconds_to_hms(seconds)

        # Then
        assert result == expected

    def test_should_format_seconds_over_one_hour(self):
        # Given
        seconds = 3665
        expected = "1:01:05"

        # When
        result = seconds_to_hms(seconds)

        # Then
        assert result == expected

    def test_should_return_zero_for_negative_input(self):
        # Given
        seconds = -10
        expected = "0:00"

        # When
        result = seconds_to_hms(seconds)

        # Then
        assert result == expected

    def test_should_return_friendly_duration_from_path(
        self,
        temp_audio_file: Path,
    ):
        # Given
        expected = "0:01"

        # When
        result = get_audio_duration_friendly(temp_audio_file)

        # Then
        assert result == expected


# ─────────────────────────────────────────────
# Type validation tests
# ─────────────────────────────────────────────


class TestInvalidInputTypes:
    def test_should_raise_type_error_for_unsupported_type(self):
        # Given
        unsupported = 12345

        # When / Then
        with pytest.raises(TypeError):
            get_audio_duration(unsupported)
