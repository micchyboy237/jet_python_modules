# tests/conftest.py
import numpy as np
import pytest
from typing import Callable
from pathlib import Path
from scipy.io import wavfile


@pytest.fixture
def sine_1sec_440hz(request) -> np.ndarray:
    """Fixture: 1 second 440 Hz sine wave at requested amplitude"""
    amplitude = getattr(request, 'param', 1.0)
    sample_rate = 44100
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def silence_2sec() -> np.ndarray:
    """Fixture: 2 seconds of digital silence @ 44.1kHz"""
    return np.zeros(44100 * 2, dtype=np.float32)


@pytest.fixture
def tmp_wav_file(tmp_path: Path) -> Callable:
    """
    Factory fixture: create temporary WAV file and return its path
    Usage:
        wav_path = tmp_wav_file(samples=samples, sample_rate=48000, name="test.wav")
    """
    def _create_wav(
        samples: np.ndarray,
        sample_rate: int = 44100,
        name: str = "test.wav",
        dtype_for_write: str | np.dtype = "auto"
    ) -> Path:
        path = tmp_path / name

        if dtype_for_write == "auto":
            if samples.dtype in (np.float32, np.float64):
                wavfile.write(path, sample_rate, samples)
            else:
                wavfile.write(path, sample_rate, samples)
        else:
            wavfile.write(path, sample_rate, samples.astype(dtype_for_write))

        return path

    return _create_wav
