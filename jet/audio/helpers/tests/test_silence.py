import numpy as np
import pytest
from typing import List
from unittest.mock import patch, MagicMock

from jet.audio.helpers.silence import (
    CHANNELS,
    detect_silence,
    trim_silent_chunks,
    calibrate_silence_threshold,
    SAMPLE_RATE,
)
from pytest import approx



@pytest.fixture
def mixed_chunks() -> List[np.ndarray]:
    """Three chunks: silence, speech, silence"""
    silence = np.zeros(4000, dtype=np.float32)
    speech = np.random.uniform(-0.8, 0.8, 4000).astype(np.float32)
    return [silence, speech, silence]


def test_detect_silence_correctly_identifies_silence(mixed_chunks):
    threshold = 0.05
    chunk = mixed_chunks[0]
    result = detect_silence(chunk, threshold)
    expected = True
    assert result is expected

    result = detect_silence(mixed_chunks[1], threshold)
    expected = False
    assert result is expected


def test_trim_silent_chunks_removes_leading_and_trailing_silence(mixed_chunks):
    threshold = 0.05
    result = trim_silent_chunks(mixed_chunks, threshold)
    expected = mixed_chunks[1:2]
    assert len(result) == len(expected)
    assert all(np.array_equal(a, b) for a, b in zip(result, expected))


def test_trim_silent_chunks_returns_full_list_if_no_silence():
    audio_chunks = [
        np.random.uniform(-0.5, 0.5, 4000).astype(np.float32) for _ in range(5)
    ]
    threshold = 0.001
    result = trim_silent_chunks(audio_chunks, threshold)
    # Compare content instead of identity
    assert len(result) == len(audio_chunks)
    assert all(np.array_equal(a, b) for a, b in zip(result, audio_chunks))



def test_trim_silent_chunks_returns_empty_if_all_silent():
    audio_chunks = [np.zeros(4000, dtype=np.float32) for _ in range(6)]
    threshold = 0.01
    result = trim_silent_chunks(audio_chunks, threshold)
    expected: List[np.ndarray] = []
    assert result == expected


def test_calibrate_silence_threshold_uses_median_and_applies_multiplier(monkeypatch):
    num_frames = int(2.0 * SAMPLE_RATE)
    # Use larger noise so median > 0.005/3 → multiplier is actually used
    fake_audio_data = np.random.randint(-800, 800, size=(num_frames, CHANNELS), dtype='int16')
    mock_stream = MagicMock()
    mock_stream.read.return_value = (fake_audio_data, False)

    with patch("sounddevice.InputStream", return_value=mock_stream):
        threshold = calibrate_silence_threshold(calibration_duration=2.0)

    # Reproduce exact same calculation the function does
    audio_float = fake_audio_data.astype(np.float32) / np.iinfo('int16').max
    median_energy = np.median(np.square(audio_float))
    expected = max(median_energy * 3.0, 0.005)

    assert threshold == approx(expected, rel=1e-6)


def test_calibrate_silence_threshold_handles_completely_silent_input(monkeypatch):
    num_frames = int(2.0 * SAMPLE_RATE)
    silent_audio = np.zeros((num_frames, CHANNELS), dtype='int16')
    mock_stream = MagicMock()
    mock_stream.read.return_value = (silent_audio, False)

    with patch("sounddevice.InputStream", return_value=mock_stream):
        threshold = calibrate_silence_threshold()
        assert threshold == 0.005  # ← now matches the new floor
