import numpy as np
import pytest
from typing import List
from unittest.mock import patch, MagicMock

from jet.audio.helpers.silence import (
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
    fake_audio_data = np.random.randint(-50, 50, size=(num_frames, 1), dtype='int16')

    mock_stream = MagicMock()
    mock_stream.read.return_value = (fake_audio_data, False)

    # patch the exact number of channels your code actually uses
    monkeypatch.setattr("jet.audio.helpers.silence.CHANNELS", 2)

    with patch("sounddevice.InputStream", return_value=mock_stream):
        threshold = calibrate_silence_threshold(calibration_duration=2.0)

        abs_data = np.abs(fake_audio_data.flatten())
        expected_threshold = max(np.median(abs_data) * 1.5, 0.01)
        assert threshold == approx(expected_threshold, rel=1e-6)


def test_calibrate_silence_threshold_handles_completely_silent_input(monkeypatch):
    monkeypatch.setattr("jet.audio.helpers.silence.CHANNELS", 2)
    num_frames = int(2.0 * SAMPLE_RATE)
    silent_audio = np.zeros((num_frames, 1), dtype='int16')

    mock_stream = MagicMock()
    mock_stream.read.return_value = (silent_audio, False)

    with patch("sounddevice.InputStream", return_value=mock_stream):
        threshold = calibrate_silence_threshold()
        assert threshold == 0.01