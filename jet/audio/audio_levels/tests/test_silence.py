# tests/test_silence.py
import pytest
import numpy as np

from jet.audio.audio_levels.silence import is_silent


@pytest.fixture
def sr() -> int:
    return 44100


@pytest.fixture
def full_scale_sine(sr):
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    return 0.999 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def very_quiet_noise(sr):
    return np.random.normal(0, 0.001, sr // 2)  # ≈ -60 dBFS RMS


@pytest.fixture
def pure_silence(sr):
    return np.zeros(sr // 2)


def test_given_complete_silence_when_checked_then_is_silent(pure_silence):
    # Given complete silence
    # When checking with default parameters
    result = is_silent(pure_silence)

    # Then should be detected as silent
    expected = True
    assert result == expected


def test_given_loud_signal_when_checked_then_is_not_silent(full_scale_sine):
    # Given nearly full-scale signal
    # When using typical threshold
    result = is_silent(full_scale_sine, threshold_db=-40.0)

    # Then should not be considered silent
    expected = False
    assert result == expected


def test_given_quiet_noise_when_using_peak_then_is_silent(very_quiet_noise):
    # Given very quiet noise (~ -60 dBFS RMS, peak even lower)
    # When using realistic peak detection with -45 dBFS threshold
    result = is_silent(very_quiet_noise, threshold_db=-45.0)

    # Then should be detected as silent
    expected = True
    assert result == expected


def test_given_quiet_noise_when_using_rms_then_may_not_be_silent(very_quiet_noise):
    # Given very quiet noise
    # When using RMS mode with stricter threshold
    result = is_silent(very_quiet_noise, threshold_db=-55.0, use_rms=True)

    # Then should be detected as silent (RMS ≈ -60 dBFS)
    expected = True
    assert result == expected


def test_given_short_silence_when_min_duration_required_then_not_silent(sr):
    # Given very short silence (10ms)
    short = np.zeros(int(0.01 * sr))

    # When requiring minimum duration of 0.1s
    result = is_silent(short, min_duration_sec=0.1, sample_rate=sr)

    # Then should NOT be considered silent
    expected = False
    assert result == expected


def test_given_empty_array_when_checked_then_is_silent():
    # Given empty audio array
    empty = np.array([])

    # When checking silence
    result = is_silent(empty)

    # Then should be considered silent
    expected = True
    assert result == expected