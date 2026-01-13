# tests/test_audio_dbfs.py
import pytest
import numpy as np

from jet.audio.audio_levels.dbfs import get_audio_dbfs


@pytest.fixture
def full_scale_sine():
    t = np.linspace(0, 1, 44100, endpoint=False)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def minus_12db_sine(full_scale_sine):
    return full_scale_sine * 10**(-12 / 20)


@pytest.fixture
def silent():
    return np.zeros(4096)


def test_full_scale_peak_should_be_almost_0(full_scale_sine):
    result = get_audio_dbfs(full_scale_sine)
    expected = 0.0  # very close
    assert abs(result - expected) < 0.001


def test_minus_12db_peak_should_be_minus_12(minus_12db_sine):
    result = get_audio_dbfs(minus_12db_sine)
    expected = -12.0
    assert abs(result - expected) < 0.01


def test_rms_of_full_scale_sine(full_scale_sine):
    result = get_audio_dbfs(full_scale_sine, metric="rms")
    expected = -3.0103
    assert abs(result - expected) < 0.001


def test_silent_signal_returns_negative_infinity(silent):
    result = get_audio_dbfs(silent)
    assert result == -np.inf


def test_tuple_return_gives_both_values(full_scale_sine):
    peak, rms = get_audio_dbfs(full_scale_sine, return_type="tuple")
    assert peak > -0.1
    assert -3.1 < rms < -3.0


def test_very_small_signal_is_very_negative():
    tiny = np.ones(1000) * 1e-5
    result = get_audio_dbfs(tiny)
    assert result < -90