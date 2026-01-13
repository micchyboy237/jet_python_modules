# tests/test_audio_rms.py
import pytest
import numpy as np

from jet.audio.audio_levels.rms import get_audio_rms


@pytest.fixture
def full_scale_sine():
    t = np.linspace(0, 1, 44100, endpoint=False)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def minus_18db_sine(full_scale_sine):
    return full_scale_sine * 10**(-18/20)


@pytest.fixture
def stereo_signal(full_scale_sine):
    return np.vstack([full_scale_sine * 0.8, full_scale_sine * 0.4])


def test_full_scale_sine_rms_is_minus_3db(full_scale_sine):
    # Given a full-scale sine wave
    # When calculating RMS in dBFS
    result = get_audio_rms(full_scale_sine)

    # Then RMS should be very close to -3.01 dBFS
    expected = -3.010299956639812
    assert abs(result - expected) < 0.001


def test_minus_18db_sine_has_rms_minus_21db(minus_18db_sine):
    # Given a sine wave -18 dB below full scale
    # When calculating RMS
    result = get_audio_rms(minus_18db_sine)

    # Then RMS should be ≈ -21.01 dBFS
    expected = -21.0103
    assert abs(result - expected) < 0.001


def test_zero_signal_returns_negative_infinity():
    # Given silent signal
    zeros = np.zeros(2048)

    # When calculating RMS in dB
    result = get_audio_rms(zeros)

    # Then should return -inf
    assert result == -np.inf


def test_linear_rms_full_scale_sine(full_scale_sine):
    # Given full scale sine
    # When asking for linear RMS
    result = get_audio_rms(full_scale_sine, return_db=False)

    # Then should be very close to 1/√2 ≈ 0.7071
    expected = 0.7071067811865475
    assert abs(result - expected) < 1e-8


def test_stereo_returns_per_channel_rms(stereo_signal):
    # Given stereo signal with different levels
    # When calculating RMS
    result = get_audio_rms(stereo_signal)

    # Then should return array with two values
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    assert result[0] > result[1]  # left is louder


def test_custom_reference():
    # Given signal with peak 0.5
    signal = np.ones(1000) * 0.5

    # When using custom reference = 0.5
    result = get_audio_rms(signal, reference="custom", custom_ref=0.5)

    # Then RMS should be 0 dB relative to reference
    assert abs(result) < 0.001