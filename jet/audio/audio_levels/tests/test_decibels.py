# tests/test_audio_decibels.py
import pytest
import numpy as np

from jet.audio.audio_levels.decibels import get_audio_decibels


# Fixtures
@pytest.fixture
def full_scale_sine():
    """1 second 440 Hz sine wave at almost full scale"""
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    return 0.999 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def minus_12db_sine(full_scale_sine):
    return full_scale_sine * 10**(-12 / 20)


@pytest.fixture
def very_quiet_sine(full_scale_sine):
    return full_scale_sine * 1e-5  # ≈ -100 dBFS


@pytest.fixture
def stereo_signal(full_scale_sine):
    left = full_scale_sine * 0.8
    right = full_scale_sine * 0.5
    return np.vstack([left, right])


# ─── Tests ────────────────────────────────────────────────────────────────

def test_full_scale_sine_peak_should_be_almost_zero(full_scale_sine):
    # Given a nearly full-scale sine wave
    # When we measure peak level with default settings (dBFS)
    result = get_audio_decibels(full_scale_sine)

    # Then peak should be very close to 0 dBFS
    expected = -0.0087  # 20*log10(0.999) ≈ -0.0087 dB
    assert abs(result - expected) < 0.001


def test_full_scale_sine_rms_should_be_almost_minus_3db(full_scale_sine):
    # Given a nearly full-scale sine wave
    # When we ask for RMS level with full_scale reference
    _, rms_db = get_audio_decibels(full_scale_sine, return_type="tuple")

    # Then RMS should be very close to -3.01 dBFS
    expected = -3.019               # more accurate for 0.999 peak
    assert abs(rms_db - expected) < 0.001


def test_minus_12db_sine_returns_minus_12_peak(minus_12db_sine):
    # Given a sine wave attenuated exactly -12 dB
    # When measuring peak level
    result = get_audio_decibels(minus_12db_sine)

    # Then peak should be ≈ -12 dBFS
    expected = -12.0
    assert abs(result - expected) < 0.02  # small tolerance for float math


def test_very_quiet_signal_returns_large_negative_value(very_quiet_sine):
    # Given a very quiet signal (~ -100 dBFS)
    # When measuring level
    result = get_audio_decibels(very_quiet_sine)

    # Then we should get a large negative value
    expected_range = (-100.2, -99.8)
    assert expected_range[0] <= result <= expected_range[1]


def test_empty_array_returns_negative_infinity():
    # Given an empty audio array
    empty = np.array([])

    # When measuring level
    result = get_audio_decibels(empty)

    # Then we should get -inf
    assert result == -np.inf


def test_zero_array_returns_negative_infinity():
    # Given all zeros
    zeros = np.zeros(1024)

    # When measuring
    result = get_audio_decibels(zeros)

    # Then we should get -inf (because we protect against log(0))
    assert result == -np.inf


def test_custom_reference_point():
    # Given a signal at 0.5 peak amplitude
    signal = np.ones(1000) * 0.5

    # When we use custom reference = 0.5 (so it should be 0 dB)
    result = get_audio_decibels(signal, reference="custom", custom_ref=0.5)

    # Then result should be very close to 0 dB
    assert abs(result) < 0.001


def test_stereo_returns_average_or_first_channel_by_default(stereo_signal):
    # Given stereo signal with different levels per channel
    # When we ask for single value (default behavior)
    result = get_audio_decibels(stereo_signal)

    # Then it should be the peak of the louder channel (left ≈ -1.94 dB)
    expected = 20 * np.log10(0.8 * 0.999)
    assert abs(result - expected) < 0.01


def test_tuple_return_gives_both_peak_and_rms(full_scale_sine):
    # Given full scale sine
    # When requesting tuple
    peak, rms = get_audio_decibels(full_scale_sine, return_type="tuple")

    # Then both values should make sense
    assert peak > -0.1
    assert -3.1 < rms < -3.0


def test_rms_reference_gives_0_for_full_scale_sine_rms(full_scale_sine):
    # Given full scale sine
    # When using rms reference
    result = get_audio_decibels(full_scale_sine, reference="rms")

    # Then peak should be ≈ +3.01 dB (because 0 dB RMS = -3.01 dBFS peak)
    expected = 3.01
    assert abs(result - expected) < 0.02