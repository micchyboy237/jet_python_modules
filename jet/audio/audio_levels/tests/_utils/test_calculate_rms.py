# tests/test_calculate_rms.py
import numpy as np
import pytest

from jet.audio.audio_levels.utils import calculate_rms


@pytest.mark.parametrize("amplitude", [1.0, 0.7071, 0.5, 0.1, 0.0])
def test_rms_of_standard_sine(sine_1sec_440hz, amplitude):
    """Using shared sine fixture: RMS of sine of varying amplitude"""
    samples = amplitude * sine_1sec_440hz
    result = calculate_rms(samples)
    expected = amplitude / np.sqrt(2) if amplitude > 0 else 0.0
    assert result == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize(
    "samples, expected_rms",
    [
        (np.array([0.0]), 0.0),
        (np.array([1.0, -1.0]), 1.0),
        (np.array([0.5, 0.5, -0.5, -0.5]), 0.5),
        (np.array([]), 0.0),
        (np.ones(100000) * 1e-6, pytest.approx(1e-6, abs=1e-9)),
        (np.concatenate([np.ones(5000), np.zeros(5000)]), 0.70710678118),
    ],
    ids=[
        "single_zero",
        "full_scale_square",
        "half_scale_square",
        "empty_array",
        "extremely_quiet",
        "half_silence_half_full",
    ]
)
def test_calculate_rms_various_signals(samples, expected_rms):
    """Given different simple signals, when calculating RMS, then result matches expectation"""
    result = calculate_rms(samples)
    assert result == pytest.approx(expected_rms, abs=1e-9, rel=1e-6)


def test_calculate_rms_gaussian_noise_statistical():
    """Statistical test: RMS of long noise approaches target stddev"""
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 0.3, 44100)
    result = calculate_rms(samples)
    assert result == pytest.approx(0.3, rel=0.15)


def test_calculate_rms_input_not_modified():
    """Given an array, after RMS calculation the input should remain unchanged"""
    original = np.array([0.8, -0.4, 0.2, 0.0], dtype=np.float64)
    copy_before = original.copy()
    _ = calculate_rms(original)
    np.testing.assert_array_equal(original, copy_before)
