# tests/test_load_audio_mono_float32.py
import numpy as np
import pytest

from jet.audio.audio_levels.utils import load_audio_mono_float32


def test_numpy_input_normalization_int16():
    """Given raw int16-like numpy array, when loading, should normalize to [-1,1]"""
    # Given - simulate 16-bit full scale
    raw = np.array([32767, -32767, 0, 16383], dtype=np.int16)
    expected = np.array([1.0, -1.0, 0.0, 0.5], dtype=np.float32)

    # When
    result, sr = load_audio_mono_float32(raw)

    # Then
    np.testing.assert_allclose(result, expected, atol=5e-5, rtol=1e-5)
    assert sr == 44100  # default fallback


def test_empty_array_handling():
    """Given empty array, loader should return empty + default sr"""
    result, sr = load_audio_mono_float32(np.array([]))

    assert len(result) == 0
    assert sr == 44100


def test_unsupported_dtype_raises():
    """Given unsupported dtype (uint8 example), should raise"""
    bad_data = np.array([128, 255, 0], dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported integer dtype"):
        load_audio_mono_float32(bad_data)
