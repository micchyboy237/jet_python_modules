"""
Unit tests for audio_waveform.py

Covers:
- CircularBuffer behavior
- Data ordering
- Overflow correctness
- Type normalization
- Error cases
"""

import numpy as np
import pytest
from jet.audio.audio_waveform import CircularBuffer

# -----------------------------------------------------------------------------
# CircularBuffer Tests
# -----------------------------------------------------------------------------


class TestCircularBuffer:
    """
    Tests for generic circular buffer behavior.
    """

    # -------------------------------------------------------------------------
    # Given empty buffer
    # When to_array is called
    # Then it should return empty float32 array
    # -------------------------------------------------------------------------
    def test_empty_buffer(self):
        buf = CircularBuffer(5)

        result = buf.to_array()
        expected = np.array([], dtype=np.float32)

        assert np.array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Given buffer size 5
    # When appending fewer elements
    # Then order must be preserved exactly
    # -------------------------------------------------------------------------
    def test_append_within_capacity(self):
        buf = CircularBuffer(5)

        buf.append(np.array([1, 2, 3], dtype=np.float64))

        result = buf.to_array()
        expected = np.array([1, 2, 3], dtype=np.float32)

        assert np.array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Given buffer size 3
    # When appending 5 elements
    # Then only last 3 remain
    # -------------------------------------------------------------------------
    def test_overflow_keeps_latest(self):
        buf = CircularBuffer(3)

        buf.append(np.array([1, 2, 3, 4, 5]))

        result = buf.to_array()
        expected = np.array([3, 4, 5], dtype=np.float32)

        assert np.array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Given multiple appends
    # When total exceeds capacity
    # Then sliding window must be correct
    # -------------------------------------------------------------------------
    def test_multiple_appends(self):
        buf = CircularBuffer(4)

        buf.append(np.array([1, 2]))
        buf.append(np.array([3, 4, 5]))

        result = buf.to_array()
        expected = np.array([2, 3, 4, 5], dtype=np.float32)

        assert np.array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Given scalar append
    # When adding single value
    # Then buffer should contain that value
    # -------------------------------------------------------------------------
    def test_scalar_append(self):
        buf = CircularBuffer(3)

        buf.append(0.75)

        result = buf.to_array()
        expected = np.array([0.75], dtype=np.float32)

        assert np.array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Given mixed scalar + array appends
    # When exceeding capacity
    # Then ordering must be preserved
    # -------------------------------------------------------------------------
    def test_mixed_append_types(self):
        buf = CircularBuffer(3)

        buf.append(1.0)
        buf.append(np.array([2.0, 3.0]))
        buf.append(4.0)

        result = buf.to_array()
        expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)

        assert np.array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Given invalid buffer size
    # When initializing with 0
    # Then ValueError should be raised
    # -------------------------------------------------------------------------
    def test_invalid_buffer_size(self):
        with pytest.raises(ValueError):
            CircularBuffer(0)

    # -------------------------------------------------------------------------
    # Given large append
    # When adding many values
    # Then final array length equals max_len
    # -------------------------------------------------------------------------
    def test_large_append_exact_capacity(self):
        buf = CircularBuffer(100)

        data = np.arange(1000, dtype=np.float32)
        buf.append(data)

        result = buf.to_array()
        expected = data[-100:].astype(np.float32)

        assert np.array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Given negative values
    # When appending
    # Then they must be preserved exactly
    # -------------------------------------------------------------------------
    def test_negative_values_preserved(self):
        buf = CircularBuffer(4)

        buf.append(np.array([-1, -2, -3]))

        result = buf.to_array()
        expected = np.array([-1, -2, -3], dtype=np.float32)

        assert np.array_equal(result, expected)

    # -------------------------------------------------------------------------
    # Given float precision input
    # When converting to array
    # Then dtype must be float32
    # -------------------------------------------------------------------------
    def test_dtype_is_float32(self):
        buf = CircularBuffer(3)

        buf.append(np.array([1.123456789]))

        result = buf.to_array()

        assert result.dtype == np.float32
