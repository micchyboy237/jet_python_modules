# tests/test_growing_windows.py
from typing import List, Tuple

import pytest

from jet.utils.collection_utils import growing_windows


class TestGrowingWindows:
    def test_standard_example(self):
        # Given
        seq: List[int] = [1, 2, 3, 4, 5]
        max_size: int = 3
        start_offset: int = 1
        step: int = 1

        # When
        result: List[Tuple[int, ...]] = list(
            growing_windows(seq, max_size, start_offset=start_offset, step=step)
        )

        # Then
        expected: List[Tuple[int, ...]] = [(2,), (2, 3), (2, 3, 4)]
        assert result == expected

    def test_larger_step(self):
        # Given
        seq: List[int] = [1, 2, 3, 4, 5, 6, 7]
        max_size: int = 3
        start_offset: int = 2
        step: int = 2

        # When
        result: List[Tuple[int, ...]] = list(
            growing_windows(seq, max_size, start_offset=start_offset, step=step)
        )

        # Then
        expected: List[Tuple[int, ...]] = [(3,), (3, 5), (3, 5, 7)]
        assert result == expected

    def test_shorter_sequence_than_max_size(self):
        # Given
        seq: List[int] = [1, 2, 3]
        max_size: int = 5
        start_offset: int = 1
        step: int = 1

        # When
        result: List[Tuple[int, ...]] = list(
            growing_windows(seq, max_size, start_offset=start_offset, step=step)
        )

        # Then
        expected: List[Tuple[int, ...]] = [(2,), (2, 3)]
        assert result == expected

    def test_start_beyond_sequence(self):
        # Given
        seq: List[int] = [1, 2, 3]
        max_size: int = 3
        start_offset: int = 4
        step: int = 1

        # When
        result: List[Tuple[int, ...]] = list(
            growing_windows(seq, max_size, start_offset=start_offset, step=step)
        )

        # Then
        expected: List[Tuple[int, ...]] = []
        assert result == expected

    def test_empty_sequence(self):
        # Given
        seq: List[int] = []
        max_size: int = 3
        start_offset: int = 0
        step: int = 1

        # When
        result: List[Tuple[int, ...]] = list(
            growing_windows(seq, max_size, start_offset=start_offset, step=step)
        )

        # Then
        expected: List[Tuple[int, ...]] = []
        assert result == expected

    def test_invalid_parameters(self):
        # Given
        seq: List[int] = [1, 2, 3, 4, 5]

        # When / Then
        assert list(growing_windows(seq, max_size=0)) == []
        assert list(growing_windows(seq, max_size=3, step=0)) == []
        assert list(growing_windows(seq, max_size=-1)) == []

    def test_max_size_none_grows_to_end(self):
        # Given
        seq: List[int] = [1, 2, 3, 4, 5, 6]
        max_size: None = None
        start_offset: int = 1
        step: int = 1

        # When
        result: List[Tuple[int, ...]] = list(
            growing_windows(seq, max_size, start_offset=start_offset, step=step)
        )

        # Then
        expected: List[Tuple[int, ...]] = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)]
        assert result == expected

    def test_max_size_none_with_larger_step(self):
        # Given
        seq: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]
        max_size: None = None
        start_offset: int = 2
        step: int = 2

        # When
        result: List[Tuple[int, ...]] = list(
            growing_windows(seq, max_size, start_offset=start_offset, step=step)
        )

        # Then
        expected: List[Tuple[int, ...]] = [(3,), (3, 5), (3, 5, 7)]
        assert result == expected

    def test_separate_offset_and_step(self):
        # Given
        seq: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        max_size: int = 4
        start_offset: int = 2
        step: int = 3

        # When
        result: List[Tuple[int, ...]] = list(
            growing_windows(seq, max_size, start_offset=start_offset, step=step)
        )

        # Then
        expected: List[Tuple[int, ...]] = [(2,), (2, 5), (2, 5, 8)]
        assert result == expected

    def test_default_offset_zero(self):
        # Given
        seq: List[int] = [1, 2, 3, 4, 5]
        max_size: int = 3

        # When
        result: List[Tuple[int, ...]] = list(growing_windows(seq, max_size))

        # Then
        expected: List[Tuple[int, ...]] = [(1,), (1, 2), (1, 2, 3)]
        assert result == expected

    def test_negative_offset_raises(self):
        # Given
        seq: List[int] = [1, 2, 3]

        # When / Then
        with pytest.raises(ValueError, match="start_offset must be non-negative"):
            list(growing_windows(seq, start_offset=-1))