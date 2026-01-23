# tests/test_increasing_window.py
from typing import List, Optional, Union
import pytest

from jet.wordnet.utils import increasing_window


@pytest.mark.parametrize(
    "tokens, step_size, max_window_size, expected",
    [
        # Given: empty sequence
        # When: we generate windows
        # Then: we get no results
        ("", 1, None, []),
        ([], 1, None, []),

        # Given: short string, default step
        # When: generate windows
        # Then: we get all increasing prefixes
        ("abc", 1, None, ["a", "ab", "abc"]),

        # Given: list, default step
        ([10, 20, 30, 40], 1, None, [[10], [10,20], [10,20,30], [10,20,30,40]]),

        # Given: step_size = 2
        # When: generate windows
        # Then: sizes jump accordingly
        ("abcdef", 2, None, ["a", "abc", "abcde"]),

        # Given: max_window_size constraint
        # When: generate windows
        # Then: we stop at or before max_window_size
        ("abcdefgh", 1, 5, ["a", "ab", "abc", "abcd", "abcde"]),

        # Given: max_window_size smaller than first window
        # When: generate windows
        # Then: no windows produced
        ("hello", 1, 0, []),

        # Given: max_window_size between steps
        ("abcdefghijk", 3, 7, ["a", "abcd", "abcdefg"]),
    ]
)
def test_increasing_window(
    tokens: Union[str, List],
    step_size: int,
    max_window_size: Optional[int],
    expected: List[Union[str, List]]
):
    # Given
    expected_windows = expected

    # When
    result = list(increasing_window(tokens, step_size, max_window_size))

    # Then
    assert result == expected_windows, f"Expected {expected_windows!r}, got {result!r}"


def test_negative_or_zero_step_size_raises():
    # Given + When + Then
    with pytest.raises(ValueError, match="step_size must be positive"):
        list(increasing_window("abc", step_size=0))

    with pytest.raises(ValueError, match="step_size must be positive"):
        list(increasing_window([1,2,3], step_size=-1))