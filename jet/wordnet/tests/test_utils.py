import pytest
from typing import List, Any
from jet.wordnet.utils import sliding_window, increasing_window


@pytest.fixture
def sample_list() -> List[int]:
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_text() -> str:
    return "I am learning Python programming"


class TestSlidingWindow:
    def test_window_size_1(self, sample_list: List[int]) -> None:
        # Given a list and window parameters
        items = sample_list
        window_size = 1
        step_size = 1
        expected = [[1], [2], [3], [4], [5]]

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result matches the expected output
        assert result == expected

    def test_window_size_2(self, sample_list: List[int]) -> None:
        # Given a list and window parameters
        items = sample_list
        window_size = 2
        step_size = 1
        expected = [[1, 2], [2, 3], [3, 4], [4, 5]]

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result matches the expected output
        assert result == expected

    def test_window_size_equals_step(self, sample_list: List[int]) -> None:
        # Given a list and window parameters
        items = sample_list
        window_size = 2
        step_size = 2
        expected = [[1, 2], [3, 4], [5]]

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result matches the expected output
        assert result == expected

    def test_window_size_less_than_step(self, sample_list: List[int]) -> None:
        # Given a list and window parameters
        items = sample_list
        window_size = 2
        step_size = 3
        expected = [[1, 2], [4, 5]]

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result matches the expected output
        assert result == expected

    def test_window_size_equals_list(self, sample_list: List[int]) -> None:
        # Given a list and window parameters
        items = sample_list
        window_size = 5
        step_size = 1
        expected = [[1, 2, 3, 4, 5]]

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result matches the expected output
        assert result == expected

    def test_window_size_larger_than_list(self) -> None:
        # Given a short list and large window size
        items = [1, 2]
        window_size = 10
        step_size = 1
        expected = [[1, 2]]

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result matches the expected output
        assert result == expected

    def test_step_size_1(self, sample_list: List[int]) -> None:
        # Given a list and window parameters
        items = sample_list
        window_size = 3
        step_size = 1
        expected = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result matches the expected output
        assert result == expected

    def test_step_size_2(self, sample_list: List[int]) -> None:
        # Given a list and window parameters
        items = sample_list
        window_size = 3
        step_size = 2
        expected = [[1, 2, 3], [3, 4, 5]]

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result matches the expected output
        assert result == expected

    def test_step_size_3(self, sample_list: List[int]) -> None:
        # Given a list and window parameters
        items = sample_list
        window_size = 3
        step_size = 3
        expected = [[1, 2, 3], [4, 5]]

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result matches the expected output
        assert result == expected

    def test_step_size_4(self, sample_list: List[int]) -> None:
        # Given a list and window parameters
        items = sample_list
        window_size = 3
        step_size = 4
        expected = [[1, 2, 3], [4, 5]]

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result matches the expected output
        assert result == expected

    def test_empty_list(self) -> None:
        # Given an empty list
        items: List[Any] = []
        window_size = 3
        step_size = 1
        expected = []

        # When sliding_window is called
        result = list(sliding_window(items, window_size, step_size))

        # Then the result is empty
        assert result == expected


class TestIncreasingWindow:
    def test_increasing_window_basic(self, sample_text: str) -> None:
        # Given a text and window parameters
        text = sample_text
        max_window_size = 3
        step_size = 1
        expected = [
            ["I"],
            ["I", "am"],
            ["I", "am", "learning"],
            ["am"],
            ["am", "learning"],
            ["am", "learning", "Python"],
            ["learning"],
            ["learning", "Python"],
            ["learning", "Python", "programming"],
            ["Python"],
            ["Python", "programming"],
            ["programming"]
        ]

        # When increasing_window is called
        result = list(increasing_window(text, step_size, max_window_size))

        # Then the result matches the expected output
        assert result == expected

    def test_increasing_window_max_size_none(self, sample_text: str) -> None:
        # Given a text with no max window size
        text = sample_text
        max_window_size = None
        step_size = 1
        expected = [
            ["I"],
            ["I", "am"],
            ["I", "am", "learning"],
            ["I", "am", "learning", "Python"],
            ["I", "am", "learning", "Python", "programming"],
            ["am"],
            ["am", "learning"],
            ["am", "learning", "Python"],
            ["am", "learning", "Python", "programming"],
            ["learning"],
            ["learning", "Python"],
            ["learning", "Python", "programming"],
            ["Python"],
            ["Python", "programming"],
            ["programming"]
        ]

        # When increasing_window is called
        result = list(increasing_window(text, step_size, max_window_size))

        # Then the result matches the expected output
        assert result == expected

    def test_increasing_window_empty_text(self) -> None:
        # Given an empty text
        text = ""
        max_window_size = 3
        step_size = 1
        expected = []

        # When increasing_window is called
        result = list(increasing_window(text, step_size, max_window_size))

        # Then the result is empty
        assert result == expected

    def test_increasing_window_single_token(self) -> None:
        # Given a single-token text
        text = "I"
        max_window_size = 1
        step_size = 1
        expected = [["I"]]

        # When increasing_window is called
        result = list(increasing_window(text, step_size, max_window_size))

        # Then the result matches the expected output
        assert result == expected
