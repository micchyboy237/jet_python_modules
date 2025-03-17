import unittest

from jet.wordnet.utils import sliding_window, increasing_window


class TestSlidingWindow(unittest.TestCase):

    def test_window_size_1(self):
        items = [1, 2, 3, 4, 5]
        window_size = 1
        step_size = 1
        expected = [[1], [2], [3], [4], [5]]

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)

    def test_window_size_2(self):
        items = [1, 2, 3, 4, 5]
        window_size = 2
        step_size = 1
        expected = [[1, 2], [2, 3], [3, 4], [4, 5]]

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)

    def test_window_size_equals_step(self):
        items = [1, 2, 3, 4, 5]
        window_size = 2
        step_size = 2
        expected = [[1, 2], [3, 4], [5]]

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)

    def test_window_size_less_than_step(self):
        items = [1, 2, 3, 4, 5]
        window_size = 2
        step_size = 3
        expected = [[1, 2], [4, 5]]

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)

    def test_window_size_equals_list(self):
        items = [1, 2, 3, 4, 5]
        window_size = 5
        step_size = 1
        expected = [[1, 2, 3, 4, 5]]

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)

    def test_window_size_larger_than_list(self):
        items = [1, 2]
        window_size = 10
        step_size = 1
        expected = [[1, 2]]

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)

    def test_step_size_1(self):
        items = [1, 2, 3, 4, 5]
        window_size = 3
        step_size = 1
        expected = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)

    def test_step_size_2(self):
        items = [1, 2, 3, 4, 5]
        window_size = 3
        step_size = 2
        expected = [[1, 2, 3], [3, 4, 5]]

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)

    def test_step_size_3(self):
        items = [1, 2, 3, 4, 5]
        window_size = 3
        step_size = 3
        expected = [[1, 2, 3], [4, 5]]

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)

    def test_step_size_4(self):
        items = [1, 2, 3, 4, 5]
        window_size = 3
        step_size = 4
        expected = [[1, 2, 3], [4, 5]]

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)

    # # Test case where step_size is greater than window_size (should raise error)
    # def test_step_size_greater_than_window_size(self):
    #     items = [1, 2, 3, 4, 5]
    #     window_size = 3
    #     step_size = 4

    #     with self.assertRaises(ValueError):
    #         list(sliding_window(items, window_size, step_size))

    def test_empty_list(self):
        items = []
        window_size = 3
        step_size = 1
        expected = []

        result = list(sliding_window(items, window_size, step_size))
        self.assertEqual(result, expected)


class TestIncreasingWindow(unittest.TestCase):

    def test_increasing_window_basic(self):
        # Test a basic increasing window
        text = "I am learning Python programming"
        max_window_size = 3
        step_size = 1
        expected_result = [
            [
                "I"
            ],
            [
                "I",
                "am"
            ],
            [
                "I",
                "am",
                "learning"
            ],
            [
                "Python"
            ],
            [
                "Python",
                "programming"
            ]
        ]
        result = list(increasing_window(text, step_size, max_window_size))
        self.assertEqual(result, expected_result)

    def test_increasing_window_max_size_none(self):
        # Test where max_window_size is None (default behavior)
        text = "I am learning Python programming"
        max_window_size = None
        step_size = 1
        expected_result = [
            ["I"],
            ["I", "am"],
            ["I", "am", "learning"],
            ["I", "am", "learning", "Python"],
            ["I", "am", "learning", "Python", "programming"]
        ]
        result = list(increasing_window(text, step_size, max_window_size))
        self.assertEqual(result, expected_result)

    def test_increasing_window_empty_text(self):
        # Test for an empty text input
        text = ""
        max_window_size = 3
        step_size = 1
        expected_result = []
        result = list(increasing_window(text, step_size, max_window_size))
        self.assertEqual(result, expected_result)

    def test_increasing_window_single_token(self):
        # Test case where only one token is in the text
        text = "I"
        max_window_size = 1
        step_size = 1
        expected_result = [["I"]]
        result = list(increasing_window(text, step_size, max_window_size))
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
