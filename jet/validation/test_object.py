import unittest

from jet.logger import logger
from jet.validation.object import is_iterable_but_not_primitive


class TestIsIterableButNotPrimitive(unittest.TestCase):

    def log_category(self, category: str, test_list: list):
        """Logs the category being tested with the number of tests"""
        logger.info(f"\nRunning {len(test_list)} {category} tests...")

    def test_true_results(self):
        """Test cases that should return True"""
        true_tests = [
            ([1, 2, 3], True),       # List
            ((1, 2, 3), True)        # Tuple
        ]
        self.log_category("True results", true_tests)

        for obj, expected in true_tests:
            with self.subTest(obj=obj):
                self.assertEqual(is_iterable_but_not_primitive(obj), expected)

    def test_false_results(self):
        """Test cases that should return False"""
        false_tests = [
            (42, False),              # Integer
            (3.14, False),            # Float
            (True, False),            # Boolean
            ("hello", False),         # String
            (b'hello', False),        # Bytes
            ({1, 2, 3}, False),      # Set
            ({'key': 'value'}, False),  # Dictionary
            (None, False),           # None
            (self._custom_iterable(), False)  # Custom iterable
        ]
        self.log_category("False results", false_tests)

        for obj, expected in false_tests:
            with self.subTest(obj=obj):
                self.assertEqual(is_iterable_but_not_primitive(obj), expected)

    def _custom_iterable(self):
        """Returns a custom iterable object."""
        class MyIterable:
            def __iter__(self):
                return iter([1, 2, 3])

        return MyIterable()


if __name__ == "__main__":
    unittest.main()
