import unittest
import re
from jet.wordnet.sentence import (
    is_ordered_list_marker,
    is_ordered_list_sentence,
    is_list_marker,
    is_list_sentence,
    is_unordered_list_marker,
)


class TestIsOrderedListMarker(unittest.TestCase):
    def test_numeric_marker(self):
        text = "1. "
        expected = True
        result = is_ordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_letter_marker(self):
        text = "a) "
        expected = True
        result = is_ordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_roman_numeral_marker(self):
        text = "IV. "
        expected = True
        result = is_ordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_unordered_marker(self):
        text = "- "
        expected = False
        result = is_ordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_invalid_marker(self):
        text = "# "
        expected = False
        result = is_ordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_numeric_sentence(self):
        text = "1. Item"
        expected = False
        result = is_ordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")


class TestIsOrderedListSentence(unittest.TestCase):
    def test_numeric_sentence(self):
        text = "1. Hello"
        expected = True
        result = is_ordered_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_letter_sentence(self):
        text = "a) World"
        expected = True
        result = is_ordered_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_roman_numeral_sentence(self):
        text = "IV. Test"
        expected = True
        result = is_ordered_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_unordered_sentence(self):
        text = "- Item"
        expected = False
        result = is_ordered_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_invalid_marker(self):
        text = "# Item"
        expected = False
        result = is_ordered_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")


class TestIsListMarker(unittest.TestCase):
    def test_numeric_marker(self):
        text = "1. "
        expected = True
        result = is_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_letter_marker(self):
        text = "a) "
        expected = True
        result = is_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_unordered_marker(self):
        text = "* "
        expected = True
        result = is_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_invalid_marker(self):
        text = "# "
        expected = False
        result = is_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_numeric_sentence(self):
        text = "1. Item"
        expected = False
        result = is_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_empty_string(self):
        text = ""
        expected = False
        result = is_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")


class TestIsListSentence(unittest.TestCase):
    def test_numeric_sentence(self):
        text = "1. Hello"
        expected = True
        result = is_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_letter_sentence(self):
        text = "a) World"
        expected = True
        result = is_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_unordered_sentence(self):
        text = "- Item"
        expected = True
        result = is_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_invalid_marker(self):
        text = "# Item"
        expected = False
        result = is_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_numeric_marker(self):
        text = "1. "
        expected = False
        result = is_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_no_marker(self):
        text = "Hello"
        expected = False
        result = is_list_sentence(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")


class TestIsUnorderedListMarker(unittest.TestCase):
    def test_dash_marker(self):
        text = "- "
        expected = True
        result = is_unordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_asterisk_marker(self):
        text = "* "
        expected = True
        result = is_unordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_plus_marker(self):
        text = "+ "
        expected = True
        result = is_unordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_ordered_numeric_marker(self):
        text = "1. "
        expected = False
        result = is_unordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_ordered_letter_marker(self):
        text = "a) "
        expected = False
        result = is_unordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_roman_numeral_marker(self):
        text = "IV. "
        expected = False
        result = is_unordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_invalid_marker(self):
        text = "# "
        expected = False
        result = is_unordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")

    def test_empty_string(self):
        text = ""
        expected = False
        result = is_unordered_list_marker(text)
        self.assertEqual(result, expected,
                         f"For {text}, expected {expected}, got {result}")


if __name__ == '__main__':
    unittest.main()
