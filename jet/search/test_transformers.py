import unittest
from transformers import unescape, decode_encoded_characters


class TestTransformers(unittest.TestCase):

    def test_unescape(self):
        # Test for unescaping HTML entities
        text = "This is a test with &nbsp; and &amp;"
        expected = "This is a test with   and &"
        self.assertEqual(unescape(text), expected)

        # Test for text without HTML entities
        text = "No special characters"
        expected = "No special characters"
        self.assertEqual(unescape(text), expected)

    def test_decode_encoded_characters(self):
        # Test for decoding encoded characters
        text = "This is a line with &nbsp; and curly quotes “ and ”"
        expected = "This is a line with   and curly quotes \" and \""
        self.assertEqual(decode_encoded_characters(text), expected)

        # Test for multiple lines of text
        text = "First line with &amp; and curly quotes “ and ”\nSecond line"
        expected = "First line with & and curly quotes \" and \"\nSecond line"
        self.assertEqual(decode_encoded_characters(text), expected)

        # Test for text without any encoded characters
        text = "No special characters here"
        expected = "No special characters here"
        self.assertEqual(decode_encoded_characters(text), expected)

        # Test for text with apostrophes and quotes
        text = "Curly apostrophe ’ and quote “ here"
        expected = "Curly apostrophe ' and quote \" here"
        self.assertEqual(decode_encoded_characters(text), expected)


if __name__ == '__main__':
    unittest.main()
