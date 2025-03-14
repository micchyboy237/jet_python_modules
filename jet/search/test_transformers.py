import unittest
from jet.search.transformers import unescape, decode_encoded_characters, clean_string


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


class TestDecodeEncodedCharacters(unittest.TestCase):
    def test_html_entities(self):
        self.assertEqual(decode_encoded_characters(
            "Tom &amp; Jerry"), "Tom & Jerry")
        self.assertEqual(decode_encoded_characters(
            "3 &lt; 5 &gt; 2"), "3 < 5 > 2")
        self.assertEqual(decode_encoded_characters(
            "It&#39;s a sunny day"), "It's a sunny day")

    def test_special_characters(self):
        self.assertEqual(decode_encoded_characters(
            "Hello，world。"), "Hello, world.")
        self.assertEqual(decode_encoded_characters(
            "He said, “Hello”"), 'He said, "Hello"')
        self.assertEqual(decode_encoded_characters(
            "Price is １００％"), "Price is 100%")

    def test_punctuation_and_numbers(self):
        self.assertEqual(decode_encoded_characters("１＋１＝２"), "1＋1＝2")
        self.assertEqual(decode_encoded_characters("～Special～"), "~Special~")
        self.assertEqual(decode_encoded_characters("〈Title〉"), "<Title>")


class TestCleanString(unittest.TestCase):
    def test_removes_unmatched_quotes(self):
        self.assertEqual(clean_string('Hello "World'), 'Hello World')
        self.assertEqual(clean_string('Hello World"'), 'Hello World')
        self.assertEqual(clean_string('"Hello World"'), 'Hello World')

    def test_removes_unmatched_parentheses(self):
        self.assertEqual(clean_string("Hello (World"), "Hello World")
        self.assertEqual(clean_string("Hello World)"), "Hello World")
        self.assertEqual(clean_string("(Hello) World"), "(Hello) World")

    def test_removes_unmatched_brackets(self):
        self.assertEqual(clean_string("Text [brackets"), "Text brackets")
        self.assertEqual(clean_string("Text brackets]"), "Text brackets")
        self.assertEqual(clean_string("[Text] brackets"), "[Text] brackets")

    def test_removes_leading_and_trailing_commas(self):
        self.assertEqual(clean_string(",Hello, World,"), "Hello, World")
        self.assertEqual(clean_string(",Test String,"), "Test String")

    def test_handles_multiple_cases(self):
        self.assertEqual(clean_string(' "Hello [World]!" '), 'Hello [World]!')
        self.assertEqual(clean_string("'Test'"), "Test")
        self.assertEqual(clean_string('"A mismatched "quote'),
                         "A mismatched quote")


if __name__ == "__main__":
    unittest.main()
