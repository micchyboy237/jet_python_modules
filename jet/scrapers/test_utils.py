import unittest
# Assuming function is in clean_text.py
from jet.scrapers.utils import clean_spaces, clean_non_alphanumeric


class TestCleanSpaces(unittest.TestCase):
    def test_basic_spacing(self):
        self.assertEqual(clean_spaces("Hello  world"), "Hello world")

    def test_punctuation_spacing(self):
        self.assertEqual(clean_spaces("Hello , world !"), "Hello, world!")

    def test_mixed_spacing(self):
        self.assertEqual(clean_spaces(
            "  This  is   a  test .  "), "This is a test.")

    def test_no_extra_spaces(self):
        self.assertEqual(clean_spaces("NoExtraSpaces"), "NoExtraSpaces")

    def test_multiple_punctuation(self):
        self.assertEqual(clean_spaces(
            "Hello , world ! How  are you ?"), "Hello, world! How are you?")

    def test_exclude_chars(self):
        self.assertEqual(clean_spaces("Hello-world !",
                         exclude_chars=["-"]), "Hello-world!")


# class TestCleanNonAlphanumeric(unittest.TestCase):

#     def test_alphanumeric_string(self):
#         self.assertEqual(clean_non_alphanumeric(
#             "HelloWorld123"), "HelloWorld123")

#     def test_string_with_spaces(self):
#         self.assertEqual(clean_non_alphanumeric(
#             "Hello World 123"), "HelloWorld123")

#     def test_string_with_special_characters(self):
#         self.assertEqual(clean_non_alphanumeric(
#             "Hello@#$%^&*()_+World123!"), "HelloWorld123")

#     def test_only_special_characters(self):
#         self.assertEqual(clean_non_alphanumeric("!@#$%^&*()"), "")

#     def test_empty_string(self):
#         self.assertEqual(clean_non_alphanumeric(""), "")

#     def test_numbers_only(self):
#         self.assertEqual(clean_non_alphanumeric("1234567890"), "1234567890")

#     def test_letters_only(self):
#         self.assertEqual(clean_non_alphanumeric("abcdefXYZ"), "abcdefXYZ")

#     def test_mixed_case_letters_and_numbers(self):
#         self.assertEqual(clean_non_alphanumeric("AbC123xYz"), "AbC123xYz")

#     def test_include_chars_spaces(self):
#         self.assertEqual(clean_non_alphanumeric(
#             "Hello, World! 123", include_chars=[" "]), "Hello World123")

#     def test_include_chars_commas(self):
#         self.assertEqual(clean_non_alphanumeric(
#             "Hello, World! 123", include_chars=[","]), "Hello,World123")

#     def test_include_chars_spaces_and_commas(self):
#         self.assertEqual(clean_non_alphanumeric(
#             "Hello, World! 123", include_chars=[",", " "]), "Hello, World 123")

#     def test_include_chars_currency_symbols(self):
#         self.assertEqual(clean_non_alphanumeric(
#             "Price: $100.99", include_chars=["$", "."]), "Price$100.99")


if __name__ == '__main__':
    unittest.main()
