import unittest
from jet.scrapers.utils import clean_punctuations, clean_spaces, clean_non_alphanumeric


class TestCleanSpaces(unittest.TestCase):
    def test_basic_punctuation_spacing(self):
        sample = "Hello  !This  is a test .What ?Yes !Go,there:here;[now] {wait}."
        expected = "Hello! This is a test. What? Yes! Go, there: here;[now] {wait}."
        result = clean_spaces(sample)
        self.assertEqual(result, expected)

    def test_extra_spaces_before_punctuation(self):
        sample = "Hello  ,  world  !  How are you  ?"
        expected = "Hello, world! How are you?"
        result = clean_spaces(sample)
        self.assertEqual(result, expected)

    def test_no_space_after_punctuation(self):
        sample = "Hello,world!How are you?Good."
        expected = "Hello, world! How are you? Good."
        result = clean_spaces(sample)
        self.assertEqual(result, expected)

    def test_mixed_spacing_issues(self):
        sample = "Hello  !This|is a test .What ?Yes !Go,there:here;[now] {wait}|next.Add topic *[No.]"
        expected = "Hello! This|is a test. What? Yes! Go, there: here;[now] {wait}|next. Add topic *[No.]"
        result = clean_spaces(sample)
        self.assertEqual(result, expected)

    def test_multiple_punctuations(self):
        sample = "Wait...what?!This is crazy!!"
        expected = "Wait... what?! This is crazy!!"
        result = clean_spaces(sample)
        self.assertEqual(result, expected)

    def test_edge_cases(self):
        sample = "  !Hello,world:this;is[a]{test}!"
        expected = "! Hello, world: this; is[a]{test}!"
        result = clean_spaces(sample)
        self.assertEqual(result, expected)


class TestCleanPunctuations(unittest.TestCase):
    def test_multiple_exclamations(self):
        sample = "Wow!!! This is amazing!!!"
        expected = "Wow! This is amazing!"
        result = clean_punctuations(sample)
        self.assertEqual(result, expected)

    def test_multiple_questions(self):
        sample = "What??? Why??"
        expected = "What? Why?"
        result = clean_punctuations(sample)
        self.assertEqual(result, expected)

    def test_mixed_punctuations(self):
        sample = "Really..?!?.) Are you sure???"
        expected = "Really) Are you sure?"
        result = clean_punctuations(sample)
        self.assertEqual(result, expected)

    def test_mixed_punctuations_middle(self):
        sample = "Wait... What.!?"
        expected = "Wait. What?"
        result = clean_punctuations(sample)
        self.assertEqual(result, expected)

    def test_single_punctuation(self):
        sample = "Hello! How are you?"
        expected = "Hello! How are you?"
        result = clean_punctuations(sample)
        self.assertEqual(result, expected)

    def test_no_punctuation(self):
        sample = "This is a test"
        expected = "This is a test"
        result = clean_punctuations(sample)
        self.assertEqual(result, expected)

    def test_only_punctuations(self):
        sample = "!!!???!!!"
        expected = "!"
        result = clean_punctuations(sample)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()


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
