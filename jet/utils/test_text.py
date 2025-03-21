import unittest

from jet.utils.text import fix_and_unidecode, find_word_indexes, find_sentence_indexes, extract_word_sentences, extract_substrings


class TestFixAndUnidecode(unittest.TestCase):
    def test_normal_unicode(self):
        sample = "Chromium\\n\\u2554"
        expected = "Chromium\\n+"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)

    def test_normal_unicode_2(self):
        sample = "Café"
        expected = "Cafe"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)

    def test_multiple_escapes(self):
        sample = "Hello \\u2603 World! \\nNew Line"
        expected = "Hello  World! \\nNew Line"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)

    def test_no_escapes(self):
        sample = "Simple text without escapes"
        expected = "Simple text without escapes"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)

    def test_mixed_escaped_and_plain(self):
        sample = "Plain text \\n with \\u03A9 Omega"
        expected = "Plain text \\n with O Omega"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)

    def test_double_escaped(self):
        sample = "Double escape \\\\u2554"
        expected = "Double escape \\+"
        result = fix_and_unidecode(sample)
        self.assertEqual(result, expected)


class TestFindWordIndexes(unittest.TestCase):

    def test_find_word_indexes(self):
        text = "The quick brown fox jumps over the lazy fox in the forest."

        # Test case: word appears multiple times
        word = "fox"
        expected = [[16, 18], [40, 42]]
        self.assertEqual(find_word_indexes(text, word), expected)

        # Test case: word appears once
        word = "quick"
        expected = [[4, 8]]
        self.assertEqual(find_word_indexes(text, word), expected)

        # Test case: word does not appear
        word = "cat"
        expected = []
        self.assertEqual(find_word_indexes(text, word), expected)

        # Test case: word is at the start
        word = "The"
        expected = [[0, 2]]
        self.assertEqual(find_word_indexes(text, word), expected)

        # Test case: word is at the end
        word = "forest."
        expected = [[53, 59]]
        self.assertEqual(find_word_indexes(text, word), expected)

        # Test case: empty string
        word = ""
        expected = []
        self.assertEqual(find_word_indexes(text, word), expected)


class TestFindSentenceIndexes(unittest.TestCase):

    def test_find_sentence_indexes(self):
        text = ("The quick brown fox jumps over the lazy dog. "
                "A fox is clever. The forest is quiet at night.")

        # Test case: Word appears in multiple sentences
        word = "fox"
        expected = [[0, 43], [45, 59]]
        self.assertEqual(find_sentence_indexes(text, word), expected)

        # Test case: Word appears in only one sentence
        word = "forest"
        expected = [[61, 89]]
        self.assertEqual(find_sentence_indexes(text, word), expected)

        # Test case: Word does not appear in any sentence
        word = "cat"
        expected = []
        self.assertEqual(find_sentence_indexes(text, word), expected)

        # Test case: Word at the start of a sentence
        text = "Foxes are smart. The dog barks at night."
        word = "Foxes"
        expected = [[0, 15]]
        self.assertEqual(find_sentence_indexes(text, word), expected)

        # Test case: Case-sensitive search (should return empty)
        word = "FOX"
        expected = []
        self.assertEqual(find_sentence_indexes(text, word), expected)

    def test_empty_text(self):
        # Test case: Empty text
        text = ""
        word = "fox"
        expected = []
        self.assertEqual(find_sentence_indexes(text, word), expected)


class TestExtractWordSentences(unittest.TestCase):

    def test_extract_word_sentences(self):
        text = ("The quick brown fox jumps over the lazy dog. "
                "A fox is clever. The forest is quiet at night.")

        # Test case: Word appears in multiple sentences
        word = "fox"
        expected = ["The quick brown fox jumps over the lazy dog.",
                    "A fox is clever."]
        self.assertEqual(extract_word_sentences(text, word), expected)

        # Test case: Word appears in only one sentence
        word = "forest"
        expected = ["The forest is quiet at night."]
        self.assertEqual(extract_word_sentences(text, word), expected)

        # Test case: Word does not appear in any sentence
        word = "cat"
        expected = []
        self.assertEqual(extract_word_sentences(text, word), expected)

        # Test case: Word is at the start of a sentence
        text = "Foxes are smart. The dog barks at night."
        word = "Foxes"
        expected = ["Foxes are smart."]
        self.assertEqual(extract_word_sentences(text, word), expected)

        # Test case: Case-sensitive search (should return empty)
        word = "FOX"
        expected = []
        self.assertEqual(extract_word_sentences(text, word), expected)

    def test_empty_text(self):
        # Test case: Empty text
        text = ""
        word = "fox"
        expected = []
        self.assertEqual(extract_word_sentences(text, word), expected)


class TestExtractSubstrings(unittest.TestCase):

    def test_extract_substrings(self):
        text = "The quick brown fox jumps over the lazy fox in the forest."

        # Test case: extracting multiple words
        indexes = [[16, 18], [40, 42]]
        expected = ["fox", "fox"]
        self.assertEqual(extract_substrings(text, indexes), expected)

        # Test case: extracting a single word
        indexes = [[4, 8]]
        expected = ["quick"]
        self.assertEqual(extract_substrings(text, indexes), expected)

        # Test case: extracting from empty indexes
        indexes = []
        expected = []
        self.assertEqual(extract_substrings(text, indexes), expected)

        # Test case: extracting at start and end
        indexes = [[0, 2], [53, 59]]
        expected = ["The", "forest."]
        self.assertEqual(extract_substrings(text, indexes), expected)

        # Test case: invalid indexes (out of bounds) should not raise an error
        indexes = [[60, 65]]  # Beyond text length
        with self.assertRaises(IndexError):
            extract_substrings(text, indexes)


if __name__ == '__main__':
    unittest.main()
