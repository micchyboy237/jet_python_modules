import unittest
from nltk.corpus import wordnet
from jet.features.nltk_utils import get_pos_tag, get_word_counts_lemmatized


class TestWordCountLemmatized(unittest.TestCase):
    def test_empty_string(self):
        """Test handling of empty string input."""
        result = get_word_counts_lemmatized("")
        self.assertEqual(result, {}, "Empty string should return empty dict")

    def test_punctuation_removed(self):
        """Test that punctuation is removed from word counts."""
        result = get_word_counts_lemmatized("hello, world!")
        expected = {'hello': 1, 'world': 1}
        self.assertEqual(result, expected, "Punctuation should be removed")

    def test_case_insensitivity(self):
        """Test that words are case-insensitive."""
        result = get_word_counts_lemmatized("Hello HELLO hello")
        expected = {'hello': 3}
        self.assertEqual(result, expected, "Words should be case-insensitive")

    def test_lemmatization(self):
        """Test that words are properly lemmatized."""
        result = get_word_counts_lemmatized("running runs run")
        expected = {'run': 3}
        self.assertEqual(result, expected,
                         "Words should be lemmatized to base form")

    def test_pos_tag_mapping(self):
        """Test POS tag mapping for lemmatization."""
        test_cases = [
            ('run', [('run', wordnet.VERB)]),  # Verb
            ('happy', [('happy', wordnet.ADJ)]),  # Adjective
            ('cat', [('cat', wordnet.NOUN)]),  # Noun
            ('quickly', [('quickly', wordnet.ADV)]),  # Adverb
            ('cats running', [('cats', wordnet.NOUN),
             ('running', wordnet.VERB)]),  # Sentence
        ]
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = get_pos_tag(input_text)
                self.assertEqual(
                    result, expected, f"POS tags for '{input_text}' should map to {expected}")

    def test_mixed_input(self):
        """Test handling of mixed input with punctuation, case, and lemmatization."""
        result = get_word_counts_lemmatized("Cats cat! Running runs.")
        expected = {'cat': 2, 'run': 2}
        self.assertEqual(
            result, expected, "Mixed input should handle punctuation, case, and lemmatization")


if __name__ == '__main__':
    unittest.main()
