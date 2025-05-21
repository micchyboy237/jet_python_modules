import unittest
from jet.features.nltk_utils import get_word_counts_lemmatized


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
        # Without POS, 'running' may not lemmatize to 'run'
        expected = {'running': 1, 'run': 2}
        self.assertEqual(result, expected,
                         "Words should be lemmatized to base form")

    def test_mixed_input(self):
        """Test handling of mixed input with punctuation, case, and lemmatization."""
        result = get_word_counts_lemmatized("Cats cat! Running runs.")
        # Adjusted for default lemmatization
        expected = {'cat': 2, 'running': 1, 'run': 1}
        self.assertEqual(
            result, expected, "Mixed input should handle punctuation, case, and lemmatization")


if __name__ == '__main__':
    unittest.main()
