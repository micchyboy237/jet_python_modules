import unittest
from jet.features.nltk_utils import get_word_counts_lemmatized, get_word_sentence_combination_counts


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


class TestWordSentenceCombinationCounts:
    def test_non_sequential_min_count_2(self):
        # Test input
        text = "The quick brown fox jumps. The brown fox jumps quickly."

        # Expected output for non-sequential bigrams (in_sequence=False) with min_count=2
        # Sentence 1: quick, brown, fox, jump -> pairs: (quick,brown), (quick,fox), (quick,jump), (brown,fox), (brown,jump), (fox,jump)
        # Sentence 2: brown, fox, jump, quick -> pairs: (brown,fox), (brown,jump), (brown,quick), (fox,jump), (fox,quick), (jump,quick)
        # Combined counts: (brown,fox):2, (brown,jump):2, (fox,jump):2, others:1
        expected = {
            ('brown', 'fox'): 2,
            ('brown', 'jump'): 2,
            ('fox', 'jump'): 2
        }
        result = get_word_sentence_combination_counts(
            text, n=2, min_count=2, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_sequential_min_count_2(self):
        # Test input
        text = "The quick brown fox jumps. The brown fox jumps quickly."

        # Expected output for sequential bigrams (in_sequence=True) with min_count=2
        # Sentence 1: quick, brown, fox, jump -> sequential pairs: (quick,brown), (brown,fox), (fox,jump)
        # Sentence 2: brown, fox, jump, quick -> sequential pairs: (brown,fox), (fox,jump), (jump,quick)
        # Combined counts: (brown,fox):2, (fox,jump):2, others:1
        expected = {
            ('brown', 'fox'): 2,
            ('fox', 'jump'): 2
        }
        result = get_word_sentence_combination_counts(
            text, n=2, min_count=2, in_sequence=True)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_input(self):
        # Test input
        text = ""

        # Expected output for empty text
        expected = {}
        result = get_word_sentence_combination_counts(
            text, n=2, min_count=1, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_word_sentence(self):
        # Test input
        text = "Quick."

        # Expected output for single-word sentence (no bigrams possible)
        expected = {}
        result = get_word_sentence_combination_counts(
            text, n=2, min_count=1, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"


if __name__ == '__main__':
    unittest.main()
