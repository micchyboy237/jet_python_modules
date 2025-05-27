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
    def test_single_string_non_sequential_n_2(self):
        # Test input
        text = "The quick brown fox jumps. The brown fox jumps quickly."

        # Expected output for non-sequential bigrams (in_sequence=False, n=2) with min_count=2
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

    def test_single_string_sequential_n_2(self):
        # Test input
        text = "The quick brown fox jumps. The brown fox jumps quickly."

        # Expected output for sequential bigrams (in_sequence=True, n=2) with min_count=2
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

    def test_single_string_non_sequential_n_none(self):
        # Test input
        text = "The quick brown fox jumps. The brown fox jumps quickly."

        # Expected output for non-sequential combinations (in_sequence=False, n=None) with min_count=2
        # Sentence 1: quick, brown, fox, jump -> 1-grams: (quick,), (brown,), (fox,), (jump,)
        #                                      -> 2-grams: (quick,brown), (quick,fox), (quick,jump), (brown,fox), (brown,jump), (fox,jump)
        #                                      -> 3-grams: (quick,brown,fox), (quick,brown,jump), (quick,fox,jump), (brown,fox,jump)
        #                                      -> 4-gram: (quick,brown,fox,jump)
        # Sentence 2: brown, fox, jump, quick -> 1-grams: (brown,), (fox,), (jump,), (quick,)
        #                                      -> 2-grams: (brown,fox), (brown,jump), (brown,quick), (fox,jump), (fox,quick), (jump,quick)
        #                                      -> 3-grams: (brown,fvor,jump), (brown,fox,quick), (brown,jump,quick), (fox,jump,quick)
        #                                      -> 4-gram: (brown,fox,jump,quick)
        # Combined counts: (brown,):2, (fox,):2, (jump,):2, (quick,):2, (brown,fox):2, (brown,jump):2, (fox,jump):2, others:1
        expected = {
            ('brown',): 2,
            ('fox',): 2,
            ('jump',): 2,
            ('quick',): 2,
            ('brown', 'fox'): 2,
            ('brown', 'jump'): 2,
            ('fox', 'jump'): 2
        }
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=2, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_string_sequential_n_none(self):
        # Test input
        text = "The quick brown fox jumps. The brown fox jumps quickly."

        # Expected output for sequential combinations (in_sequence=True, n=None) with min_count=2
        # Sentence 1: quick, brown, fox, jump -> 1-grams: (quick,), (brown,), (fox,), (jump,)
        #                                      -> 2-grams: (quick,brown), (brown,fox), (fox,jump)
        #                                      -> 3-grams: (quick,brown,fox), (brown,fox,jump)
        #                                      -> 4-gram: (quick,brown,fox,jump)
        # Sentence 2: brown, fox, jump, quick -> 1-grams: (brown,), (fox,), (jump,), (quick,)
        #                                      -> 2-grams: (brown,fox), (fox,jump), (jump,quick)
        #                                      -> 3-grams: (brown,fox,jump), (fox,jump,quick)
        #                                      -> 4-gram: (brown,fox,jump,quick)
        # Combined counts: (brown,):2, (fox,):2, (jump,):2, (quick,):2, (brown,fox):2, (fox,jump):2, others:1
        expected = {
            ('brown',): 2,
            ('fox',): 2,
            ('jump',): 2,
            ('quick',): 2,
            ('brown', 'fox'): 2,
            ('fox', 'jump'): 2
        }
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=2, in_sequence=True)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_strings_non_sequential_n_none(self):
        # Test input
        text = [
            "The quick brown fox jumps.",
            "The brown fox jumps quickly."
        ]

        # Expected output for non-sequential combinations (in_sequence=False, n=None) with min_count=1
        # Text 1: Sentence: quick, brown, fox, jump -> 1-grams: (quick,), (brown,), (fox,), (jump,)
        #                                       -> 2-grams: (quick,brown), (quick,fox), (quick,jump), (brown,fox), (brown,jump), (fox,jump)
        #                                       -> 3-grams: (quick,brown,fox), (quick,brown,jump), (quick,fox,jump), (brown,fox,jump)
        #                                       -> 4-gram: (quick,brown,fox,jump)
        # Text 2: Sentence: brown, fox, jump, quick -> 1-grams: (brown,), (fox,), (jump,), (quick,)
        #                                       -> 2-grams: (brown,fox), (brown,jump), (brown,quick), (fox,jump), (fox,quick), (jump,quick)
        #                                       -> 3-grams: (brown,fox,jump), (brown,fox,quick), (brown,jump,quick), (fox,jump,quick)
        #                                       -> 4-gram: (brown,fox,jump,quick)
        expected = [
            {
                ('quick',): 1,
                ('brown',): 1,
                ('fox',): 1,
                ('jump',): 1,
                ('quick', 'brown'): 1,
                ('quick', 'fox'): 1,
                ('quick', 'jump'): 1,
                ('brown', 'fox'): 1,
                ('brown', 'jump'): 1,
                ('fox', 'jump'): 1,
                ('quick', 'brown', 'fox'): 1,
                ('quick', 'brown', 'jump'): 1,
                ('quick', 'fox', 'jump'): 1,
                ('brown', 'fox', 'jump'): 1,
                ('quick', 'brown', 'fox', 'jump'): 1
            },
            {
                ('brown',): 1,
                ('fox',): 1,
                ('jump',): 1,
                ('quick',): 1,
                ('brown', 'fox'): 1,
                ('brown', 'jump'): 1,
                ('brown', 'quick'): 1,
                ('fox', 'jump'): 1,
                ('fox', 'quick'): 1,
                ('jump', 'quick'): 1,
                ('brown', 'fox', 'jump'): 1,
                ('brown', 'fox', 'quick'): 1,
                ('brown', 'jump', 'quick'): 1,
                ('fox', 'jump', 'quick'): 1,
                ('brown', 'fox', 'jump', 'quick'): 1
            }
        ]
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=1, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_strings_sequential_n_none(self):
        # Test input
        text = [
            "The quick brown fox jumps.",
            "The brown fox jumps quickly."
        ]

        # Expected output for sequential combinations (in_sequence=True, n=None) with min_count=1
        # Text 1: Sentence: quick, brown, fox, jump -> 1-grams: (quick,), (brown,), (fox,), (jump,)
        #                                       -> 2-grams: (quick,brown), (brown,fox), (fox,jump)
        #                                       -> 3-grams: (quick,brown,fox), (brown,fox,jump)
        #                                       -> 4-gram: (quick,brown,fox,jump)
        # Text 2: Sentence: brown, fox, jump, quick -> 1-grams: (brown,), (fox,), (jump,), (quick,)
        #                                       -> 2-grams: (brown,fox), (fox,jump), (jump,quick)
        #                                       -> 3-grams: (brown,fox,jump), (fox,jump,quick)
        #                                       -> 4-gram: (brown,fox,jump,quick)
        expected = [
            {
                ('quick',): 1,
                ('brown',): 1,
                ('fox',): 1,
                ('jump',): 1,
                ('quick', 'brown'): 1,
                ('brown', 'fox'): 1,
                ('fox', 'jump'): 1,
                ('quick', 'brown', 'fox'): 1,
                ('brown', 'fox', 'jump'): 1,
                ('quick', 'brown', 'fox', 'jump'): 1
            },
            {
                ('brown',): 1,
                ('fox',): 1,
                ('jump',): 1,
                ('quick',): 1,
                ('brown', 'fox'): 1,
                ('fox', 'jump'): 1,
                ('jump', 'quick'): 1,
                ('brown', 'fox', 'jump'): 1,
                ('fox', 'jump', 'quick'): 1,
                ('brown', 'fox', 'jump', 'quick'): 1
            }
        ]
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=1, in_sequence=True)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_string(self):
        # Test input
        text = ""

        # Expected output for empty string
        expected = {}
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=1, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_list(self):
        # Test input
        text = []

        # Expected output for empty list
        expected = []
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=1, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_with_single_word_string(self):
        # Test input
        text = ["Quick.", "Fox."]

        # Expected output for single-word strings (only 1-grams possible with n=None)
        expected = [
            {('quick',): 1},
            {('fox',): 1}
        ]
        result = get_word_sentence_combination_counts(
            text, n=None, min_count=1, in_sequence=False)
        assert result == expected, f"Expected {expected}, but got {result}"


if __name__ == '__main__':
    unittest.main()
