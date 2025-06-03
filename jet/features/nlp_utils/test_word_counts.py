import pytest
from .word_counts import get_word_counts_lemmatized


class TestGetWordCountsLemmatized:
    def test_single_string_no_pos(self):
        text = "The cats are running and jumping quickly."
        expected = {'cat': 1, 'run': 1, 'jump': 1, 'quickly': 1}
        result = get_word_counts_lemmatized(text)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_string_noun_verb_pos(self):
        text = "The cats are running and jumping quickly."
        expected = {'cat': 1, 'run': 1, 'jump': 1}
        result = get_word_counts_lemmatized(text, pos=['noun', 'verb'])
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_string_min_count(self):
        text = "The cats are running and jumping quickly cats."
        expected = {'cat': 2}
        result = get_word_counts_lemmatized(text, min_count=2)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_of_strings(self):
        texts = [
            "The cats are running and jumping quickly.",
            "Dogs bark loudly and run fast."
        ]
        expected = [
            {'cat': 1, 'run': 1, 'jump': 1, 'quickly': 1},
            {'dog': 1, 'bark': 1, 'run': 1, 'loudly': 1, 'fast': 1}
        ]
        result = get_word_counts_lemmatized(texts)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_with_pos_and_min_count(self):
        texts = [
            "The cats are running and jumping quickly cats.",
            "Dogs bark loudly and run fast cats."
        ]
        expected = [
            {"cat": 2, "run": 1},
            {"run": 1, "cat": 1}
        ]
        result = get_word_counts_lemmatized(
            texts, pos=['noun', 'verb'], min_count=2)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_string(self):
        text = ""
        expected = {}
        result = get_word_counts_lemmatized(text)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_list(self):
        texts = []
        expected = []
        result = get_word_counts_lemmatized(texts)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_invalid_input(self):
        text = 123
        with pytest.raises(TypeError, match="Input must be a string or a list of strings"):
            result = get_word_counts_lemmatized(text)

    def test_single_string_with_percent_threshold(self):
        text = "The cats are running and jumping quickly cats cats."
        # Tokens: ['cat', 'run', 'jump', 'quickly'] → frequencies: cat: 3, others: 1
        # Total valid tokens = 6 → threshold = ceil(6 * 33.33 / 100) = 2
        expected = {'cat': 3}  # This is correct
        result = get_word_counts_lemmatized(text, percent_threshold=33.33)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_string_as_score(self):
        text = "The cats are running and jumping quickly cats."

        result = get_word_counts_lemmatized(text, as_score=True)

        # Build expected from actual logic:
        import math
        raw_scores = {
            'cat': 2 * (1 + math.log(3)),
            'run': 1 * (1 + math.log(3)),
            'jump': 1 * (1 + math.log(4)),
            'quickly': 1 * (1 + math.log(7)),
        }
        max_score = max(raw_scores.values())
        expected = {k: (v / max_score) * 100 for k, v in raw_scores.items()}

        assert result == pytest.approx(
            expected, rel=1e-6), f"Expected {expected}, but got {result}"

    def test_list_of_strings_as_score(self):
        texts = [
            "The cats are running and jumping quickly cats.",
            "Dogs bark loudly and run fast."
        ]

        import math

        def score(count, length): return count * (1 + math.log(length))

        s1_raw = {
            'cat': score(2, 3),
            'run': score(1, 3),
            'jump': score(1, 4),
            'quickly': score(1, 7),
        }
        s1_max = max(s1_raw.values())
        s1 = {k: (v / s1_max) * 100 for k, v in s1_raw.items()}

        s2_raw = {
            'dog': score(1, 3),
            'bark': score(1, 4),
            'run': score(1, 3),
            'loudly': score(1, 6),
            'fast': score(1, 4),
        }
        s2_max = max(s2_raw.values())
        s2 = {k: (v / s2_max) * 100 for k, v in s2_raw.items()}

        expected = [s1, s2]

        result = get_word_counts_lemmatized(texts, as_score=True)
        assert result == pytest.approx(
            expected, rel=1e-6), f"Expected {expected}, but got {result}"
