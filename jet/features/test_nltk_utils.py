import pytest
import time
from jet.features.nltk_utils import get_word_counts_lemmatized, get_word_sentence_combination_counts


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


class TestGetWordSentenceCombinationCounts:
    def test_single_string_no_n(self):
        input_text = "The cats are running. Cats jump quickly."
        expected = {'cat': 2, 'run': 1, 'jump': 1, 'quickly': 1}
        result = get_word_sentence_combination_counts(
            input_text, n=1, min_count=1)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_string_min_count(self):
        input_text = "The cats are running. Cats jump quickly."
        expected = {'cat': 2}
        result = get_word_sentence_combination_counts(
            input_text, n=1, min_count=2)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_string_in_sequence(self):
        input_text = "The cat runs fast."
        expected = {"cat,run": 1, "run,fast": 1}
        result = get_word_sentence_combination_counts(
            input_text, n=2, in_sequence=True, min_count=1)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_single_string_no_n(self):
        input_text = "The cat runs."
        expected = {"cat": 1, "run": 1, "cat,run": 1}
        result = get_word_sentence_combination_counts(
            input_text, n=None, min_count=1)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_of_strings_no_n(self):
        input_texts = [
            "The cats are running. Cats jump quickly.",
            "Dogs bark loudly. Dogs run fast."
        ]
        expected = [
            {'cat': 2, 'run': 1, 'jump': 1, 'quickly': 1},
            {'dog': 2, 'bark': 1, 'run': 1, 'loudly': 1, 'fast': 1}
        ]
        result = get_word_sentence_combination_counts(
            input_texts, n=1, min_count=1)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_with_min_count(self):
        input_texts = [
            "The cats are running. Cats jump quickly.",
            "Dogs bark loudly. Dogs run fast cats."
        ]
        expected = [
            {'cat': 2, 'run': 1},
            {'dog': 2, 'run': 1, 'cat': 1}
        ]
        result = get_word_sentence_combination_counts(
            input_texts, n=1, min_count=2)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_list_with_bigrams_min_count(self):
        input_texts = [
            "Cats are running fast. Cats run quickly.",
            "Dogs are running fast. Dogs run quickly."
        ]
        expected = [
            {'cat,run': 2, 'run,fast': 1, 'run,quickly': 1},
            {'dog,run': 2, 'run,fast': 1, 'run,quickly': 1}
        ]
        result = get_word_sentence_combination_counts(
            input_texts, n=2, in_sequence=True, min_count=2)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_string(self):
        input_text = ""
        expected = {}
        result = get_word_sentence_combination_counts(input_text)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_list(self):
        input_texts = []
        expected = []
        result = get_word_sentence_combination_counts(input_texts)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_invalid_input(self):
        input_text = 123
        with pytest.raises(ValueError, match="Input must be a string or a list of strings"):
            result = get_word_sentence_combination_counts(input_text)

    def test_long_text(self):
        sentence = "The cat runs and jumps."
        input_text = " ".join([sentence] * 1000)
        expected = {"cat": 1000, "run": 1000, "jump": 1000}
        start_time = time.time()
        result = get_word_sentence_combination_counts(
            input_text, n=1, min_count=2, show_progress=True)
        elapsed_time = time.time() - start_time
        assert result == expected, f"Expected {expected}, but got {result}"
        assert elapsed_time < 5, f"Processing took too long: {elapsed_time} seconds"

    def test_large_n_in_sequence(self):
        input_text = "The cat runs jumps sleeps eats plays walks dreams."
        expected = {
            "cat,run,jump,sleep": 1,
            "run,jump,sleep,eat": 1,
            "jump,sleep,eat,play": 1,
            "sleep,eat,play,walk": 1,
            "eat,play,walk,dream": 1
        }
        result = get_word_sentence_combination_counts(
            input_text, n=4, in_sequence=True, min_count=1)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_large_n_non_sequence(self):
        input_text = "The cat runs jumps sleeps."
        expected = {"cat,run,jump,sleep": 1}
        result = get_word_sentence_combination_counts(
            input_text, n=4, in_sequence=False, min_count=1)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_large_list(self):
        input_texts = ["The cat runs and jumps."] * 100
        expected = [{"cat": 1, "run": 1, "jump": 1}] * 100
        start_time = time.time()
        result = get_word_sentence_combination_counts(
            input_texts, n=1, min_count=1, show_progress=False)
        elapsed_time = time.time() - start_time
        assert result == expected, f"Expected {expected}, but got {result}"
        assert elapsed_time < 2, f"Processing took too long: {elapsed_time} seconds"

    def test_max_n_cap(self):
        input_text = " ".join(["word" + str(i) for i in range(15)])
        expected = {}
        result = get_word_sentence_combination_counts(
            input_text, n=11, min_count=1)
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_long_list_large_texts_n_none(self):
        single_text = " ".join(["The cat runs and jumps daily."] * 50)
        input_texts = [single_text] * 20
        expected = [{
            "cat": 50, "run": 50, "jump": 50, "daily": 50,
            "cat,run": 50, "cat,jump": 50, "cat,daily": 50,
            "run,jump": 50, "run,daily": 50, "jump,daily": 50
        }] * 20
        start_time = time.time()
        result = get_word_sentence_combination_counts(
            input_texts, n=None, min_count=1, in_sequence=False, max_n=2, show_progress=True)
        elapsed_time = time.time() - start_time
        assert result == expected, f"Expected {expected}, but got {result}"
        assert elapsed_time < 5, f"Processing took too long: {elapsed_time} seconds"
