import pytest
from jet.features.nltk_utils import get_word_counts_lemmatized, get_word_sentence_combination_counts
import time


def test_get_word_counts_lemmatized_single_string():
    input_text = "The cats are running and the cat runs."
    expected = {"cat": 2, "run": 2}
    result = get_word_counts_lemmatized(input_text)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_counts_lemmatized_list_of_strings():
    input_texts = ["The cat runs.", "Cats are running."]
    expected = [{"cat": 1, "run": 1}, {"cat": 1, "run": 1}]
    result = get_word_counts_lemmatized(input_texts)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_counts_lemmatized_empty_string():
    input_text = ""
    expected = {}
    result = get_word_counts_lemmatized(input_text)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_counts_lemmatized_invalid_input():
    input_text = 123
    with pytest.raises(ValueError) as exc_info:
        get_word_counts_lemmatized(input_text)
    expected = "Input must be a string or a list of strings"
    result = str(exc_info.value)
    assert result == expected, f"Expected error message {expected}, but got {result}"


def test_get_word_sentence_combination_counts_single_string():
    input_text = "The cat runs. The dog runs."
    expected = {"cat": 1, "run": 2, "dog": 1}
    result = get_word_sentence_combination_counts(input_text, n=1, min_count=1)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_sentence_combination_counts_list_of_strings():
    input_texts = ["The cat runs.", "The dog runs."]
    expected = [{"cat": 1, "run": 1}, {"dog": 1, "run": 1}]
    result = get_word_sentence_combination_counts(
        input_texts, n=1, min_count=1)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_sentence_combination_counts_in_sequence():
    input_text = "The cat runs fast."
    expected = {"cat,run": 1, "run,fast": 1}
    result = get_word_sentence_combination_counts(
        input_text, n=2, in_sequence=True)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_sentence_combination_counts_min_count():
    input_text = "The cat runs. The cat sleeps."
    expected = {"cat": 2}
    result = get_word_sentence_combination_counts(input_text, n=1, min_count=2)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_sentence_combination_counts_no_n():
    input_text = "The cat runs."
    expected = {"cat": 1, "run": 1, "cat,run": 1}
    result = get_word_sentence_combination_counts(input_text, min_count=1)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_sentence_combination_counts_empty_string():
    input_text = ""
    expected = {}
    result = get_word_sentence_combination_counts(input_text)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_sentence_combination_counts_long_text():
    sentence = "The cat runs and jumps."
    input_text = " ".join([sentence] * 1000)
    expected = {"cat": 1000, "run": 1000, "jump": 1000}
    start_time = time.time()
    result = get_word_sentence_combination_counts(
        input_text, n=1, min_count=2, show_progress=True)
    elapsed_time = time.time() - start_time
    assert result == expected, f"Expected {expected}, but got {result}"
    assert elapsed_time < 5, f"Processing took too long: {elapsed_time} seconds"


def test_get_word_sentence_combination_counts_large_n_in_sequence():
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


def test_get_word_sentence_combination_counts_large_n_non_sequence():
    input_text = "The cat runs jumps sleeps."
    expected = {"cat,run,jump,sleep": 1}
    result = get_word_sentence_combination_counts(
        input_text, n=4, in_sequence=False, min_count=1)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_sentence_combination_counts_large_list():
    input_texts = ["The cat runs and jumps."] * 100
    expected = [{"cat": 1, "run": 1, "jump": 1}] * 100
    start_time = time.time()
    result = get_word_sentence_combination_counts(
        input_texts, n=1, min_count=1, show_progress=False)
    elapsed_time = time.time() - start_time
    assert result == expected, f"Expected {expected}, but got {result}"
    assert elapsed_time < 2, f"Processing took too long: {elapsed_time} seconds"


def test_get_word_sentence_combination_counts_max_n_cap():
    input_text = " ".join(["word" + str(i) for i in range(15)])
    expected = {}
    result = get_word_sentence_combination_counts(
        input_text, n=11, min_count=1)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_sentence_combination_counts_long_list_large_texts_n_none():
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
