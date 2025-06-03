import pytest
from .word_sentence_counts import process_single_text, process_wrapper, get_word_sentence_combination_counts
from nltk.tokenize import word_tokenize
from nltk import download

# Ensure NLTK data is downloaded
download('punkt', quiet=True)
download('averaged_perceptron_tagger', quiet=True)
download('wordnet', quiet=True)
download('stopwords', quiet=True)


def test_process_single_text():
    input_text = "The cat runs. The cat jumps."
    expected = {('cat',): 2, ('run',): 1, ('jump',): 1}
    result = process_single_text(
        input_text, n=1, min_count=1, in_sequence=False)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_process_wrapper():
    input_text = "The cat runs. The cat jumps."
    expected = {('cat',): 2, ('run',): 1, ('jump',): 1}
    result = process_wrapper(input_text, n=1, min_count=1, in_sequence=False)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_sentence_combination_counts_single_string():
    input_text = "The cat runs. The cat jumps."
    expected = {('cat',): 2, ('run',): 1, ('jump',): 1}
    result = get_word_sentence_combination_counts(
        input_text, n=1, min_count=1, in_sequence=False, show_progress=False)
    assert result == expected, f"Expected {expected}, but got {result}"


def test_get_word_sentence_combination_counts_list():
    input_texts = ["The cat runs.", "The cat jumps."]
    expected = [
        {('cat',): 1, ('run',): 1},
        {('cat',): 1, ('jump',): 1}
    ]
    result = get_word_sentence_combination_counts(
        input_texts, n=1, min_count=1, in_sequence=False, show_progress=False)
    assert result == expected, f"Expected {expected}, but got {result}"
