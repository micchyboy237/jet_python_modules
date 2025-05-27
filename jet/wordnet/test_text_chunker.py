import pytest
from jet.wordnet.text_chunker import chunk_sentences_with_indices, chunk_texts, chunk_sentences, chunk_texts_with_indices


def test_chunk_texts_no_overlap():
    input_text = "This is a test text with several words to be chunked into smaller pieces"
    expected = [
        "This is a test text with several words",
        "to be chunked into smaller pieces"
    ]
    result = chunk_texts(input_text, chunk_size=8, chunk_overlap=0)
    assert result == expected


def test_chunk_texts_with_overlap():
    input_text = "This is a test text with several words to be chunked"
    expected = [
        "This is a test text with several words",
        "with several words to be chunked"
    ]
    result = chunk_texts(input_text, chunk_size=8, chunk_overlap=3)
    assert result == expected


def test_chunk_sentences_no_overlap():
    input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    expected = [
        "This is sentence one. This is sentence two.",
        "This is sentence three. This is sentence four."
    ]
    result = chunk_sentences(input_text, chunk_size=2, sentence_overlap=0)
    assert result == expected


def test_chunk_sentences_with_overlap():
    input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    expected = [
        "This is sentence one. This is sentence two.",
        "This is sentence two. This is sentence three.",
        "This is sentence three. This is sentence four."
    ]
    result = chunk_sentences(input_text, chunk_size=2, sentence_overlap=1)
    assert result == expected


def test_chunk_texts_no_overlap_with_indices():
    input_text = "This is a test text with several words to be chunked into smaller pieces"
    expected = [
        "This is a test text with several words",
        "to be chunked into smaller pieces"
    ]
    expected_indices = [0, 0]
    result, result_indices = chunk_texts_with_indices(
        input_text, chunk_size=8, chunk_overlap=0)
    assert result == expected
    assert result_indices == expected_indices


def test_chunk_texts_with_overlap_with_indices():
    input_text = "This is a test text with several words to be chunked"
    expected = [
        "This is a test text with several words",
        "with several words to be chunked"
    ]
    expected_indices = [0, 0]
    result, result_indices = chunk_texts_with_indices(
        input_text, chunk_size=8, chunk_overlap=3)
    assert result == expected
    assert result_indices == expected_indices


def test_chunk_sentences_no_overlap_with_indices():
    input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    expected = [
        "This is sentence one. This is sentence two.",
        "This is sentence three. This is sentence four."
    ]
    expected_indices = [0, 0]
    result, result_indices = chunk_sentences_with_indices(
        input_text, chunk_size=2, sentence_overlap=0)
    assert result == expected
    assert result_indices == expected_indices


def test_chunk_sentences_with_overlap_with_indices():
    input_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    expected = [
        "This is sentence one. This is sentence two.",
        "This is sentence two. This is sentence three.",
        "This is sentence three. This is sentence four."
    ]
    expected_indices = [0, 0, 0]
    result, result_indices = chunk_sentences_with_indices(
        input_text, chunk_size=2, sentence_overlap=1)
    assert result == expected
    assert result_indices == expected_indices
