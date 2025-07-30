import pytest
from typing import List, Union
from unittest.mock import Mock
from jet.wordnet.text_chunker import chunk_texts, build_chunk, get_overlap_sentences
from jet.models.tokenizer.base import detokenize, get_tokenizer_fn
import nltk.tokenize

# Mock dependencies


def mock_tokenize(text: str) -> List[str]:
    return text.split()  # Simple word-based tokenizer for testing


def mock_detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


def mock_get_words(text: str) -> List[str]:
    return text.split()  # Match placeholder get_words in chunk_texts


@pytest.fixture
def setup_mocks(monkeypatch):
    """Mock tokenizer, detokenizer, and helper functions."""
    monkeypatch.setattr(
        "jet.wordnet.text_chunker.get_tokenizer_fn", lambda model: mock_tokenize)
    monkeypatch.setattr(
        "jet.models.tokenizer.base.detokenize", mock_detokenize)
    monkeypatch.setattr("jet.wordnet.text_chunker.get_words", mock_get_words)
    monkeypatch.setattr("jet.wordnet.text_chunker.is_list_marker",
                        lambda s: s.strip().startswith(("- ", "1. ", "* ")))
    monkeypatch.setattr(
        "jet.wordnet.text_chunker.is_list_sentence", lambda s: len(s.split()) > 2)


@pytest.fixture
def cleanup_nltk(monkeypatch):
    """Mock NLTK sent_tokenize."""
    monkeypatch.setattr("nltk.tokenize.sent_tokenize",
                        lambda x: x.split(". ") if x else [])


class TestChunkTexts:
    def test_word_based_chunking_single_sentence_fits(self, setup_mocks, cleanup_nltk):
        """Test word-based chunking when a single sentence fits within chunk_size."""
        # Given
        input_text = "This is a short sentence."
        chunk_size = 10
        chunk_overlap = 0
        expected = ["This is a short sentence."]

        # When
        result = chunk_texts(input_text, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap, model=None)

        # Then
        assert result == expected, (
            f"Expected {expected}, "
            f"but got {result}"
        )

    def test_word_based_chunking_sentence_exceeds_starts_new_chunk(self, setup_mocks, cleanup_nltk):
        """Test that a sentence exceeding chunk_size starts a new chunk."""
        # Given
        input_text = "This is a short sentence. This is a very long sentence that exceeds the chunk size limit."
        chunk_size = 10
        chunk_overlap = 0
        expected = [
            "This is a short sentence.",
            "This is a very long sentence that exceeds the chunk size limit."
        ]

        # When
        result = chunk_texts(input_text, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap, model=None)

        # Then
        assert result == expected, (
            f"Expected {expected}, "
            f"but got {result}"
        )

    def test_word_based_chunking_with_overlap(self, setup_mocks, cleanup_nltk):
        """Test word-based chunking with overlap respecting sentence boundaries."""
        # Given
        input_text = "First sentence here. Second sentence now. Third sentence last."
        chunk_size = 8  # Adjusted to ensure overlap is tested
        chunk_overlap = 4
        expected = [
            "First sentence here.",
            "Second sentence now.",
            "Third sentence last."
        ]

        # When
        result = chunk_texts(input_text, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap, model=None)

        # Then
        assert result == expected, (
            f"Expected {expected}, "
            f"but got {result}"
        )

    def test_token_based_chunking_single_sentence_fits(self, setup_mocks, cleanup_nltk):
        """Test token-based chunking when a single sentence fits within chunk_size."""
        # Given
        input_text = "This is a short sentence."
        chunk_size = 10
        chunk_overlap = 0
        model = "mock_model"
        expected = ["This is a short sentence."]

        # When
        result = chunk_texts(input_text, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap, model=model)

        # Then
        assert result == expected, (
            f"Expected {expected}, "
            f"but got {result}"
        )

    def test_token_based_chunking_sentence_exceeds_starts_new_chunk(self, setup_mocks, cleanup_nltk):
        """Test that a sentence exceeding chunk_size starts a new chunk in token-based mode."""
        # Given
        input_text = "Short sentence. This is a very long sentence that exceeds the chunk size limit."
        chunk_size = 10
        chunk_overlap = 0
        model = "mock_model"
        expected = [
            "Short sentence.",
            "This is a very long sentence that exceeds the chunk size limit."
        ]

        # When
        result = chunk_texts(input_text, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap, model=model)

        # Then
        assert result == expected, (
            f"Expected {expected}, "
            f"but got {result}"
        )

    def test_list_marker_handling(self, setup_mocks, cleanup_nltk):
        """Test handling of list markers combined with the next sentence."""
        # Given
        input_text = "- Item one is here. Second sentence."
        chunk_size = 10
        chunk_overlap = 0
        model = None
        expected = ["- Item one is here.", "Second sentence."]

        # When
        result = chunk_texts(input_text, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap, model=model)

        # Then
        assert result == expected, (
            f"Expected {expected}, "
            f"but got {result}"
        )

    def test_empty_input(self, setup_mocks, cleanup_nltk):
        """Test handling of empty input text."""
        # Given
        input_text = ""
        chunk_size = 10
        chunk_overlap = 0
        model = None
        expected = []

        # When
        result = chunk_texts(input_text, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap, model=model)

        # Then
        assert result == expected, (
            f"Expected {expected}, "
            f"but got {result}"
        )

    def test_multiple_texts(self, setup_mocks, cleanup_nltk):
        """Test chunking multiple input texts."""
        # Given
        input_texts = [
            "First text sentence.",
            "Second text sentence one. Second text sentence two."
        ]
        chunk_size = 10
        chunk_overlap = 0
        model = None
        expected = [
            "First text sentence.",
            "Second text sentence one.",
            "Second text sentence two."
        ]

        # When
        result = chunk_texts(input_texts, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap, model=model)

        # Then
        assert result == expected, (
            f"Expected {expected}, "
            f"but got {result}"
        )

    def test_separator_preservation(self, setup_mocks, cleanup_nltk):
        """Test that separators are preserved between sentences."""
        # Given
        input_text = "First sentence.\n\nSecond sentence."
        chunk_size = 10
        chunk_overlap = 0
        model = None
        expected = ["First sentence.\n\n", "Second sentence."]

        # When
        result = chunk_texts(input_text, chunk_size=chunk_size,
                             chunk_overlap=chunk_overlap, model=model)

        # Then
        assert result == expected, (
            f"Expected {expected}, "
            f"but got {result}"
        )

    def test_build_chunk(self, setup_mocks):
        """Test the build_chunk helper function."""
        # Given
        sentences = ["First sentence", "Second sentence"]
        separators = [". ", "!"]
        expected = "First sentence. Second sentence!"

        # When
        result = build_chunk(sentences, separators)

        # Then
        assert result == expected, (
            f"Expected {expected}, "
            f"but got {result}"
        )

    def test_get_overlap_sentences(self, setup_mocks):
        """Test the get_overlap_sentences helper function."""
        # Given
        sentences = ["First sentence", "Second sentence", "Third sentence"]
        separators = [". ", ". ", "."]
        max_overlap = 3
        size_fn = mock_tokenize
        expected_sentences = ["Third sentence"]
        expected_separators = ["."]
        expected_size = 2

        # When
        result_sentences, result_separators, result_size = get_overlap_sentences(
            sentences, separators, max_overlap, size_fn
        )

        # Then
        assert result_sentences == expected_sentences, (
            f"Expected sentences {expected_sentences}, "
            f"but got {result_sentences}"
        )
        assert result_separators == expected_separators, (
            f"Expected separators {expected_separators}, "
            f"but got {result_separators}"
        )
        assert result_size == expected_size, (
            f"Expected size {expected_size}, "
            f"but got {result_size}"
        )
