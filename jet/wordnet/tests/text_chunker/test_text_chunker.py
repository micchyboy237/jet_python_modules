import pytest
from typing import List, Optional
from unittest.mock import Mock
from jet.wordnet.text_chunker import chunk_texts, build_chunk, get_overlap_sentences
from jet.wordnet.sentence import split_sentences, is_list_marker, is_list_sentence
from jet.wordnet.words import get_words


class TestChunkTexts:
    @pytest.fixture
    def mock_dependencies(self, mocker):
        """Set up mocks for external dependencies."""
        mocker.patch('jet.wordnet.text_chunker.split_sentences',
                     return_value=[])
        mocker.patch('jet.wordnet.text_chunker.get_words', return_value=[])
        mocker.patch('jet.wordnet.text_chunker.get_tokenizer_fn',
                     return_value=lambda x: [])
        mocker.patch('jet.wordnet.sentence.is_list_marker', return_value=False)
        mocker.patch('jet.wordnet.sentence.is_list_sentence',
                     return_value=False)
        return mocker

    def test_chunk_texts_empty_input(self, mock_dependencies):
        """Test chunk_texts with empty input string."""
        # Given an empty input string
        input_text = ""
        expected = []

        # When chunk_texts is called
        result = chunk_texts(input_text, chunk_size=128, chunk_overlap=0)

        # Then the result should be an empty list
        assert result == expected, "Expected empty list for empty input"

    def test_chunk_texts_single_sentence_within_chunk_size(self, mocker):
        """Test chunk_texts with a single sentence fitting within chunk size."""
        # Given a single sentence and mocked dependencies
        input_text = "This is a test sentence."
        expected_sentences = ["This is a test sentence."]
        expected_words = ["This", "is", "a", "test", "sentence"]
        expected = ["This is a test sentence."]

        mocker.patch('jet.wordnet.text_chunker.split_sentences',
                     return_value=expected_sentences)
        mocker.patch('jet.wordnet.text_chunker.get_words',
                     return_value=expected_words)

        # When chunk_texts is called with chunk_size larger than sentence
        result = chunk_texts(input_text, chunk_size=10, chunk_overlap=0)

        # Then the result should contain the single sentence
        assert result == expected, f"Expected {expected}, got {result}"

    def test_chunk_texts_multiple_sentences_exceeding_chunk_size(self, mocker):
        """Test chunk_texts with multiple sentences exceeding chunk size."""
        # Given multiple sentences that exceed chunk size
        input_text = "First sentence. Second sentence. Third sentence."
        expected_sentences = ["First sentence.",
                              "Second sentence.", "Third sentence."]
        expected_words = {
            "First sentence.": ["First", "sentence"],
            "Second sentence.": ["Second", "sentence"],
            "Third sentence.": ["Third", "sentence"]
        }
        expected = ["First sentence.", "Second sentence.", "Third sentence."]

        def mock_get_words(sentence):
            return expected_words.get(sentence, [])

        mocker.patch('jet.wordnet.text_chunker.split_sentences',
                     return_value=expected_sentences)
        mocker.patch('jet.wordnet.text_chunker.get_words',
                     side_effect=mock_get_words)

        # When chunk_texts is called with small chunk_size
        result = chunk_texts(input_text, chunk_size=2, chunk_overlap=0)

        # Then the result should split sentences into separate chunks
        assert result == expected, f"Expected {expected}, got {result}"

    def test_chunk_texts_with_overlap(self, mocker):
        """Test chunk_texts with overlap, ensuring sentence boundaries are respected."""
        # Given sentences with overlap configuration
        input_text = "First sentence. Second sentence. Third sentence."
        expected_sentences = ["First sentence.",
                              "Second sentence.", "Third sentence."]
        expected_words = {
            "First sentence.": ["First", "sentence"],
            "Second sentence.": ["Second", "sentence"],
            "Third sentence.": ["Third", "sentence"]
        }
        expected = [
            "First sentence. Second sentence.",
            "Second sentence. Third sentence."
        ]

        def mock_get_words(sentence):
            return expected_words.get(sentence, [])

        mocker.patch('jet.wordnet.text_chunker.split_sentences',
                     return_value=expected_sentences)
        mocker.patch('jet.wordnet.text_chunker.get_words',
                     side_effect=mock_get_words)

        # When chunk_texts is called with chunk_size and overlap
        result = chunk_texts(input_text, chunk_size=4, chunk_overlap=2)

        # Then the result should include overlap with sentence boundaries
        assert result == expected, f"Expected {expected}, got {result}"

    def test_chunk_texts_with_list_items(self, mocker):
        """Test chunk_texts with list items, ensuring they are combined correctly."""
        # Given text with list items
        input_text = "1. First item. Second sentence."
        expected_sentences = ["1. First item.", "Second sentence."]
        expected_words = {
            "1. First item.": ["1", "First", "item"],
            "Second sentence.": ["Second", "sentence"]
        }
        expected = ["1. First item.", "Second sentence."]

        def mock_get_words(sentence):
            return expected_words.get(sentence, [])

        mocker.patch('jet.wordnet.text_chunker.split_sentences',
                     return_value=expected_sentences)
        mocker.patch('jet.wordnet.text_chunker.get_words',
                     side_effect=mock_get_words)
        mocker.patch('jet.wordnet.sentence.is_list_marker',
                     side_effect=lambda x: x == "1.")
        mocker.patch('jet.wordnet.sentence.is_list_sentence',
                     side_effect=lambda x: x == "1. First item.")

        # When chunk_texts is called
        result = chunk_texts(input_text, chunk_size=3, chunk_overlap=0)

        # Then the result should handle list items correctly
        assert result == expected, f"Expected {expected}, got {result}"

    def test_chunk_texts_with_model_tokenizer(self, mocker):
        """Test chunk_texts using a model tokenizer."""
        # Given text and a model for token-based chunking
        input_text = "This is a test sentence."
        expected_sentences = ["This is a test sentence."]
        expected_tokens = ["token1", "token2", "token3"]
        expected = ["This is a test sentence."]

        mock_tokenizer = Mock(return_value=expected_tokens)
        mocker.patch('jet.wordnet.text_chunker.split_sentences',
                     return_value=expected_sentences)
        mocker.patch('jet.wordnet.text_chunker.get_tokenizer_fn',
                     return_value=mock_tokenizer)

        # When chunk_texts is called with a model
        result = chunk_texts(input_text, chunk_size=5,
                             chunk_overlap=0, model="test-model")

        # Then the result should use the tokenizer and fit within chunk size
        assert result == expected, f"Expected {expected}, got {result}"
        mock_tokenizer.assert_called()

    def test_chunk_texts_list_of_strings(self, mocker):
        """Test chunk_texts with a list of strings as input."""
        # Given a list of strings
        input_texts = ["First sentence.", "Second sentence."]
        expected_sentences = [["First sentence."], ["Second sentence."]]
        expected_words = {
            "First sentence.": ["First", "sentence"],
            "Second sentence.": ["Second", "sentence"]
        }
        expected = ["First sentence.", "Second sentence."]

        def mock_get_words(sentence):
            return expected_words.get(sentence, [])

        mocker.patch('jet.wordnet.text_chunker.split_sentences',
                     side_effect=expected_sentences)
        mocker.patch('jet.wordnet.text_chunker.get_words',
                     side_effect=mock_get_words)

        # When chunk_texts is called with a list of strings
        result = chunk_texts(input_texts, chunk_size=5, chunk_overlap=0)

        # Then the result should process each string independently
        assert result == expected, f"Expected {expected}, got {result}"
