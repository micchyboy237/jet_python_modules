import pytest
from typing import List
from unittest.mock import Mock
from jet.wordnet.text_chunker import chunk_texts_with_data, ChunkResult
import uuid


class TestChunkTextsWithData:
    def test_empty_input(self):
        """Test chunk_texts_with_data with empty input string."""
        # Given an empty input string
        input_text = ""
        expected: List[ChunkResult] = []

        # When chunk_texts_with_data is called
        result = chunk_texts_with_data(
            input_text, chunk_size=128, chunk_overlap=0)

        # Then the result should be an empty list
        assert result == expected, "Expected empty list for empty input"

    def test_single_sentence_within_chunk_size(self):
        """Test chunk_texts_with_data with a single sentence fitting within chunk size."""
        # Given a single sentence
        input_text = "This is a test sentence."
        expected_sentences = ["This is a test sentence."]
        expected_words = ["This", "is", "a", "test", "sentence"]
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 5,
                "content": "This is a test sentence.",
                "start_idx": 0,
                "end_idx": 24,  # Updated to include final period
                "line_idx": 0
            }
        ]

        # When chunk_texts_with_data is called
        result = chunk_texts_with_data(
            input_text, chunk_size=10, chunk_overlap=0)

        # Then the result should contain a single chunk with correct metadata
        assert len(result) == 1
        assert result[0]["content"] == expected[0]["content"]
        assert result[0]["doc_index"] == expected[0]["doc_index"]
        assert result[0]["chunk_index"] == expected[0]["chunk_index"]
        assert result[0]["num_tokens"] == expected[0]["num_tokens"]
        assert result[0]["start_idx"] == expected[0]["start_idx"]
        assert result[0]["end_idx"] == expected[0]["end_idx"]
        assert result[0]["line_idx"] == expected[0]["line_idx"]

    def test_multiple_sentences_exceeding_chunk_size(self):
        """Test chunk_texts_with_data with multiple sentences exceeding chunk size."""
        # Given multiple sentences
        input_text = "First sentence. Second sentence. Third sentence."
        expected_sentences = ["First sentence.",
                              "Second sentence.", "Third sentence."]
        expected_words = {
            "First sentence.": ["First", "sentence"],
            "Second sentence.": ["Second", "sentence"],
            "Third sentence.": ["Third", "sentence"]
        }
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 2,
                "content": "First sentence.",
                "start_idx": 0,
                "end_idx": 15,  # Corrected to end of "First sentence."
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 0,
                "chunk_index": 1,
                "num_tokens": 2,
                "content": "Second sentence.",
                "start_idx": 16,
                "end_idx": 32,  # Corrected to end of "Second sentence."
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 0,
                "chunk_index": 2,
                "num_tokens": 2,
                "content": "Third sentence.",
                "start_idx": 33,
                "end_idx": 48,  # Corrected to end of "Third sentence."
                "line_idx": 0
            }
        ]

        # When chunk_texts_with_data is called
        result = chunk_texts_with_data(
            input_text, chunk_size=2, chunk_overlap=0)

        # Then the result should split sentences with correct metadata
        assert len(result) == len(expected)
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res["content"] == exp["content"]
            assert res["doc_index"] == exp["doc_index"]
            assert res["chunk_index"] == exp["chunk_index"]
            assert res["num_tokens"] == exp["num_tokens"]
            assert res["start_idx"] == exp["start_idx"]
            assert res["end_idx"] == exp["end_idx"]
            assert res["line_idx"] == exp["line_idx"]

    def test_with_overlap(self):
        """Test chunk_texts_with_data with overlap, respecting sentence boundaries."""
        # Given sentences with overlap
        input_text = "First sentence. Second sentence. Third sentence."
        expected_sentences = ["First sentence.",
                              "Second sentence.", "Third sentence."]
        expected_words = {
            "First sentence.": ["First", "sentence"],
            "Second sentence.": ["Second", "sentence"],
            "Third sentence.": ["Third", "sentence"]
        }
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 4,
                "content": "First sentence. Second sentence.",
                "start_idx": 0,
                "end_idx": 32,  # Corrected to end of "First sentence. Second sentence."
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 0,
                "chunk_index": 1,
                "num_tokens": 4,
                "content": "Second sentence. Third sentence.",
                "start_idx": 16,
                "end_idx": 48,  # Corrected to end of "Second sentence. Third sentence."
                "line_idx": 0
            }
        ]

        # When chunk_texts_with_data is called
        result = chunk_texts_with_data(
            input_text, chunk_size=4, chunk_overlap=2)

        # Then the result should include overlap with correct metadata
        assert len(result) == len(expected)
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res["content"] == exp["content"]
            assert res["doc_index"] == exp["doc_index"]
            assert res["chunk_index"] == exp["chunk_index"]
            assert res["num_tokens"] == exp["num_tokens"]
            assert res["start_idx"] == exp["start_idx"]
            assert res["end_idx"] == exp["end_idx"]
            assert res["line_idx"] == exp["line_idx"]

    def test_with_list_items(self):
        """Test chunk_texts_with_data with list items, ensuring correct combination."""
        # Given text with list items
        input_text = "1. First item. Second sentence."
        expected_sentences = ["1. First item.", "Second sentence."]
        expected_words = {
            "1. First item.": ["1", "First", "item"],
            "Second sentence.": ["Second", "sentence"]
        }
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 3,
                "content": "1. First item.",
                "start_idx": 0,
                "end_idx": 14,  # Corrected to end of "1. First item."
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 0,
                "chunk_index": 1,
                "num_tokens": 2,
                "content": "Second sentence.",
                "start_idx": 15,
                "end_idx": 31,  # Corrected to end of "Second sentence."
                "line_idx": 0
            }
        ]

        # When chunk_texts_with_data is called
        result = chunk_texts_with_data(
            input_text, chunk_size=3, chunk_overlap=0)

        # Then the result should handle list items with correct metadata
        assert len(result) == len(expected)
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res["content"] == exp["content"]
            assert res["doc_index"] == exp["doc_index"]
            assert res["chunk_index"] == exp["chunk_index"]
            assert res["num_tokens"] == exp["num_tokens"]
            assert res["start_idx"] == exp["start_idx"]
            assert res["end_idx"] == exp["end_idx"]
            assert res["line_idx"] == exp["line_idx"]

    def test_with_model_tokenizer(self):
        """Test chunk_texts_with_data using a model tokenizer."""
        # Given text with a model
        input_text = "This is a test sentence."
        expected_sentences = ["This is a test sentence."]
        expected_tokens = ["token1", "token2", "token3"]
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 6,
                "content": "This is a test sentence.",
                "start_idx": 0,
                "end_idx": 24,
                "line_idx": 0
            }
        ]

        # When chunk_texts_with_data is called
        result = chunk_texts_with_data(
            input_text, chunk_size=5, chunk_overlap=0, model="all-MiniLM-L6-v2")

        # Then the result should use tokenizer with correct metadata
        assert len(result) == 1
        assert result[0]["content"] == expected[0]["content"]
        assert result[0]["doc_index"] == expected[0]["doc_index"]
        assert result[0]["chunk_index"] == expected[0]["chunk_index"]
        assert result[0]["num_tokens"] == expected[0]["num_tokens"]
        assert result[0]["start_idx"] == expected[0]["start_idx"]
        assert result[0]["end_idx"] == expected[0]["end_idx"]
        assert result[0]["line_idx"] == expected[0]["line_idx"]

    def test_list_of_strings(self):
        """Test chunk_texts_with_data with a list of strings."""
        # Given a list of strings
        input_texts = ["First sentence.",
                       "Second sentence. Third sentence. Fourth sentence"]
        expected_sentences = [["First sentence."], [
            "Second sentence."], ["Third sentence."], ["Fourth sentence"]]
        expected_words = {
            "First sentence.": ["First", "sentence"],
            "Second sentence.": ["Second", "sentence"],
            "Third sentence.": ["Third", "sentence"],
            "Fourth sentence": ["Fourth", "sentence"]
        }
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 2,
                "content": "First sentence.",
                "start_idx": 0,
                "end_idx": 15,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 1,
                "chunk_index": 0,
                "num_tokens": 4,
                "content": "Second sentence. Third sentence.",
                "start_idx": 0,
                "end_idx": 32,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_index": 1,
                "chunk_index": 1,
                "num_tokens": 2,
                "content": "Fourth sentence",
                "start_idx": 33,
                "end_idx": 48,
                "line_idx": 0
            }
        ]

        # When chunk_texts_with_data is called
        result = chunk_texts_with_data(
            input_texts, chunk_size=5, chunk_overlap=0)

        # Then the result should process each string with correct metadata
        assert len(result) == len(expected)
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res["content"] == exp["content"]
            assert res["doc_index"] == exp["doc_index"]
            assert res["chunk_index"] == exp["chunk_index"]
            assert res["num_tokens"] == exp["num_tokens"]
            assert res["start_idx"] == exp["start_idx"]
            assert res["end_idx"] == exp["end_idx"]
            assert res["line_idx"] == exp["line_idx"]
