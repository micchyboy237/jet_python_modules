import pytest
from typing import Optional, Union
from tokenizers import Tokenizer
from jet.data.header_utils._base import chunk_content
from jet.models.model_types import ModelType
from jet.models.tokenizer.base import get_tokenizer, count_tokens


class TestChunkContent:
    """Unit tests for the chunk_content function."""

    @pytest.fixture(scope="class")
    def tokenizer(self) -> Tokenizer:
        """Fixture to provide a tokenizer for all tests."""
        return get_tokenizer("all-MiniLM-L6-v2")

    def test_empty_content(self, tokenizer: Tokenizer):
        """Test chunk_content with empty content."""
        # Given
        content = ""
        chunk_size = 10
        chunk_overlap = 2
        buffer = 2
        expected = []

        # When
        result = chunk_content(
            content, tokenizer, chunk_size, chunk_overlap, buffer)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_none_chunk_size(self, tokenizer: Tokenizer):
        """Test chunk_content with None chunk_size."""
        # Given
        content = "This is a test string."
        chunk_size = None
        chunk_overlap = 2
        buffer = 2
        expected = [content]

        # When
        result = chunk_content(
            content, tokenizer, chunk_size, chunk_overlap, buffer)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_none_tokenizer(self):
        """Test chunk_content with None tokenizer."""
        # Given
        content = "This is a test string."
        chunk_size = 10
        chunk_overlap = 2
        buffer = 2
        expected = [content]

        # When
        result = chunk_content(content, None, chunk_size,
                               chunk_overlap, buffer)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_small_content_single_chunk(self, tokenizer: Tokenizer):
        """Test chunk_content with content smaller than chunk size."""
        # Given
        content = "Short text."
        chunk_size = 50
        chunk_overlap = 5
        buffer = 5
        expected = [content]

        # When
        result = chunk_content(
            content, tokenizer, chunk_size, chunk_overlap, buffer)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_content_with_chunking(self, tokenizer: Tokenizer):
        """Test chunk_content with content requiring multiple chunks."""
        # Given
        content = "This is a long text that needs to be split into multiple chunks for processing."
        chunk_size = 10
        chunk_overlap = 2
        buffer = 2
        # Approximate expected chunks based on tokenization
        expected = [
            "This is a long text that needs to",
            "needs to be split into multiple chunks for",
            "chunks for processing."
        ]

        # When
        result = chunk_content(
            content, tokenizer, chunk_size, chunk_overlap, buffer)

        # Then
        assert len(result) == len(
            expected), f"Expected {len(expected)} chunks, but got {len(result)}"
        for i, (exp, res) in enumerate(zip(expected, result)):
            assert res.strip() == exp.strip(
            ), f"Chunk {i} mismatch: expected '{exp}', got '{res}'"

    def test_content_with_header_prefix(self, tokenizer: Tokenizer):
        """Test chunk_content with a header prefix."""
        # Given
        content = "This is the main content to be chunked."
        header_prefix = "Header Title\n"
        chunk_size = 10
        chunk_overlap = 2
        buffer = 2
        # Approximate expected chunks, including header
        expected = [
            "This is the main content to",
            "content to be chunked."
        ]

        # When
        result = chunk_content(
            content, tokenizer, chunk_size, chunk_overlap, buffer, header_prefix)

        # Then
        assert len(result) == len(
            expected), f"Expected {len(expected)} chunks, but got {len(result)}"
        for i, (exp, res) in enumerate(zip(expected, result)):
            assert res.strip() == exp.strip(
            ), f"Chunk {i} mismatch: expected '{exp}', got '{res}'"

    def test_large_content_with_zero_overlap(self, tokenizer: Tokenizer):
        """Test chunk_content with large content and zero overlap."""
        # Given
        content = "Word " * 100  # Creates a long string with repeated words
        chunk_size = 20
        chunk_overlap = 0
        buffer = 5
        # Approximate expected chunks (assuming ~5 tokens per "Word ")
        expected_chunk_count = 7  # Rough estimate based on token count

        # When
        result = chunk_content(
            content, tokenizer, chunk_size, chunk_overlap, buffer)
        token_counts = count_tokens(tokenizer, result, prevent_total=True)

        # Then
        assert len(
            result) == expected_chunk_count, f"Expected {expected_chunk_count} chunks, but got {len(result)}"
        for i, (chunk, token_count) in enumerate(zip(result, token_counts)):
            assert len(chunk.strip()) > 0, f"Chunk {i} is empty"
            assert token_count <= chunk_size - \
                buffer, f"Chunk {i} exceeds token limit"

    def test_invalid_chunk_size(self, tokenizer: Tokenizer):
        """Test chunk_content with negative or zero chunk_size."""
        # Given
        content = "This is a test string."
        chunk_size = 0
        chunk_overlap = 2
        buffer = 2
        expected = [content]

        # When
        result = chunk_content(
            content, tokenizer, chunk_size, chunk_overlap, buffer)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"
