import pytest
import nltk
from typing import List, Dict, Callable, Any
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy


@pytest.fixture
def tokenizer():
    def _tokenizer(text: Any) -> List[str]:
        if isinstance(text, str):
            return nltk.word_tokenize(text)
        return [nltk.word_tokenize(t) for t in text]
    return _tokenizer


@pytest.fixture
def split_fn():
    return nltk.sent_tokenize


@pytest.fixture
def default_chunk_size():
    return 16


class TestChunkHeadersByHierarchyWithRoot:
    def test_basic_hierarchy_with_root(self, tokenizer: Callable, split_fn: Callable, default_chunk_size: int):
        # Given: A markdown text with root header and multiple sentences
        markdown_text = (
            "# Root\n"
            "This is a sentence in root.\n"
            "## Subheader\n"
            "This is a very long sentence that fits chunksize. Short sentence.\n"
            "Joined short sentence for merging.\n"
            "### Subsubheader\n"
            "This is another long sentence.\n"
            "### Sibling\n"
            "This is a long sibling sentence.\n"
            "#### Subsubsubheader\n"
            "This is the 5th long sentence."
        )
        # When: Chunking with default parameters
        expected = [
            {
                "content": "This is a sentence in root.",
                "num_tokens": 9,
                "header": "# Root",
                "parent_header": None,
                "level": 1
            },
            {
                "content": "This is a very long sentence that fits chunksize.",
                "num_tokens": 16,
                "header": "## Subheader",
                "parent_header": "# Root",
                "level": 2
            },
            {
                "content": "Short sentence.\nJoined short sentence for merging.",
                "num_tokens": 15,
                "header": "## Subheader",
                "parent_header": "# Root",
                "level": 2
            },
            {
                "content": "This is another long sentence.",
                "num_tokens": 11,
                "header": "### Subsubheader",
                "parent_header": "## Subheader",
                "level": 3
            },
            {
                "content": "This is a long sibling sentence.",
                "num_tokens": 11,
                "header": "### Sibling",
                "parent_header": "## Subheader",
                "level": 3
            },
            {
                "content": "This is the 5th long sentence.",
                "num_tokens": 11,
                "header": "#### Subsubsubheader",
                "parent_header": "### Sibling",
                "level": 4
            },
        ]
        result = chunk_headers_by_hierarchy(
            markdown_text, default_chunk_size, tokenizer, split_fn)
        # Then: Results match expected chunks
        assert result == expected, "Expected chunked hierarchy with root to match"

    def test_hierarchy_with_root_and_overlap(self, tokenizer: Callable, split_fn: Callable, default_chunk_size: int):
        # Given: A markdown text with root header and overlap
        markdown_text = (
            "# Root\n"
            "This is a sentence in root.\n"
            "## Subheader\n"
            "This is a very long sentence that fits chunksize.\n"
            "Short sentence.\n"
        )
        chunk_overlap = 5
        # When: Chunking with overlap
        expected = [
            {
                "content": "This is a sentence in root.",
                "num_tokens": 9,
                "header": "# Root",
                "parent_header": None,
                "level": 1
            },
            {
                "content": "in root. This is a very long sentence that fits chunksize.",
                "num_tokens": 16,
                "header": "## Subheader",
                "parent_header": "# Root",
                "level": 2
            },
            {
                "content": "that fits chunksize. Short sentence.",
                "num_tokens": 11,
                "header": "## Subheader",
                "parent_header": "# Root",
                "level": 2
            },
        ]
        result = chunk_headers_by_hierarchy(
            markdown_text, default_chunk_size, tokenizer, split_fn, chunk_overlap)
        # Then: Results include overlapping tokens
        assert result == expected, "Expected chunked hierarchy with overlap to match"


class TestChunkHeadersByHierarchyNoRoot:
    def test_hierarchy_without_root(self, tokenizer: Callable, split_fn: Callable, default_chunk_size: int):
        # Given: A markdown text without root header
        markdown_text = (
            "## Subheader\n"
            "This is a very long sentence that fits chunksize. Short sentence.\n"
            "Joined short sentence for merging.\n"
            "### Subsubheader\n"
            "This is another long sentence.\n"
            "### Sibling\n"
            "This is a long sibling sentence.\n"
            "#### Subsubsubheader\n"
            "This is the 5th long sentence."
        )
        # When: Chunking with default parameters
        expected = [
            {
                "content": "This is a very long sentence that fits chunksize.",
                "num_tokens": 16,
                "header": "## Subheader",
                "parent_header": None,
                "level": 2
            },
            {
                "content": "Short sentence.\nJoined short sentence for merging.",
                "num_tokens": 15,
                "header": "## Subheader",
                "parent_header": None,
                "level": 2
            },
            {
                "content": "This is another long sentence.",
                "num_tokens": 11,
                "header": "### Subsubheader",
                "parent_header": "## Subheader",
                "level": 3
            },
            {
                "content": "This is a long sibling sentence.",
                "num_tokens": 11,
                "header": "### Sibling",
                "parent_header": "## Subheader",
                "level": 3
            },
            {
                "content": "This is the 5th long sentence.",
                "num_tokens": 11,
                "header": "#### Subsubsubheader",
                "parent_header": "### Sibling",
                "level": 4
            },
        ]
        result = chunk_headers_by_hierarchy(
            markdown_text, default_chunk_size, tokenizer, split_fn)
        # Then: Results match expected chunks
        assert result == expected, "Expected chunked hierarchy without root to match"

    def test_hierarchy_without_root_and_overlap(self, tokenizer: Callable, split_fn: Callable, default_chunk_size: int):
        # Given: A markdown text without root header and overlap
        markdown_text = (
            "## Subheader\n"
            "This is a very long sentence that fits chunksize.\n"
            "Short sentence.\n"
        )
        chunk_overlap = 5
        # When: Chunking with overlap
        expected = [
            {
                "content": "This is a very long sentence that fits chunksize.",
                "num_tokens": 16,
                "header": "## Subheader",
                "parent_header": None,
                "level": 2
            },
            {
                "content": "that fits chunksize. Short sentence.",
                "num_tokens": 11,
                "header": "## Subheader",
                "parent_header": None,
                "level": 2
            },
        ]
        result = chunk_headers_by_hierarchy(
            markdown_text, default_chunk_size, tokenizer, split_fn, chunk_overlap)
        # Then: Results include overlapping tokens
        assert result == expected, "Expected chunked hierarchy without root and overlap to match"


class TestChunkHeadersByHierarchyEdgeCases:
    def test_empty_input(self, tokenizer: Callable, split_fn: Callable, default_chunk_size: int):
        # Given: An empty markdown text
        markdown_text = ""
        # When: Chunking with default parameters
        expected: List[Dict[str, Any]] = []
        # Then: Empty input returns empty list
        result = chunk_headers_by_hierarchy(
            markdown_text, default_chunk_size, tokenizer, split_fn)
        assert result == expected, "Expected empty input to return empty list"

    def test_only_headers_no_content(self, tokenizer: Callable, split_fn: Callable, default_chunk_size: int):
        # Given: Markdown text with only headers
        markdown_text = "# Header\n## Subheader\n### Subsubheader\n"
        # When: Chunking with default parameters
        expected = []
        # Then: Headers with no content return empty list
        result = chunk_headers_by_hierarchy(
            markdown_text, default_chunk_size, tokenizer, split_fn)
        assert result == expected, "Expected headers with no content to be handled correctly"

    def test_large_chunk_size(self, tokenizer: Callable, split_fn: Callable):
        # Given: A markdown text with large chunk size
        markdown_text = (
            "# Root\n"
            "This is a very long sentence that fits chunksize. Short sentence.\n"
            "Joined short sentence for merging."
        )
        chunk_size = 100
        # When: Chunking with large chunk size
        expected = [
            {
                "content": "This is a very long sentence that fits chunksize.\nShort sentence.\nJoined short sentence for merging.",
                "num_tokens": 21,
                "header": "# Root",
                "parent_header": None,
                "level": 1
            },
        ]
        result = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)
        # Then: Results combine sentences into one chunk
        assert result == expected, "Expected large chunk size to combine sentences"

    def test_invalid_overlap(self, tokenizer: Callable, split_fn: Callable, default_chunk_size: int):
        # Given: A markdown text with invalid overlap
        markdown_text = (
            "## Subheader\n"
            "This is a very long sentence that fits chunksize."
        )
        chunk_overlap = default_chunk_size
        # When: Chunking with overlap equal to chunk size
        # Then: Raises ValueError
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            chunk_headers_by_hierarchy(
                markdown_text, default_chunk_size, tokenizer, split_fn, chunk_overlap)

    def test_small_overlap_with_single_sentence(self, tokenizer: Callable, split_fn: Callable, default_chunk_size: int):
        # Given: A markdown text with a single sentence and small overlap
        markdown_text = (
            "## Subheader\n"
            "This is a very long sentence that fits chunksize."
        )
        chunk_overlap = 3
        # When: Chunking with small overlap
        expected = [
            {
                "content": "This is a very long sentence that fits chunksize.",
                "num_tokens": 16,
                "header": "## Subheader",
                "parent_header": None,
                "level": 2
            },
        ]
        result = chunk_headers_by_hierarchy(
            markdown_text, default_chunk_size, tokenizer, split_fn, chunk_overlap)
        # Then: Results match single chunk with no overlap needed
        assert result == expected, "Expected single sentence with small overlap to match"
