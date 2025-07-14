import pytest
import nltk
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
from typing import Dict, TypedDict, Callable, Union, List, Optional


class Metadata(TypedDict):
    start_idx: int
    end_idx: int


class ChunkResult(TypedDict):
    content: str
    num_tokens: int
    header: str
    parent_header: Optional[str]
    level: int
    parent_level: Optional[int]
    doc_index: int
    chunk_index: int
    metadata: Metadata


@pytest.fixture(scope="class")
def chunking_shared():
    def tokenizer(x):
        return nltk.word_tokenize(x) if isinstance(x, str) else [nltk.word_tokenize(t) for t in x]
    split_fn = nltk.sent_tokenize
    chunk_size = 16
    return tokenizer, split_fn, chunk_size


@pytest.fixture(scope="class")
def markdown_text():
    return """
# Root Header
This is a sentence in root.

## Level 2 Header
This is a very long sentence that fits chunksize.
Short sentence.
Joined short sentence for merging.

### Level 3 Header
This is another long sentence.
This is a long sibling sentence.
This is the 5th long sentence.
"""


class TestChunkHeadersByHierarchy:
    def test_chunk_headers_by_hierarchy_with_root(self, chunking_shared, markdown_text):
        tokenizer, split_fn, chunk_size = chunking_shared
        expected = [
            {
                "content": "This is a sentence in root.",
                "num_tokens": 10,
                "header": "# Root Header",
                "parent_header": None,
                "level": 1,
                "parent_level": None,
                "doc_index": 0,
                "chunk_index": 0,
                "metadata": {"start_idx": 14, "end_idx": 40}
            },
            {
                "content": "This is a very long sentence that fits chunksize.",
                "num_tokens": 15,
                "header": "## Level 2 Header",
                "parent_header": "# Root Header",
                "level": 2,
                "parent_level": 1,
                "doc_index": 1,
                "chunk_index": 0,
                "metadata": {"start_idx": 59, "end_idx": 106}
            },
            {
                "content": "Short sentence.\nJoined short sentence for merging.",
                "num_tokens": 14,
                "header": "## Level 2 Header",
                "parent_header": "# Root Header",
                "level": 2,
                "parent_level": 1,
                "doc_index": 1,
                "chunk_index": 1,
                "metadata": {"start_idx": 107, "end_idx": 162}
            },
            {
                "content": "This is another long sentence.",
                "num_tokens": 12,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 2,
                "chunk_index": 0,
                "metadata": {"start_idx": 181, "end_idx": 210}
            },
            {
                "content": "This is a long sibling sentence.",
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 2,
                "chunk_index": 1,
                "metadata": {"start_idx": 211, "end_idx": 242}
            },
            {
                "content": "This is the 5th long sentence.",
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 2,
                "chunk_index": 2,
                "metadata": {"start_idx": 243, "end_idx": 272}
            }
        ]
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)
        assert results == expected

    def test_chunk_headers_by_hierarchy_no_root(self, chunking_shared, markdown_text):
        # Generate initial chunks with no root
        markdown_text = "\n".join(line for line in markdown_text.splitlines()
                                  if not line.startswith("# Root Header") and "This is a sentence in root." not in line)
        tokenizer, split_fn, chunk_size = chunking_shared
        expected = [
            {
                "content": "This is a very long sentence that fits chunksize.",
                "num_tokens": 15,
                "header": "## Level 2 Header",
                "parent_header": None,
                "level": 2,
                "parent_level": None,
                "doc_index": 0,
                "chunk_index": 0,
                "metadata": {"start_idx": 19, "end_idx": 66}
            },
            {
                "content": "Short sentence.\nJoined short sentence for merging.",
                "num_tokens": 14,
                "header": "## Level 2 Header",
                "parent_header": None,
                "level": 2,
                "parent_level": None,
                "doc_index": 0,
                "chunk_index": 1,
                "metadata": {"start_idx": 67, "end_idx": 122}
            },
            {
                "content": "This is another long sentence.",
                "num_tokens": 12,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 1,
                "chunk_index": 0,
                "metadata": {"start_idx": 141, "end_idx": 170}
            },
            {
                "content": "This is a long sibling sentence.",
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 1,
                "chunk_index": 1,
                "metadata": {"start_idx": 171, "end_idx": 202}
            },
            {
                "content": "This is the 5th long sentence.",
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 1,
                "chunk_index": 2,
                "metadata": {"start_idx": 203, "end_idx": 232}
            }
        ]
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)
        assert results == expected
