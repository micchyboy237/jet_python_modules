import pytest
import nltk
from jet.models.embeddings.chunking import (
    chunk_headers_by_hierarchy,
)


@pytest.fixture(scope="class")
def chunking_shared():
    def tokenizer(x):
        return nltk.word_tokenize(x) if isinstance(x, str) else [nltk.word_tokenize(t) for t in x]
    split_fn = nltk.sent_tokenize
    chunk_size = 16
    return tokenizer, split_fn, chunk_size


class TestMergeTextsByHierarchy:
    def test_chunk_headers_by_hierarchy_with_root(self, chunking_shared):
        markdown_text = """
# Root Header
This is a sentence in root.
## Sub Header
This is a very long sentence that fits chunksize.
Short sentence.
Joined short sentence for merging.
### Sub Sub Header
This is another long sentence.
This is a long sibling sentence.
This is the 5th long sentence.
"""
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
                "chunk_index": 0
            },
            {
                "content": "This is a very long sentence that fits chunksize.",
                "num_tokens": 14,
                "header": "## Sub Header",
                "parent_header": "# Root Header",
                "level": 2,
                "parent_level": 1,
                "doc_index": 1,
                "chunk_index": 0
            },
            {
                "content": "Short sentence.\nJoined short sentence for merging.",
                "num_tokens": 13,
                "header": "## Sub Header",
                "parent_header": "# Root Header",
                "level": 2,
                "parent_level": 1,
                "doc_index": 1,
                "chunk_index": 1
            },
            {
                "content": "This is another long sentence.",
                "num_tokens": 12,
                "header": "### Sub Sub Header",
                "parent_header": "## Sub Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 2,
                "chunk_index": 0
            },
            {
                "content": "This is a long sibling sentence.",
                "num_tokens": 13,
                "header": "### Sub Sub Header",
                "parent_header": "## Sub Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 3,
                "chunk_index": 0
            },
            {
                "content": "This is the 5th long sentence.",
                "num_tokens": 13,
                "header": "### Sub Sub Header",
                "parent_header": "## Sub Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 4,
                "chunk_index": 0
            }
        ]
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)
        assert results == expected

    def test_chunk_headers_by_hierarchy_no_root(self, chunking_shared):
        markdown_text = """
## Sub Header
This is a very long sentence that fits chunksize.
Short sentence.
Joined short sentence for merging.
### Sub Sub Header
This is another long sentence.
This is a long sibling sentence.
This is the 5th long sentence.
"""
        tokenizer, split_fn, chunk_size = chunking_shared
        expected = [
            {
                "content": "This is a very long sentence that fits chunksize.",
                "num_tokens": 14,
                "header": "## Sub Header",
                "parent_header": None,
                "level": 2,
                "parent_level": None,
                "doc_index": 0,
                "chunk_index": 0
            },
            {
                "content": "Short sentence.\nJoined short sentence for merging.",
                "num_tokens": 13,
                "header": "## Sub Header",
                "parent_header": None,
                "level": 2,
                "parent_level": None,
                "doc_index": 0,
                "chunk_index": 1
            },
            {
                "content": "This is another long sentence.",
                "num_tokens": 12,
                "header": "### Sub Sub Header",
                "parent_header": "## Sub Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 1,
                "chunk_index": 0
            },
            {
                "content": "This is a long sibling sentence.",
                "num_tokens": 13,
                "header": "### Sub Sub Header",
                "parent_header": "## Sub Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 2,
                "chunk_index": 0
            },
            {
                "content": "This is the 5th long sentence.",
                "num_tokens": 13,
                "header": "### Sub Sub Header",
                "parent_header": "## Sub Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 3,
                "chunk_index": 0
            }
        ]
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)
        assert results == expected
