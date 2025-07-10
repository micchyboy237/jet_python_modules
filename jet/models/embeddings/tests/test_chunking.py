import pytest
import nltk
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy


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
                "chunk_index": 0
            },
            {
                "content": "This is a very long sentence that fits chunksize.",
                "num_tokens": 15,
                "header": "## Level 2 Header",
                "parent_header": "# Root Header",
                "level": 2,
                "parent_level": 1,
                "doc_index": 1,
                "chunk_index": 0
            },
            {
                "content": "Short sentence.\nJoined short sentence for merging.",
                "num_tokens": 14,
                "header": "## Level 2 Header",
                "parent_header": "# Root Header",
                "level": 2,
                "parent_level": 1,
                "doc_index": 1,
                "chunk_index": 1
            },
            {
                "content": "This is another long sentence.",
                "num_tokens": 12,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 2,
                "chunk_index": 0
            },
            {
                "content": "This is a long sibling sentence.",
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 2,
                "chunk_index": 1
            },
            {
                "content": "This is the 5th long sentence.",
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 2,
                "chunk_index": 2
            }
        ]
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)
        assert results == expected

    def test_chunk_headers_by_hierarchy_no_root(self, chunking_shared, markdown_text):
        # Generate initial chunks with no root
        markdown_text = "\n".join(line for line in markdown_text.splitlines(
        ) if not line.startswith("# Root Header") and "This is a sentence in root." not in line)
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
                "chunk_index": 0
            },
            {
                "content": "Short sentence.\nJoined short sentence for merging.",
                "num_tokens": 14,
                "header": "## Level 2 Header",
                "parent_header": None,
                "level": 2,
                "parent_level": None,
                "doc_index": 0,
                "chunk_index": 1
            },
            {
                "content": "This is another long sentence.",
                "num_tokens": 12,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 1,
                "chunk_index": 0
            },
            {
                "content": "This is a long sibling sentence.",
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 1,
                "chunk_index": 1
            },
            {
                "content": "This is the 5th long sentence.",
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 1,
                "chunk_index": 2
            }
        ]
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)
        assert results == expected

    # def test_chunk_headers_by_hierarchy_with_overlap(self, chunking_shared):
    #     tokenizer, split_fn, chunk_size = chunking_shared
    #     markdown_text = """## Level 2 Header
    # First sentence here.
    # Second sentence here.
    # Third sentence here."""
    #     chunk_overlap = 3
    #     results = chunk_headers_by_hierarchy(
    #         markdown_text, chunk_size, tokenizer, split_fn, chunk_overlap)
    #     expected = [
    #         {
    #             "content": "First sentence here.",
    #             "num_tokens": 7,
    #             "header": "## Level 2 Header",
    #             "parent_header": None,
    #             "level": 2,
    #             "parent_level": None,
    #             "doc_index": 0,
    #             "chunk_index": 0
    #         },
    #         {
    #             "content": "sentence here.\nSecond sentence here.",
    #             "num_tokens": 10,
    #             "header": "## Level 2 Header",
    #             "parent_header": None,
    #             "level": 2,
    #             "parent_level": None,
    #             "doc_index": 0,
    #             "chunk_index": 1
    #         },
    #         {
    #             "content": "sentence here.\nThird sentence here.",
    #             "num_tokens": 10,
    #             "header": "## Level 2 Header",
    #             "parent_header": None,
    #             "level": 2,
    #             "parent_level": None,
    #             "doc_index": 0,
    #             "chunk_index": 2
    #         }
    #     ]
    #     assert results == expected
