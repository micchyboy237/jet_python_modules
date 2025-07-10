import pytest
import nltk
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy, merge_same_level_chunks


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
                "doc_index": 3,
                "chunk_index": 0
            },
            {
                "content": "This is the 5th long sentence.",
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 4,
                "chunk_index": 0
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
                "doc_index": 2,
                "chunk_index": 0
            },
            {
                "content": "This is the 5th long sentence.",
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "level": 3,
                "parent_level": 2,
                "doc_index": 3,
                "chunk_index": 0
            }
        ]
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)
        assert results == expected


@pytest.fixture(scope="class")
def markdown_text_multi_level2():
    return """
# Root Header
Root content.

## Level 2 First
First short.

## Level 2 Second
Second short.

## Level 2 Third
Third short.

### Level 3 Header
Level 3 content.
"""


class TestMergeSameLevelChunks:
    # Shared chunk configuration for readability
    BASE_CHUNK = {
        "parent_header": None,
        "parent_level": None,
    }

    def test_merge_same_level_chunks_with_root(self, chunking_shared, markdown_text_multi_level2):
        tokenizer, split_fn, chunk_size = chunking_shared

        # Generate initial chunks
        chunks = chunk_headers_by_hierarchy(
            markdown_text_multi_level2, chunk_size, tokenizer, split_fn)

        # Expected merged chunks
        expected = [
            # Root level document
            {
                **self.BASE_CHUNK,
                "content": "# Root Header\nRoot content.",
                "level": 1,
                "headers": ["# Root Header"],
                "header": "# Root Header",
                "num_tokens": 4,
                "doc_index": 0,
                "chunk_count": 1
            },
            # Level 2 documents merged
            {
                **self.BASE_CHUNK,
                "content": "## Level 2 First\nFirst short.\n\n## Level 2 Second\nSecond short.\n\n## Level 2 Third\nThird short.",
                "level": 2,
                "headers": ["## Level 2 First", "## Level 2 Second", "## Level 2 Third"],
                "header": "## Level 2 First\n## Level 2 Second\n## Level 2 Third",
                "num_tokens": 15,
                "parent_header": "# Root Header",
                "parent_level": 1,
                "doc_index": 1,
                "chunk_count": 3
            },
            # Level 3 document
            {
                **self.BASE_CHUNK,
                "content": "### Level 3 Header\nLevel 3 content.",
                "level": 3,
                "headers": ["### Level 3 Header"],
                "header": "### Level 3 Header",
                "num_tokens": 6,
                "parent_header": "## Level 2 Third",
                "parent_level": 2,
                "doc_index": 2,
                "chunk_count": 1
            }
        ]
        results = merge_same_level_chunks(chunks, chunk_size, tokenizer)
        assert results == expected

    def test_merge_same_level_chunks_within_chunk_size(self, chunking_shared, markdown_text_multi_level2):
        tokenizer, split_fn, chunk_size = chunking_shared

        # Generate initial chunks
        chunks = chunk_headers_by_hierarchy(
            markdown_text_multi_level2, chunk_size, tokenizer, split_fn)

        # Expected merged chunks
        expected = [
            # Root level document
            {
                **self.BASE_CHUNK,
                "content": "# Root Header\nRoot content.",
                "level": 1,
                "headers": ["# Root Header"],
                "header": "# Root Header",
                "num_tokens": 4,
                "doc_index": 0,
                "chunk_count": 1
            },
            # Level 2 documents merged (different doc_index, within chunk_size)
            {
                **self.BASE_CHUNK,
                "content": "## Level 2 First\nFirst short.\n\n## Level 2 Second\nSecond short.\n\n## Level 2 Third\nThird short.",
                "level": 2,
                "headers": ["## Level 2 First", "## Level 2 Second", "## Level 2 Third"],
                "header": "## Level 2 First\n## Level 2 Second\n## Level 2 Third",
                "num_tokens": 15,
                "parent_header": "# Root Header",
                "parent_level": 1,
                "doc_index": 1,
                "chunk_count": 3
            },
            # Level 3 document
            {
                **self.BASE_CHUNK,
                "content": "### Level 3 Header\nLevel 3 content.",
                "level": 3,
                "headers": ["### Level 3 Header"],
                "header": "### Level 3 Header",
                "num_tokens": 6,
                "parent_header": "## Level 2 Third",
                "parent_level": 2,
                "doc_index": 2,
                "chunk_count": 1
            }
        ]
        results = merge_same_level_chunks(chunks, chunk_size, tokenizer)
        assert results == expected
