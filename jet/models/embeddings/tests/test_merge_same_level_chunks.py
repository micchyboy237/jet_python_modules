import pytest
import nltk
from jet.models.embeddings.chunking import chunk_headers_by_hierarchy
from jet.models.embeddings.merge_same_level_chunks import merge_same_level_chunks


@pytest.fixture(scope="class")
def chunking_shared():
    def tokenizer(x):
        return nltk.word_tokenize(x) if isinstance(x, str) else [nltk.word_tokenize(t) for t in x]
    split_fn = nltk.sent_tokenize
    chunk_size = 16
    return tokenizer, split_fn, chunk_size


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
