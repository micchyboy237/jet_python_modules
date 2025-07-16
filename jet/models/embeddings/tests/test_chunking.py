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
                "doc_id": "46e2c430-ddef-4d34-adb7-6fdc73c59fce",
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 10,
                "header": "# Root Header",
                "parent_header": None,
                "content": "This is a sentence in root.",
                "level": 1,
                "parent_level": None,
                "metadata": {
                    "start_idx": 15,
                    "end_idx": 41
                }
            },
            {
                "doc_id": "7f441525-d17c-42c3-83fe-2c5439172b7b",
                "doc_index": 1,
                "chunk_index": 0,
                "num_tokens": 15,
                "header": "## Level 2 Header",
                "parent_header": "# Root Header",
                "content": "This is a very long sentence that fits chunksize.",
                "level": 2,
                "parent_level": 1,
                "metadata": {
                    "start_idx": 62,
                    "end_idx": 110
                }
            },
            {
                "doc_id": "7f441525-d17c-42c3-83fe-2c5439172b7b",
                "doc_index": 1,
                "chunk_index": 1,
                "num_tokens": 14,
                "header": "## Level 2 Header",
                "parent_header": "# Root Header",
                "content": "Short sentence.\nJoined short sentence for merging.",
                "level": 2,
                "parent_level": 1,
                "metadata": {
                    "start_idx": 112,
                    "end_idx": 161
                }
            },
            {
                "doc_id": "d8a92ae1-da89-4d6f-88ad-1ebdc0a2e949",
                "doc_index": 2,
                "chunk_index": 0,
                "num_tokens": 12,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "content": "This is another long sentence.",
                "level": 3,
                "parent_level": 2,
                "metadata": {
                    "start_idx": 183,
                    "end_idx": 212
                }
            },
            {
                "doc_id": "d8a92ae1-da89-4d6f-88ad-1ebdc0a2e949",
                "doc_index": 2,
                "chunk_index": 1,
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "content": "This is a long sibling sentence.",
                "level": 3,
                "parent_level": 2,
                "metadata": {
                    "start_idx": 214,
                    "end_idx": 245
                }
            },
            {
                "doc_id": "d8a92ae1-da89-4d6f-88ad-1ebdc0a2e949",
                "doc_index": 2,
                "chunk_index": 2,
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "content": "This is the 5th long sentence.",
                "level": 3,
                "parent_level": 2,
                "metadata": {
                    "start_idx": 247,
                    "end_idx": 276
                }
            }
        ]
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)
        assert len(results) == len(expected)
        for res, exp in zip(results, expected):
            # Check doc_id is present and is a string
            assert isinstance(res["doc_id"], str) and res["doc_id"]
            # Compare all other fields except doc_id
            res_no_id = {k: v for k, v in res.items() if k != "doc_id"}
            exp_no_id = {k: v for k, v in exp.items() if k != "doc_id"}
            assert res_no_id == exp_no_id

    def test_chunk_headers_by_hierarchy_no_root(self, chunking_shared, markdown_text):
        # Generate initial chunks with no root
        markdown_text = "\n".join(line for line in markdown_text.splitlines()
                                  if not line.startswith("# Root Header") and "This is a sentence in root." not in line)
        tokenizer, split_fn, chunk_size = chunking_shared
        expected = [
            {
                "doc_id": "a4fa0490-7fae-4425-b72d-822a928b8cc4",
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 15,
                "header": "## Level 2 Header",
                "parent_header": None,
                "content": "This is a very long sentence that fits chunksize.",
                "level": 2,
                "parent_level": None,
                "metadata": {
                    "start_idx": 19,
                    "end_idx": 67
                }
            },
            {
                "doc_id": "a4fa0490-7fae-4425-b72d-822a928b8cc4",
                "doc_index": 0,
                "chunk_index": 1,
                "num_tokens": 14,
                "header": "## Level 2 Header",
                "parent_header": None,
                "content": "Short sentence.\nJoined short sentence for merging.",
                "level": 2,
                "parent_level": None,
                "metadata": {
                    "start_idx": 69,
                    "end_idx": 118
                }
            },
            {
                "doc_id": "46e2a72a-9daa-49ea-b49b-b996b5b45ef5",
                "doc_index": 1,
                "chunk_index": 0,
                "num_tokens": 12,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "content": "This is another long sentence.",
                "level": 3,
                "parent_level": 2,
                "metadata": {
                    "start_idx": 140,
                    "end_idx": 169
                }
            },
            {
                "doc_id": "46e2a72a-9daa-49ea-b49b-b996b5b45ef5",
                "doc_index": 1,
                "chunk_index": 1,
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "content": "This is a long sibling sentence.",
                "level": 3,
                "parent_level": 2,
                "metadata": {
                    "start_idx": 171,
                    "end_idx": 202
                }
            },
            {
                "doc_id": "46e2a72a-9daa-49ea-b49b-b996b5b45ef5",
                "doc_index": 1,
                "chunk_index": 2,
                "num_tokens": 13,
                "header": "### Level 3 Header",
                "parent_header": "## Level 2 Header",
                "content": "This is the 5th long sentence.",
                "level": 3,
                "parent_level": 2,
                "metadata": {
                    "start_idx": 204,
                    "end_idx": 233
                }
            }
        ]
        results = chunk_headers_by_hierarchy(
            markdown_text, chunk_size, tokenizer, split_fn)
        assert len(results) == len(expected)
        for res, exp in zip(results, expected):
            # Check doc_id is present and is a string
            assert isinstance(res["doc_id"], str) and res["doc_id"]
            # Compare all other fields except doc_id
            res_no_id = {k: v for k, v in res.items() if k != "doc_id"}
            exp_no_id = {k: v for k, v in exp.items() if k != "doc_id"}
            assert res_no_id == exp_no_id
