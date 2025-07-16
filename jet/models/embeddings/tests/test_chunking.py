import pytest
import nltk
from jet.data.utils import generate_unique_id
from jet.models.embeddings.chunking import chunk_docs_by_hierarchy, chunk_headers_by_hierarchy
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
                "id": "86472f83-3503-46cb-b80d-63d4a5aaf564",
                "parent_id": None,
                "header_doc_id": "dce2fa07-5470-4822-90a4-2bdc481cc41c",
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
                "id": "0511bc15-0b5d-437e-8fdc-e105db96fe1c",
                "parent_id": "dce2fa07-5470-4822-90a4-2bdc481cc41c",
                "header_doc_id": "9424a3d0-8dd4-4965-a268-6d45e0d590aa",
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
                "id": "e87d7ed2-7017-4dad-8b90-a1ec5af65bd4",
                "parent_id": "dce2fa07-5470-4822-90a4-2bdc481cc41c",
                "header_doc_id": "9424a3d0-8dd4-4965-a268-6d45e0d590aa",
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
                "id": "7f79b935-11f2-4e45-9fa1-918c0a8cf467",
                "parent_id": "9424a3d0-8dd4-4965-a268-6d45e0d590aa",
                "header_doc_id": "dd52452f-2e93-47e9-9bbc-18c8bac87fe5",
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
                "id": "897c107f-8138-4b62-a5cd-386589d4fcd4",
                "parent_id": "9424a3d0-8dd4-4965-a268-6d45e0d590aa",
                "header_doc_id": "dd52452f-2e93-47e9-9bbc-18c8bac87fe5",
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
                "id": "2449d6a6-0a71-4944-bb47-9a61006adbf9",
                "parent_id": "9424a3d0-8dd4-4965-a268-6d45e0d590aa",
                "header_doc_id": "dd52452f-2e93-47e9-9bbc-18c8bac87fe5",
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
            assert isinstance(res["id"], str) and res["id"]
            assert isinstance(res["header_doc_id"],
                              str) and res["header_doc_id"]
            assert isinstance(res.get("parent_id", None), (str, type(None)))
            res_no_ids = {k: v for k, v in res.items() if k not in [
                "id", "header_doc_id", "parent_id"]}
            exp_no_ids = {k: v for k, v in exp.items() if k not in [
                "id", "header_doc_id", "parent_id"]}
            assert res_no_ids == exp_no_ids

    def test_chunk_headers_by_hierarchy_no_root(self, chunking_shared, markdown_text):
        # Generate initial chunks with no root
        markdown_text = "\n".join(line for line in markdown_text.splitlines()
                                  if not line.startswith("# Root Header") and "This is a sentence in root." not in line)
        tokenizer, split_fn, chunk_size = chunking_shared
        expected = [
            {
                "id": "6fd6dacb-025b-4e91-9e7f-d3c48367ddaa",
                "parent_id": None,
                "header_doc_id": "515581a4-941e-48e8-b658-03c698fae9d5",
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
                "id": "03b8dbdb-b6f5-4a82-a803-c10278dfabbb",
                "parent_id": None,
                "header_doc_id": "515581a4-941e-48e8-b658-03c698fae9d5",
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
                "id": "8778a759-c476-4a91-b1b9-b89fabcdaadd",
                "parent_id": "515581a4-941e-48e8-b658-03c698fae9d5",
                "header_doc_id": "4d7c36a2-8450-48f4-bdb2-904d90067ee3",
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
                "id": "5f494cee-3371-42c0-9d02-b222c8a83e8f",
                "parent_id": "515581a4-941e-48e8-b658-03c698fae9d5",
                "header_doc_id": "4d7c36a2-8450-48f4-bdb2-904d90067ee3",
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
                "id": "0970804a-2296-4a09-a49b-f439f4c6f974",
                "parent_id": "515581a4-941e-48e8-b658-03c698fae9d5",
                "header_doc_id": "4d7c36a2-8450-48f4-bdb2-904d90067ee3",
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
            assert isinstance(res["id"], str) and res["id"]
            assert isinstance(res["header_doc_id"],
                              str) and res["header_doc_id"]
            assert isinstance(res.get("parent_id", None), (str, type(None)))
            res_no_ids = {k: v for k, v in res.items() if k not in [
                "id", "header_doc_id", "parent_id"]}
            exp_no_ids = {k: v for k, v in exp.items() if k not in [
                "id", "header_doc_id", "parent_id"]}
            assert res_no_ids == exp_no_ids


class TestChunkDocsByHierarchy:
    def test_chunk_docs_by_hierarchy_multiple_docs(self, chunking_shared, markdown_text):
        # Given: Two markdown documents with hierarchical headers and content
        tokenizer, split_fn, chunk_size = chunking_shared
        doc1 = markdown_text
        doc2 = """
## Another Header
This is a different document.
Another sentence in this doc.

### Sub Header
This is a sub-level sentence.
Another sub-level sentence.
"""
        markdown_texts = [doc1, doc2]
        doc_ids = [generate_unique_id(), generate_unique_id()]

        # Expected results combining chunks from both documents
        expected = [
            {
                "id": "72791c29-c6d9-4c08-9405-393f22295029",
                "parent_id": None,
                "header_doc_id": "0243df06-9020-4822-90fc-5eaa78543be2",
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
                },
                "doc_id": "635ec309-ef09-42c1-b974-2f742fb8c35c"
            },
            {
                "id": "216ce372-987a-4576-a4d5-e3b08c3168c4",
                "parent_id": "0243df06-9020-4822-90fc-5eaa78543be2",
                "header_doc_id": "d0288004-ca75-4e02-b6db-aa4b9a470c51",
                "doc_index": 0,
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
                },
                "doc_id": "635ec309-ef09-42c1-b974-2f742fb8c35c"
            },
            {
                "id": "ef4870ab-3cf0-494b-804c-ebc117acbe39",
                "parent_id": "0243df06-9020-4822-90fc-5eaa78543be2",
                "header_doc_id": "d0288004-ca75-4e02-b6db-aa4b9a470c51",
                "doc_index": 0,
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
                },
                "doc_id": "635ec309-ef09-42c1-b974-2f742fb8c35c"
            },
            {
                "id": "5d5f1fe0-5232-4052-a991-a6d9d90f72d6",
                "parent_id": "d0288004-ca75-4e02-b6db-aa4b9a470c51",
                "header_doc_id": "8c2c3f71-2876-49b4-8d93-34b6d354d070",
                "doc_index": 0,
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
                },
                "doc_id": "635ec309-ef09-42c1-b974-2f742fb8c35c"
            },
            {
                "id": "3daceba1-0b6b-43a1-8073-2a21b2f0806a",
                "parent_id": "d0288004-ca75-4e02-b6db-aa4b9a470c51",
                "header_doc_id": "8c2c3f71-2876-49b4-8d93-34b6d354d070",
                "doc_index": 0,
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
                },
                "doc_id": "635ec309-ef09-42c1-b974-2f742fb8c35c"
            },
            {
                "id": "aef0e16f-c871-42f6-9bd7-a03acaeb20ef",
                "parent_id": "d0288004-ca75-4e02-b6db-aa4b9a470c51",
                "header_doc_id": "8c2c3f71-2876-49b4-8d93-34b6d354d070",
                "doc_index": 0,
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
                },
                "doc_id": "635ec309-ef09-42c1-b974-2f742fb8c35c"
            },
            {
                "id": "58cf2cc1-cee1-4a7d-88c6-50209a197354",
                "parent_id": None,
                "header_doc_id": "a319fcff-086a-4ecd-920b-5d02c1c84c27",
                "doc_index": 1,
                "chunk_index": 0,
                "num_tokens": 16,
                "header": "## Another Header",
                "parent_header": None,
                "content": "This is a different document.\nAnother sentence in this doc.",
                "level": 2,
                "parent_level": None,
                "metadata": {
                    "start_idx": 19,
                    "end_idx": 77
                },
                "doc_id": "f1f3bb98-1912-4aeb-8ae1-04495421dc59"
            },
            {
                "id": "854cbf57-5b21-4015-9217-92c5c91ebd06",
                "parent_id": "a319fcff-086a-4ecd-920b-5d02c1c84c27",
                "header_doc_id": "28f04b37-c340-4c16-9fb4-38d355c437e3",
                "doc_index": 1,
                "chunk_index": 0,
                "num_tokens": 15,
                "header": "### Sub Header",
                "parent_header": "## Another Header",
                "content": "This is a sub-level sentence.\nAnother sub-level sentence.",
                "level": 3,
                "parent_level": 2,
                "metadata": {
                    "start_idx": 95,
                    "end_idx": 151
                },
                "doc_id": "f1f3bb98-1912-4aeb-8ae1-04495421dc59"
            }
        ]

        # When: We chunk the documents using chunk_docs_by_hierarchy
        results = chunk_docs_by_hierarchy(
            markdown_texts, chunk_size, tokenizer, split_fn, doc_ids
        )

        # Then: The results should match the expected chunks with correct doc_ids
        assert len(results) == len(
            expected), f"Expected {len(expected)} chunks, got {len(results)}"
        for res, exp in zip(results, expected):  # Fixed invalid loop variable
            assert isinstance(
                res["id"], str) and res["id"], "Chunk ID must be a non-empty string"
            assert isinstance(
                res["header_doc_id"], str) and res["header_doc_id"], "Header doc ID must be a non-empty string"
            assert isinstance(res.get("parent_id", None), (str, type(
                None))), "Parent ID must be a string or None"
            assert isinstance(
                res["doc_id"], str) and res["doc_id"], "Doc ID must be a non-empty string"
            res_no_ids = {k: v for k, v in res.items() if k not in [
                "id", "header_doc_id", "parent_id", "doc_id"]}
            exp_no_ids = {k: v for k, v in exp.items() if k not in [
                "id", "header_doc_id", "parent_id", "doc_id"]}
            assert res_no_ids == exp_no_ids, f"Chunk mismatch: {res_no_ids} != {exp_no_ids}"
