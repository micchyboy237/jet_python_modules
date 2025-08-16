import pytest
from typing import List
from unittest.mock import Mock
from jet.wordnet.text_chunker import chunk_texts_with_data, ChunkResult
import uuid


class TestChunkTextsWithData:
    def test_repeated_sentences(self):
        # Given: A text with repeated sentences
        text = "Hello world. Hello world. This is a test."
        chunk_size = 3
        expected = [
            {
                "content": "Hello world.",
                "start_idx": 0,
                "end_idx": 12,
                "line_idx": 0,
                "num_tokens": 2,  # Assuming word-based tokenization
                "doc_id": "test_doc",
                "doc_index": 0,
                "chunk_index": 0,
                "overlap_start_idx": None,
                "overlap_end_idx": None
            },
            {
                "content": "Hello world.",
                "start_idx": 13,
                "end_idx": 25,
                "line_idx": 0,
                "num_tokens": 2,
                "doc_id": "test_doc",
                "doc_index": 0,
                "chunk_index": 1,
                "overlap_start_idx": None,
                "overlap_end_idx": None
            },
            {
                "content": "This is a test.",
                "start_idx": 26,
                "end_idx": 41,
                "line_idx": 0,
                "num_tokens": 4,
                "doc_id": "test_doc",
                "doc_index": 0,
                "chunk_index": 2,
                "overlap_start_idx": None,
                "overlap_end_idx": None
            }
        ]

        # When: Chunking the text
        result = chunk_texts_with_data(
            texts=[text],
            chunk_size=chunk_size,
            doc_ids=["test_doc"],
            model=None  # Use word-based tokenization
        )

        # Then: Verify chunks have correct start_idx, end_idx, and content
        for r, e in zip(result, expected):
            assert r["content"] == e["content"], f"Expected content {e['content']}, got {r['content']}"
            assert r["start_idx"] == e[
                "start_idx"], f"Expected start_idx {e['start_idx']}, got {r['start_idx']}"
            assert r["end_idx"] == e["end_idx"], f"Expected end_idx {e['end_idx']}, got {r['end_idx']}"
            assert r["num_tokens"] == e[
                "num_tokens"], f"Expected num_tokens {e['num_tokens']}, got {r['num_tokens']}"
