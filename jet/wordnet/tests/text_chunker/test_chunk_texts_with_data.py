import pytest
from typing import Dict, List
from jet.wordnet.text_chunker import chunk_texts_with_data


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

    def test_small_final_chunk_merging(self):
        # Given: A text with enough content to create multiple chunks, with a small final chunk
        text = "This is sentence one. This is sentence two. This is sentence three. Short."
        chunk_size = 10  # Small chunk size for testing
        chunk_overlap = 2
        model = "embeddinggemma"
        expected_chunks: List[Dict] = [
            {
                "num_tokens": pytest.approx(10, abs=2),
                "content": "This is sentence one. This is sentence two."
            },
            {
                "num_tokens": pytest.approx(10, abs=2),
                "content": "This is sentence two. This is sentence three. Short."
            }
        ]

        # When: Chunking the text
        result = chunk_texts_with_data(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model, min_chunk_size=5
        )

        # Then: Verify chunks are merged correctly
        assert len(result) == len(expected_chunks), f"Expected {len(expected_chunks)} chunks, got {len(result)}"
        for i, (res, exp) in enumerate(zip(result, expected_chunks)):
            assert res["num_tokens"] >= 5, f"Chunk {i} has {res['num_tokens']} tokens, below minimum"
            assert res["content"] == exp["content"], f"Chunk {i} content mismatch"
            assert res["num_tokens"] == pytest.approx(exp["num_tokens"], abs=2), f"Chunk {i} token count mismatch"
                
    def test_no_duplicate_final_chunk(self):
        # Given: A text where the final chunk is small and contained in the previous chunk due to overlap
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunk_size = 10
        chunk_overlap = 5
        model = "embeddinggemma"
        expected_chunks: List[Dict] = [
            {
                "num_tokens": pytest.approx(10, abs=2),
                "content": "This is sentence one. This is sentence two."
            },
            {
                "num_tokens": pytest.approx(10, abs=2),
                "content": "This is sentence two. This is sentence three."
            }
        ]

        # When: Chunking the text with overlap
        result = chunk_texts_with_data(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, model=model, min_chunk_size=5
        )

        # Then: Verify no duplicate content in chunks
        assert len(result) == len(expected_chunks), f"Expected {len(expected_chunks)} chunks, got {len(result)}"
        for i, (res, exp) in enumerate(zip(result, expected_chunks)):
            assert res["num_tokens"] >= 5, f"Chunk {i} has {res['num_tokens']} tokens, below minimum"
            assert res["content"] == exp["content"], f"Chunk {i} content mismatch"
            assert res["num_tokens"] == pytest.approx(exp["num_tokens"], abs=2), f"Chunk {i} token count mismatch"
        # Ensure no duplicate content
        for i in range(len(result) - 1):
            assert result[i + 1]["content"] not in result[i]["content"], f"Chunk {i + 1} is fully contained in chunk {i}"
