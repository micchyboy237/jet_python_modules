import pytest
from typing import List
from jet._token.token_utils import token_counter
from jet.wordnet.text_chunker import chunk_texts, chunk_texts_with_data, truncate_texts

class TestTextChunker:
    @pytest.fixture
    def sample_text(self) -> str:
        return "This is the first sentence. Here is the second one. And this is the third. Final sentence here."

    @pytest.fixture
    def sample_texts(self) -> List[str]:
        return [
            "First doc sentence one. First doc sentence two.",
            "Second doc sentence one. Second doc sentence two. Second doc sentence three."
        ]

    def test_chunk_texts_strict_sentences(self, sample_text: str):
        # Given: A text with multiple sentences and a small chunk size
        input_text = sample_text
        chunk_size = 12
        expected = [
            "This is the first sentence. Here is the second one.",
            "And this is the third. Final sentence here."
        ]

        # When: chunk_texts is called with strict_sentences=True
        results = chunk_texts(input_text, chunk_size=chunk_size, model="llama3.2", strict_sentences=True)
        token_counter(results, "llama3.2", prevent_total=True)
        # Then: Chunks respect sentence boundaries
        assert results == expected, f"Expected {expected}, but got {results}"

    def test_chunk_texts_non_strict_sentences(self, sample_text: str):
        # Given: A text and a chunk size with a model
        input_text = sample_text
        chunk_size = 10

        results = chunk_texts(input_text, chunk_size=chunk_size, model="llama3.2", strict_sentences=False)
        token_counts = token_counter(results, "llama3.2", prevent_total=True)

        # Then: Chunks are based on tokens, not sentences
        assert len(results) > 0, "Expected non-empty chunks"
        assert all(chunk_tokens <= chunk_size for chunk_tokens in token_counts), "Chunks exceed token size"

    def test_chunk_texts_with_data_strict_sentences(self, sample_text: str):
        # Given: A text with a small chunk size and doc_id
        input_text = sample_text
        chunk_size = 10
        doc_id = "test_doc"
        expected_num_chunks = 3

        # When: chunk_texts_with_data is called with strict_sentences=True
        results = chunk_texts_with_data([input_text], chunk_size=chunk_size, model="llama3.2", doc_ids=[doc_id], strict_sentences=True)

        # Then: Chunks respect sentence boundaries and include metadata
        assert len(results) == expected_num_chunks, f"Expected {expected_num_chunks} chunks, got {len(results)}"
        assert all(r["doc_id"] == doc_id for r in results), "Doc IDs mismatch"
        assert all(r["content"] in input_text for r in results), "Chunk content not in input text"

    def test_chunk_texts_with_data_non_strict_sentences(self, sample_text: str):
        # Given: A text and a chunk size with a model
        input_text = sample_text
        chunk_size = 10
        doc_id = "test_doc"

        results = chunk_texts_with_data([input_text], chunk_size=chunk_size, model="llama3.2", doc_ids=[doc_id], strict_sentences=False)

        # Then: Chunks are based on tokens with correct metadata
        assert len(results) > 0, "Expected non-empty chunks"
        assert all(r["num_tokens"] <= chunk_size for r in results), "Chunks exceed token size"
        assert all(r["doc_id"] == doc_id for r in results), "Doc IDs mismatch"

    def test_truncate_texts_strict_sentences(self, sample_text: str):
        # Given: A text and a max token limit
        input_text = sample_text
        max_tokens = 15
        expected = "This is the first sentence. Here is the second one."

        results = truncate_texts(input_text, model="llama3.2", max_tokens=max_tokens, strict_sentences=True)

        # Then: Truncation respects sentence boundaries
        assert results == [expected], f"Expected {[expected]}, but got {results}"

    def test_truncate_texts_non_strict_sentences(self, sample_text: str):
        # Given: A text and a max token limit
        input_text = sample_text
        max_tokens = 15

        # When: truncate_texts is called with strict_sentences=False
        results = truncate_texts(input_text, model="llama3.2", max_tokens=max_tokens, strict_sentences=False)

        # Then: Truncation is based on tokens
        assert len(results[0].split()) <= max_tokens, "Truncated text exceeds token limit"
