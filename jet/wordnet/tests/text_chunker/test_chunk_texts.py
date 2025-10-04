import pytest
from typing import List
from jet.wordnet.text_chunker import chunk_texts
from unittest.mock import patch

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
        chunk_size = 10
        expected = [
            "This is the first sentence. ",
            "Here is the second one. ",
            "And this is the third. ",
            "Final sentence here."
        ]

        # When: chunk_texts is called with strict_sentences=True
        result = chunk_texts(input_text, chunk_size=chunk_size, model="nomic-embed-text-v2-moe", strict_sentences=True)

        # Then: Chunks respect sentence boundaries
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_chunk_texts_non_strict_sentences(self, sample_text: str):
        # Given: A text and a chunk size with a model
        input_text = sample_text
        chunk_size = 10

        # When: chunk_texts is called with strict_sentences=False and a model
        with patch("jet.wordnet.text_chunker.get_tokenizer_fn") as mock_tokenizer:
            mock_tokenizer.return_value.encode.return_value = input_text.split()
            mock_tokenizer.return_value.decode.side_effect = lambda x: " ".join(x)
            result = chunk_texts(input_text, chunk_size=chunk_size, model="nomic-embed-text-v2-moe", strict_sentences=False)

        # Then: Chunks are based on tokens, not sentences
        assert len(result) > 0, "Expected non-empty chunks"
        assert all(len(mock_tokenizer.return_value.encode(chunk)) <= chunk_size for chunk in result), "Chunks exceed token size"
