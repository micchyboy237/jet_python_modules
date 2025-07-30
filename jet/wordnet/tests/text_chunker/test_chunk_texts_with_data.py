import pytest
from typing import List
from unittest.mock import Mock
from jet.wordnet.text_chunker import chunk_texts_with_data, ChunkResult
import uuid


class TestChunkTextsWithData:
    def test_empty_input(self):
        """Test chunk_texts_with_data with empty input string."""
        # Given: An empty input string
        input_text = ""
        # When: chunk_texts_with_data is called with empty input
        expected: List[ChunkResult] = []
        result = chunk_texts_with_data(
            input_text, chunk_size=128, chunk_overlap=0)
        # Then: An empty list is returned
        assert result == expected, "Expected empty list for empty input"

    def test_single_sentence_within_chunk_size(self):
        """Test chunk_texts_with_data with a single sentence fitting within chunk size."""
        # Given: A single sentence and chunk size larger than sentence
        input_text = "This is a test sentence."
        expected_sentences = ["This is a test sentence."]
        expected_words = ["This", "is", "a", "test", "sentence"]
        expected_doc_id = str(uuid.uuid4())
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_id,
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 5,
                "content": "This is a test sentence.",
                "start_idx": 0,
                "end_idx": 24,
                "line_idx": 0
            }
        ]
        # When: chunk_texts_with_data is called
        result = chunk_texts_with_data(
            input_text, chunk_size=10, chunk_overlap=0)
        # Then: A single chunk is returned with correct metadata
        assert len(result) == 1
        # doc_id is consistent
        assert result[0]["doc_id"] == result[0]["doc_id"]
        assert result[0]["content"] == expected[0]["content"]
        assert result[0]["doc_index"] == expected[0]["doc_index"]
        assert result[0]["chunk_index"] == expected[0]["chunk_index"]
        assert result[0]["num_tokens"] == expected[0]["num_tokens"]
        assert result[0]["start_idx"] == expected[0]["start_idx"]
        assert result[0]["end_idx"] == expected[0]["end_idx"]
        assert result[0]["line_idx"] == expected[0]["line_idx"]

    def test_multiple_sentences_exceeding_chunk_size(self):
        """Test chunk_texts_with_data with multiple sentences exceeding chunk size."""
        # Given: Multiple sentences with small chunk size
        input_text = "First sentence. Second sentence. Third sentence."
        expected_sentences = ["First sentence.",
                              "Second sentence.", "Third sentence."]
        expected_words = {
            "First sentence.": ["First", "sentence"],
            "Second sentence.": ["Second", "sentence"],
            "Third sentence.": ["Third", "sentence"]
        }
        expected_doc_id = str(uuid.uuid4())
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_id,
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 2,
                "content": "First sentence.",
                "start_idx": 0,
                "end_idx": 15,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_id,
                "doc_index": 0,
                "chunk_index": 1,
                "num_tokens": 2,
                "content": "Second sentence.",
                "start_idx": 16,
                "end_idx": 32,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_id,
                "doc_index": 0,
                "chunk_index": 2,
                "num_tokens": 2,
                "content": "Third sentence.",
                "start_idx": 33,
                "end_idx": 48,
                "line_idx": 0
            }
        ]
        # When: chunk_texts_with_data is called
        result = chunk_texts_with_data(
            input_text, chunk_size=2, chunk_overlap=0)
        # Then: Chunks are split correctly with consistent doc_id
        assert len(result) == len(expected)
        for i, (res, exp) in enumerate(zip(result, expected)):
            # Same doc_id across chunks
            assert res["doc_id"] == result[0]["doc_id"]
            assert res["content"] == exp["content"]
            assert res["doc_index"] == exp["doc_index"]
            assert res["chunk_index"] == exp["chunk_index"]
            assert res["num_tokens"] == exp["num_tokens"]
            assert res["start_idx"] == exp["start_idx"]
            assert res["end_idx"] == exp["end_idx"]
            assert res["line_idx"] == exp["line_idx"]

    def test_with_overlap(self):
        """Test chunk_texts_with_data with overlap, respecting sentence boundaries."""
        # Given: Multiple sentences with overlap
        input_text = "First sentence. Second sentence. Third sentence."
        expected_sentences = ["First sentence.",
                              "Second sentence.", "Third sentence."]
        expected_words = {
            "First sentence.": ["First", "sentence"],
            "Second sentence.": ["Second", "sentence"],
            "Third sentence.": ["Third", "sentence"]
        }
        expected_doc_id = str(uuid.uuid4())
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_id,
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 4,
                "content": "First sentence. Second sentence.",
                "start_idx": 0,
                "end_idx": 32,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_id,
                "doc_index": 0,
                "chunk_index": 1,
                "num_tokens": 4,
                "content": "Second sentence. Third sentence.",
                "start_idx": 16,
                "end_idx": 48,
                "line_idx": 0
            }
        ]
        # When: chunk_texts_with_data is called with overlap
        result = chunk_texts_with_data(
            input_text, chunk_size=4, chunk_overlap=2)
        # Then: Chunks include overlap with consistent doc_id
        assert len(result) == len(expected)
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res["doc_id"] == result[0]["doc_id"]
            assert res["content"] == exp["content"]
            assert res["doc_index"] == exp["doc_index"]
            assert res["chunk_index"] == exp["chunk_index"]
            assert res["num_tokens"] == exp["num_tokens"]
            assert res["start_idx"] == exp["start_idx"]
            assert res["end_idx"] == exp["end_idx"]
            assert res["line_idx"] == exp["line_idx"]

    def test_with_list_items(self):
        """Test chunk_texts_with_data with list items, ensuring correct combination."""
        # Given: Text with list items
        input_text = "1. First item. Second sentence."
        expected_sentences = ["1. First item.", "Second sentence."]
        expected_words = {
            "1. First item.": ["1", "First", "item"],
            "Second sentence.": ["Second", "sentence"]
        }
        expected_doc_id = str(uuid.uuid4())
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_id,
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 3,
                "content": "1. First item.",
                "start_idx": 0,
                "end_idx": 14,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_id,
                "doc_index": 0,
                "chunk_index": 1,
                "num_tokens": 2,
                "content": "Second sentence.",
                "start_idx": 15,
                "end_idx": 31,
                "line_idx": 0
            }
        ]
        # When: chunk_texts_with_data is called
        result = chunk_texts_with_data(
            input_text, chunk_size=3, chunk_overlap=0)
        # Then: List items are handled correctly with consistent doc_id
        assert len(result) == len(expected)
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res["doc_id"] == result[0]["doc_id"]
            assert res["content"] == exp["content"]
            assert res["doc_index"] == exp["doc_index"]
            assert res["chunk_index"] == exp["chunk_index"]
            assert res["num_tokens"] == exp["num_tokens"]
            assert res["start_idx"] == exp["start_idx"]
            assert res["end_idx"] == exp["end_idx"]
            assert res["line_idx"] == exp["line_idx"]

    def test_with_model_tokenizer(self):
        """Test chunk_texts_with_data using a model tokenizer."""
        # Given: A sentence with a specific tokenizer model
        input_text = "This is a test sentence."
        expected_sentences = ["This is a test sentence."]
        expected_tokens = ["token1", "token2", "token3"]
        expected_doc_id = str(uuid.uuid4())
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_id,
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 6,
                "content": "This is a test sentence.",
                "start_idx": 0,
                "end_idx": 24,
                "line_idx": 0
            }
        ]
        # When: chunk_texts_with_data is called with a model
        result = chunk_texts_with_data(
            input_text, chunk_size=5, chunk_overlap=0, model="all-MiniLM-L6-v2")
        # Then: Chunk is created with correct tokenizer and doc_id
        assert len(result) == 1
        assert result[0]["doc_id"] == result[0]["doc_id"]
        assert result[0]["content"] == expected[0]["content"]
        assert result[0]["doc_index"] == expected[0]["doc_index"]
        assert result[0]["chunk_index"] == expected[0]["chunk_index"]
        assert result[0]["num_tokens"] == expected[0]["num_tokens"]
        assert result[0]["start_idx"] == expected[0]["start_idx"]
        assert result[0]["end_idx"] == expected[0]["end_idx"]
        assert result[0]["line_idx"] == expected[0]["line_idx"]

    def test_list_of_strings(self):
        """Test chunk_texts_with_data with a list of strings."""
        # Given: A list of input texts
        input_texts = ["First sentence.",
                       "Second sentence. Third sentence. Fourth sentence"]
        expected_sentences = [["First sentence."], [
            "Second sentence."], ["Third sentence."], ["Fourth sentence"]]
        expected_words = {
            "First sentence.": ["First", "sentence"],
            "Second sentence.": ["Second", "sentence"],
            "Third sentence.": ["Third", "sentence"],
            "Fourth sentence": ["Fourth", "sentence"]
        }
        expected_doc_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_ids[0],
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 2,
                "content": "First sentence.",
                "start_idx": 0,
                "end_idx": 15,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_ids[1],
                "doc_index": 1,
                "chunk_index": 0,
                "num_tokens": 4,
                "content": "Second sentence. Third sentence.",
                "start_idx": 0,
                "end_idx": 32,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": expected_doc_ids[1],
                "doc_index": 1,
                "chunk_index": 1,
                "num_tokens": 2,
                "content": "Fourth sentence",
                "start_idx": 33,
                "end_idx": 48,
                "line_idx": 0
            }
        ]
        # When: chunk_texts_with_data is called with list of strings
        result = chunk_texts_with_data(
            input_texts, chunk_size=5, chunk_overlap=0)
        # Then: Chunks are created with correct doc_ids per document
        assert len(result) == len(expected)
        for i, (res, exp) in enumerate(zip(result, expected)):
            assert res["doc_id"] == result[i]["doc_id"]
            assert res["content"] == exp["content"]
            assert res["doc_index"] == exp["doc_index"]
            assert res["chunk_index"] == exp["chunk_index"]
            assert res["num_tokens"] == exp["num_tokens"]
            assert res["start_idx"] == exp["start_idx"]
            assert res["end_idx"] == exp["end_idx"]
            assert res["line_idx"] == exp["line_idx"]

    def test_with_custom_doc_ids(self):
        """Test chunk_texts_with_data using custom doc_ids."""
        # Given: A list of texts with custom doc_ids
        input_texts = ["Sentence one. Sentence two.",
                       "Another sentence.\nFinal sentence"]
        custom_ids = ["doc1", "doc2"]
        expected: List[ChunkResult] = [
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": "doc1",
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": 2,
                "content": "Sentence one.",
                "start_idx": 0,
                "end_idx": 13,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": "doc1",
                "doc_index": 0,
                "chunk_index": 1,
                "num_tokens": 2,
                "content": "Sentence two.",
                "start_idx": 14,
                "end_idx": 27,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": "doc2",
                "doc_index": 1,
                "chunk_index": 0,
                "num_tokens": 2,
                "content": "Another sentence.",
                "start_idx": 0,
                "end_idx": 17,
                "line_idx": 0
            },
            {
                "id": Mock(return_value=str(uuid.uuid4())),
                "doc_id": "doc2",
                "doc_index": 1,
                "chunk_index": 1,
                "num_tokens": 2,
                "content": "Final sentence",
                "start_idx": 18,
                "end_idx": 32,
                "line_idx": 1
            }
        ]
        # When: chunk_texts_with_data is called with custom doc_ids
        result = chunk_texts_with_data(
            input_texts,
            chunk_size=2,
            chunk_overlap=0,
            doc_ids=custom_ids
        )
        # Then: Chunks use custom doc_ids correctly
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert res["doc_id"] == exp["doc_id"]
            assert res["content"] == exp["content"]
            assert res["doc_index"] == exp["doc_index"]
            assert res["chunk_index"] == exp["chunk_index"]
            assert res["num_tokens"] == exp["num_tokens"]
            assert res["start_idx"] == exp["start_idx"]
            assert res["end_idx"] == exp["end_idx"]
            assert res["line_idx"] == exp["line_idx"]
