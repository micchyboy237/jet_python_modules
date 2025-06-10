import pytest
import numpy as np
from typing import List, Dict, Any, Optional, Union
from unittest.mock import Mock, patch
from sentence_transformers import SentenceTransformer
from jet.vectors.document_types import HeaderDocument
from jet.models.tasks.hybrid_search_docs_with_bm25 import (
    process_documents,
    split_document,
    filter_by_headers,
    embed_chunk,
    embed_chunks_parallel,
    get_bm25_scores,
    get_original_document,
    search_docs,
    SearchResult,
)
import logging
import uuid
import os

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mock SentenceTransformer for tests


class MockSentenceTransformer:
    def __init__(self):
        self.device = "cpu"

    def encode(self, texts, convert_to_numpy=True, batch_size=32):
        if isinstance(texts, str):
            return np.array([1.0] * 384, dtype=np.float32)
        return np.array([[1.0] * 384 for _ in texts], dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return 384

# Mock CrossEncoder for tests


class MockCrossEncoder:
    def predict(self, pairs):
        return [0.9 - i * 0.1 for i in range(len(pairs))]


@pytest.fixture
def sample_documents():
    return [
        HeaderDocument(
            id="doc1",
            text="Header 1\nContent 1.",
            metadata={"header": "Header 1",
                      "header_level": 1, "content": "Content 1."}
        ),
        HeaderDocument(
            id="doc2",
            text="Header 2\nContent 2.",
            metadata={"header": "Header 2",
                      "header_level": 1, "content": "Content 2."}
        ),
    ]


@pytest.fixture
def sample_string_documents():
    return ["Header 1\nContent 1.", "Header 2\nContent 2."]


@pytest.fixture
def sample_dict_documents():
    return [
        {
            "id": "doc1",
            "text": "Header 1\nContent 1.",
            "metadata": {"header": "Header 1", "header_level": 1, "content": "Content 1."}
        },
        {
            "id": "doc2",
            "text": "Header 2\nContent 2.",
            "metadata": {"header": "Header 2", "header_level": 1, "content": "Content 2."}
        },
    ]


class TestProcessDocuments:
    def test_process_header_documents(self, sample_documents):
        expected = [
            {"text": "Header 1\nContent 1.", "id": "doc1", "index": 0},
            {"text": "Header 2\nContent 2.", "id": "doc2", "index": 1},
        ]
        result = process_documents(sample_documents)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_process_string_documents(self, sample_string_documents):
        expected = [
            {"text": "Header 1\nContent 1.", "id": "doc_0", "index": 0},
            {"text": "Header 2\nContent 2.", "id": "doc_1", "index": 1},
        ]
        result = process_documents(sample_string_documents)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_process_dict_documents(self, sample_dict_documents):
        expected = [
            {"text": "Header 1\nContent 1.", "id": "doc1", "index": 0},
            {"text": "Header 2\nContent 2.", "id": "doc2", "index": 1},
        ]
        result = process_documents(sample_dict_documents)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_process_with_ids(self, sample_string_documents):
        ids = ["custom1", "custom2"]
        expected = [
            {"text": "Header 1\nContent 1.", "id": "custom1", "index": 0},
            {"text": "Header 2\nContent 2.", "id": "custom2", "index": 1},
        ]
        result = process_documents(sample_string_documents, ids)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_process_none_input(self):
        with pytest.raises(ValueError) as exc_info:
            process_documents(None)
        logger.debug(f"Exception: {str(exc_info.value)}")
        assert str(exc_info.value) == "Documents input cannot be None"

    def test_process_non_list_input(self):
        with pytest.raises(ValueError) as exc_info:
            process_documents("not a list")
        logger.debug(f"Exception: {str(exc_info.value)}")
        assert str(exc_info.value) == "Input must be a list"

    def test_process_mismatched_ids(self, sample_documents):
        ids = ["id1"]
        with pytest.raises(ValueError) as exc_info:
            process_documents(sample_documents, ids)
        logger.debug(f"Exception: {str(exc_info.value)}")
        assert str(exc_info.value) == "IDs length must match documents length"


class TestSplitDocument:
    def test_split_document_basic(self):
        doc_text = "# Header 1\nThis is sentence one. This is sentence two."
        doc_id = "doc1"
        doc_index = 0
        expected = [
            {
                "text": "# Header 1\nThis is sentence one. This is sentence two.",
                "headers": ["# Header 1"],
                "doc_id": "doc1",
                "doc_index": 0,
            }
        ]
        result = split_document(
            doc_text, doc_id, doc_index, chunk_size=500, overlap=0)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_split_document_with_overlap(self):
        doc_text = "# Header 1\nSentence one. Sentence two. Sentence three."
        doc_id = "doc1"
        doc_index = 0
        expected = [
            {
                "text": "# Header 1\nSentence one. Sentence two.",
                "headers": ["# Header 1"],
                "doc_id": "doc1",
                "doc_index": 0,
            },
            {
                "text": "Sentence two.\nSentence three.",
                "headers": ["# Header 1"],
                "doc_id": "doc1",
                "doc_index": 0,
            },
        ]
        result = split_document(
            doc_text, doc_id, doc_index, chunk_size=6, overlap=1)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_split_document_empty(self):
        doc_text = ""
        doc_id = "doc1"
        doc_index = 0
        expected = []
        result = split_document(doc_text, doc_id, doc_index)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"


class TestFilterByHeaders:
    def test_filter_by_headers_relevant(self):
        chunks = [
            {"text": "Content 1", "headers": [
                "Header 1"], "doc_id": "doc1", "doc_index": 0},
            {"text": "Content 2", "headers": [
                "Header 2"], "doc_id": "doc2", "doc_index": 1},
        ]
        query = "Header 1"
        expected = [
            {"text": "Content 1", "headers": [
                "Header 1"], "doc_id": "doc1", "doc_index": 0}
        ]
        result = filter_by_headers(chunks, query)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_filter_by_headers_no_match(self):
        chunks = [
            {"text": "Content 1", "headers": [
                "Header 1"], "doc_id": "doc1", "doc_index": 0}
        ]
        query = "Nonexistent"
        expected = []
        result = filter_by_headers(chunks, query)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_filter_by_headers_empty_headers(self):
        chunks = [
            {"text": "Content 1", "headers": [], "doc_id": "doc1", "doc_index": 0}
        ]
        query = "Header"
        expected = []
        result = filter_by_headers(chunks, query)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"


class TestEmbedChunk:
    def test_embed_chunk(self):
        embedder = MockSentenceTransformer()
        chunk = "This is a test chunk."
        expected = np.array([1.0] * 384, dtype=np.float32)
        result = embed_chunk(chunk, embedder)
        logger.debug(f"Result shape: %s, dtype: %s",
                     result.shape, result.dtype)
        assert np.array_equal(
            result, expected), f"Expected {expected}, but got {result}"


class TestEmbedChunksParallel:
    def test_embed_chunks_parallel(self):
        embedder = MockSentenceTransformer()
        chunk_texts = ["Chunk 1", "Chunk 2"]
        expected = np.array([[1.0] * 384, [1.0] * 384], dtype=np.float32)
        result = embed_chunks_parallel(chunk_texts, embedder)
        logger.debug(f"Result shape: %s, dtype: %s",
                     result.shape, result.dtype)
        assert np.array_equal(
            result, expected), f"Expected {expected.shape}, but got {result.shape}"

    def test_embed_chunks_parallel_empty(self):
        embedder = MockSentenceTransformer()
        chunk_texts = []
        expected = np.zeros((0, 384), dtype=np.float32)
        result = embed_chunks_parallel(chunk_texts, embedder)
        logger.debug(f"Result shape: %s, dtype: %s",
                     result.shape, result.dtype)
        assert np.array_equal(
            result, expected), f"Expected shape {expected.shape}, but got {result.shape}"


class TestGetBM25Scores:
    def test_get_bm25_scores(self):
        chunk_texts = ["This is a test document", "Another document"]
        query = "test document"
        result = get_bm25_scores(chunk_texts, query)
        expected = [isinstance(score, float) and not np.isnan(score)
                    for score in result]
        logger.debug(f"Result: {result}")
        assert all(expected), f"Expected valid float scores, but got {result}"


class TestGetOriginalDocument:
    def test_get_original_document_header(self, sample_documents):
        doc_id = "doc1"
        doc_index = 0
        expected = sample_documents[0]
        result = get_original_document(doc_id, doc_index, sample_documents)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_get_original_document_string(self, sample_string_documents):
        doc_id = "doc_0"
        doc_index = 0
        expected = HeaderDocument(
            id="doc_0", text="Header 1\nContent 1.", metadata={"original_index": 0})
        result = get_original_document(
            doc_id, doc_index, sample_string_documents)
        logger.debug(f"Result: {result}")
        assert result.id == expected.id and result.text == expected.text, f"Expected {expected}, but got {result}"

    def test_get_original_document_not_found(self, sample_documents):
        doc_id = "nonexistent"
        doc_index = 999
        expected = None
        result = get_original_document(doc_id, doc_index, sample_documents)
        logger.debug(f"Result: {result}")
        assert result == expected, f"Expected {expected}, but got {result}"


class TestSearchDocs:
    @patch("jet.models.tasks.hybrid_search_docs.SentenceTransformer", MockSentenceTransformer)
    @patch("jet.models.tasks.hybrid_search_docs.CrossEncoder", MockCrossEncoder)
    def test_search_docs(self, sample_documents):
        query = "Header 1"
        expected = [
            {
                "id": "doc1",
                "doc_index": 0,
                "rank": 1,
                "score": 0.9,
                "combined_score": pytest.approx(0.9, 0.1),
                "embedding_score": pytest.approx(1.0, 0.1),
                "headers": ["Header 1"],
                "text": "Header 1\nContent 1.",
                "document": sample_documents[0],
            }
        ]
        result = search_docs(query, sample_documents, top_k=1, rerank_top_k=1)
        logger.debug(f"Result: {result}")
        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
        assert result[0]["id"] == expected[0][
            "id"], f"Expected id {expected[0]['id']}, got {result[0]['id']}"
        assert result[0]["rank"] == expected[0][
            "rank"], f"Expected rank {expected[0]['rank']}, got {result[0]['rank']}"
        assert result[0]["text"] == expected[0][
            "text"], f"Expected text {expected[0]['text']}, got {result[0]['text']}"

    @patch("jet.models.tasks.hybrid_search_docs.SentenceTransformer", MockSentenceTransformer)
    @patch("jet.models.tasks.hybrid_search_docs.CrossEncoder", MockCrossEncoder)
    def test_search_docs_with_instruction(self, sample_documents):
        query = "Header 1"
        instruction = "Search for"
        expected = [
            {
                "id": "doc1",
                "doc_index": 0,
                "rank": 1,
                "score": 0.9,
                "combined_score": pytest.approx(0.9, 0.1),
                "embedding_score": pytest.approx(1.0, 0.1),
                "headers": ["Header 1"],
                "text": "Header 1\nContent 1.",
                "document": sample_documents[0],
            }
        ]
        result = search_docs(query, sample_documents,
                             instruction=instruction, top_k=1, rerank_top_k=1)
        logger.debug(f"Result: {result}")
        assert len(result) == 1, f"Expected 1 result, got {len(result)}"
        assert result[0]["id"] == expected[0][
            "id"], f"Expected id {expected[0]['id']}, got {result[0]['id']}"
        assert result[0]["rank"] == expected[0][
            "rank"], f"Expected rank {expected[0]['rank']}, got {result[0]['rank']}"
        assert result[0]["text"] == expected[0][
            "text"], f"Expected text {expected[0]['text']}, got {result[0]['text']}"


if __name__ == "__main__":
    pytest.main(["-v"])
