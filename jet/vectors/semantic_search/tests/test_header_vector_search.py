import pytest
import re
import numpy as np
from unittest.mock import patch, Mock
from jet.vectors.semantic_search.header_vector_search import (
    cosine_similarity,
    collect_header_chunks,
    compute_weighted_similarity,
    merge_results,
    search_headers,
    HeaderSearchResult,
    DEFAULT_EMBED_MODEL,
    MAX_CONTENT_SIZE,
    preprocess_text
)
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc, MarkdownToken
import logging


@pytest.fixture
def mock_sentence_transformer():
    """Fixture to mock SentenceTransformerRegistry and its model."""
    with patch('jet.vectors.semantic_search.header_vector_search.SentenceTransformerRegistry') as mock_registry:
        mock_model = Mock()
        mock_model.encode.side_effect = lambda x, **kwargs: np.array(
            [0.1, 0.2, 0.3]) if isinstance(x, str) else np.array([[0.1, 0.2, 0.3]] * len(x))
        mock_registry.load_model.return_value = mock_model
        yield mock_model


@pytest.fixture
def sample_header_doc():
    """Fixture providing a sample HeaderDoc for testing."""
    return {
        "doc_index": 0,
        "doc_id": "doc1",
        "header": "Test Header",
        "content": "This is a test content",
        "level": 2,
        "parent_headers": ["Root Header", "Parent Header"],
        "parent_header": "Parent Header",
        "parent_level": 1,
        "tokens": []
    }


def test_cosine_similarity():
    """Test cosine similarity calculation for various vector pairs."""
    # Given: Two orthogonal vectors
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])
    expected = 0.0
    # When: Compute cosine similarity
    result = cosine_similarity(vec1, vec2)
    # Then: Similarity should be approximately 0
    assert abs(result - expected) < 1e-10

    # Given: Two identical vectors
    vec1 = np.array([1.0, 1.0])
    vec2 = np.array([1.0, 1.0])
    expected = 1.0
    # When: Compute cosine similarity
    result = cosine_similarity(vec1, vec2)
    # Then: Similarity should be 1
    assert abs(result - expected) < 1e-10

    # Given: Zero vector and non-zero vector
    vec1 = np.array([0.0, 0.0])
    vec2 = np.array([1.0, 1.0])
    expected = 0.0
    # When: Compute cosine similarity
    result = cosine_similarity(vec1, vec2)
    # Then: Similarity should be 0
    assert result == expected


def test_collect_header_chunks(sample_header_doc):
    """Test collect_header_chunks function with default and custom tokenizers."""
    # Given: A sample header document and a custom tokenizer
    def custom_tokenizer(text): return len(text.split())
    original_content = "This is a test content"
    original_header = "Test Header"
    original_headers_context = "Test Header\nRoot Header\nParent Header"
    expected_content = preprocess_text(original_content)
    expected_header = preprocess_text(original_header)
    expected_headers_context = preprocess_text(original_headers_context)
    expected_num_tokens_default = len(
        re.findall(r'\b\w+\b|[^\w\s]', original_content))
    expected_num_tokens_custom = len(original_content.split())
    expected_chunk = (0, expected_header, expected_content,
                      original_content, 0, 22, expected_num_tokens_default)
    expected_chunk_custom = (0, expected_header, expected_content,
                             original_content, 0, 22, expected_num_tokens_custom)

    # When: Collect chunks with default tokenizer
    result_doc_indices, result_headers, result_headers_context, result_chunks = collect_header_chunks([
                                                                                                      sample_header_doc])

    # Then: Verify the output
    assert len(result_doc_indices) == 1
    assert result_doc_indices[0] == 0
    assert result_headers[0] == expected_header
    assert result_headers_context[0] == expected_headers_context
    assert len(result_chunks) == 1
    assert result_chunks[0] == expected_chunk

    # When: Collect chunks with custom tokenizer
    result_doc_indices, result_headers, result_headers_context, result_chunks = collect_header_chunks(
        [sample_header_doc], tokenizer=custom_tokenizer)

    # Then: Verify the token count with custom tokenizer
    assert result_chunks[0][6] == expected_chunk_custom[6]


def test_compute_weighted_similarity_with_content():
    """Test compute_weighted_similarity with content vector provided."""
    # Given: Sample vectors and content tokens >= 100
    query_vec = np.array([1.0, 0.0, 0.0])
    header_vec = np.array([1.0, 0.0, 0.0])
    parent_vec = np.array([0.0, 1.0, 0.0])
    content_vec = np.array([0.0, 0.0, 1.0])
    content_tokens = 100
    header_level = 2
    expected_header_content_sim = 0.0  # Orthogonal vectors
    expected_headers_sim = 0.0  # Orthogonal vectors
    expected_content_sim = 0.0  # Orthogonal vectors
    expected_weighted_sim = (0.3 / 1.0) * expected_header_content_sim + (
        0.3 / 1.0) * expected_headers_sim + (0.4 / 1.0) * expected_content_sim

    # When: Compute weighted similarity
    result_weighted_sim, result_header_content_sim, result_headers_sim, result_content_sim = compute_weighted_similarity(
        query_vec, header_vec, parent_vec, content_vec, content_tokens, header_level
    )

    # Then: Verify similarity scores
    assert abs(result_header_content_sim - expected_header_content_sim) < 1e-10
    assert abs(result_headers_sim - expected_headers_sim) < 1e-10
    assert abs(result_content_sim - expected_content_sim) < 1e-10
    assert abs(result_weighted_sim - expected_weighted_sim) < 1e-10


def test_compute_weighted_similarity_no_content():
    """Test compute_weighted_similarity without content vector."""
    # Given: Sample vectors, no content vector, and content tokens < 100
    query_vec = np.array([1.0, 0.0, 0.0])
    header_vec = np.array([1.0, 0.0, 0.0])
    parent_vec = np.array([0.0, 1.0, 0.0])
    content_vec = None
    content_tokens = 10
    header_level = 2
    expected_header_content_sim = 0.0
    expected_headers_sim = 0.0
    expected_content_sim = 0.0
    expected_weighted_sim = (0.3 / 1.0) * expected_header_content_sim + (
        0.3 / 1.0) * expected_headers_sim + (0.4 / 1.0) * expected_content_sim

    # When: Compute weighted similarity
    result_weighted_sim, result_header_content_sim, result_headers_sim, result_content_sim = compute_weighted_similarity(
        query_vec, header_vec, parent_vec, content_vec, content_tokens, header_level
    )

    # Then: Verify similarity scores
    assert abs(result_header_content_sim - expected_header_content_sim) < 1e-10
    assert abs(result_headers_sim - expected_headers_sim) < 1e-10
    assert abs(result_content_sim - expected_content_sim) < 1e-10
    assert abs(result_weighted_sim - expected_weighted_sim) < 1e-10


def test_search_headers(mock_sentence_transformer, sample_header_doc):
    """Test search_headers function with a single document."""
    # Given: A sample header document, query, and custom tokenizer
    query = "Test Query"
    original_content = "This is a test content"
    original_header = "Test Header"
    original_parent = "Parent Header"
    expected_doc_index = 0
    top_k = 1
    def custom_tokenizer(text): return len(text.split())
    expected_num_tokens_default = len(
        re.findall(r'\b\w+\b|[^\w\s]', original_content))
    expected_num_tokens_custom = len(original_content.split())

    # When: Search headers with default tokenizer
    result = list(search_headers([sample_header_doc], query, top_k=top_k))

    # Then: Verify the search results
    assert len(result) == 1
    assert result[0]['rank'] == 1
    assert isinstance(result[0]['score'], float)
    assert result[0]['content'] == original_content
    assert result[0]['metadata']['doc_index'] == expected_doc_index
    assert result[0]['metadata']['header'] == original_header
    assert result[0]['metadata']['parent_header'] == original_parent
    assert result[0]['metadata']['chunk_idx'] == 0
    assert isinstance(result[0]['metadata']
                      ['header_content_similarity'], float)
    assert isinstance(result[0]['metadata']['headers_similarity'], float)
    assert isinstance(result[0]['metadata']['content_similarity'], float)
    assert result[0]['metadata']['num_tokens'] == expected_num_tokens_default

    # When: Search headers with custom tokenizer
    result = list(search_headers(
        [sample_header_doc], query, top_k=top_k, tokenizer=custom_tokenizer))

    # Then: Verify token count with custom tokenizer
    assert result[0]['metadata']['num_tokens'] == expected_num_tokens_custom

    # When: Search headers with split_chunks=True
    result = list(search_headers(
        [sample_header_doc], query, top_k=top_k, split_chunks=True))

    # Then: Verify split chunks
    assert len(result) == 1
    assert result[0]['content'] == original_content
    assert result[0]['metadata']['num_tokens'] == expected_num_tokens_default


def test_search_headers_no_results(mock_sentence_transformer):
    """Test search_headers with empty input."""
    # Given: An empty list of header documents
    header_docs = []
    query = "test query"
    expected = []

    # When: Search headers
    result = list(search_headers(header_docs, query))

    # Then: Verify no results are returned
    assert result == expected


def test_search_headers_chunking(mock_sentence_transformer):
    """Test search_headers with chunking and merging."""
    # Given: A header document with long content and a custom tokenizer
    header_doc = {
        "doc_index": 0,
        "doc_id": "doc1",
        "header": "Test Header",
        "content": "Word " * 200,
        "level": 2,
        "parent_headers": ["Root Header", "Parent Header"],
        "parent_header": "Parent Header",
        "parent_level": 1,
        "tokens": []
    }
    original_content = "Word " * 200
    def custom_tokenizer(text): return len(text.split())
    chunk_size = 200
    chunk_overlap = 50
    expected_tokens_merged_default = len(
        re.findall(r'\b\w+\b|[^\w\s]', original_content))
    expected_tokens_merged_custom = len(original_content.split())

    # When: Search headers with default tokenizer and merging
    result_default = list(search_headers(
        [header_doc], "test query", chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    # Then: Verify merged results
    assert len(result_default) == 1
    assert result_default[0]['content'] == original_content
    assert result_default[0]['metadata']['num_tokens'] == expected_tokens_merged_default
    assert result_default[0]['metadata']['chunk_idx'] == 0
    assert result_default[0]['metadata']['header'] == "Test Header"
    assert result_default[0]['metadata']['parent_header'] == "Parent Header"

    # When: Search headers with custom tokenizer
    result_custom = list(search_headers(
        [header_doc], "test query", chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=custom_tokenizer))

    # Then: Verify token count with custom tokenizer
    assert result_custom[0]['metadata']['num_tokens'] == expected_tokens_merged_custom

    # When: Search headers with split_chunks=True
    result_split = list(search_headers(
        [header_doc], "test query", chunk_size=chunk_size, chunk_overlap=chunk_overlap, split_chunks=True))

    # Then: Verify split chunks
    assert len(result_split) > 1
    assert all(r['metadata']['end_idx'] - r['metadata']
               ['start_idx'] <= chunk_size for r in result_split)
    assert [r['metadata']['chunk_idx']
            for r in result_split] == list(range(len(result_split)))
    assert all(r['content'].strip() in original_content for r in result_split)
    assert all(r['metadata']['header'] == "Test Header" for r in result_split)
    assert all(r['metadata']['parent_header'] ==
               "Parent Header" for r in result_split)


class TestMergeResults:
    """Test suite for merge_results function."""

    def test_merge_adjacent_chunks(self):
        """Test merging of adjacent chunks from the same document."""
        # Given: Two adjacent chunks from the same document
        results = [
            {
                "rank": 1,
                "score": 0.8,
                "content": "First chunk ",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "Test Header",
                    "level": 2,
                    "parent_header": "Parent Header",
                    "parent_level": 1,
                    "start_idx": 0,
                    "end_idx": 12,
                    "chunk_idx": 0,
                    "header_content_similarity": 0.7,
                    "headers_similarity": 0.6,
                    "content_similarity": 0.9,
                    "num_tokens": 2
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "content": "Second chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "Test Header",
                    "level": 2,
                    "parent_header": "Parent Header",
                    "parent_level": 1,
                    "start_idx": 12,
                    "end_idx": 23,
                    "chunk_idx": 1,
                    "header_content_similarity": 0.7,
                    "headers_similarity": 0.6,
                    "content_similarity": 0.8,
                    "num_tokens": 2
                }
            }
        ]
        expected = [
            {
                "rank": 1,
                "score": 0.75,
                "content": "First chunk Second chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "Test Header",
                    "level": 2,
                    "parent_header": "Parent Header",
                    "parent_level": 1,
                    "start_idx": 0,
                    "end_idx": 23,
                    "chunk_idx": 0,
                    "header_content_similarity": 0.7,
                    "headers_similarity": 0.6,
                    "content_similarity": 0.85,
                    "num_tokens": 4
                }
            }
        ]

        # When: Merge results
        result = merge_results(results)

        # Then: Verify merged results
        assert len(result) == 1
        assert result[0]["content"] == expected[0]["content"]
        assert result[0]["metadata"]["start_idx"] == expected[0]["metadata"]["start_idx"]
        assert result[0]["metadata"]["end_idx"] == expected[0]["metadata"]["end_idx"]
        assert result[0]["metadata"]["chunk_idx"] == expected[0]["metadata"]["chunk_idx"]
        assert abs(result[0]["score"] - expected[0]["score"]) < 1e-10
        assert abs(result[0]["metadata"]["content_similarity"] -
                   expected[0]["metadata"]["content_similarity"]) < 1e-10
        assert result[0]["metadata"]["header"] == expected[0]["metadata"]["header"]
        assert result[0]["metadata"]["parent_header"] == expected[0]["metadata"]["parent_header"]

    def test_non_adjacent_chunks(self):
        """Test handling of non-adjacent chunks from the same document."""
        # Given: Two non-adjacent chunks from the same document
        results = [
            {
                "rank": 1,
                "score": 0.8,
                "content": "First chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "Test Header",
                    "level": 2,
                    "parent_header": "Parent Header",
                    "parent_level": 1,
                    "start_idx": 0,
                    "end_idx": 11,
                    "chunk_idx": 0,
                    "header_content_similarity": 0.7,
                    "headers_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "content": "Second chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "Test Header",
                    "level": 2,
                    "parent_header": "Parent Header",
                    "parent_level": 1,
                    "start_idx": 20,
                    "end_idx": 31,
                    "chunk_idx": 1,
                    "header_content_similarity": 0.7,
                    "headers_similarity": 0.6,
                    "content_similarity": 0.8
                }
            }
        ]
        expected = [
            {
                "rank": 1,
                "score": 0.8,
                "content": "First chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "Test Header",
                    "level": 2,
                    "parent_header": "Parent Header",
                    "parent_level": 1,
                    "start_idx": 0,
                    "end_idx": 11,
                    "chunk_idx": 0,
                    "header_content_similarity": 0.7,
                    "headers_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "content": "Second chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "Test Header",
                    "level": 2,
                    "parent_header": "Parent Header",
                    "parent_level": 1,
                    "start_idx": 20,
                    "end_idx": 31,
                    "chunk_idx": 0,
                    "header_content_similarity": 0.7,
                    "headers_similarity": 0.6,
                    "content_similarity": 0.8
                }
            }
        ]

        # When: Merge results
        result = merge_results(results)

        # Then: Verify non-merged results
        assert len(result) == 2
        for i in range(2):
            assert result[i]["content"] == expected[i]["content"]
            assert result[i]["metadata"]["start_idx"] == expected[i]["metadata"]["start_idx"]
            assert result[i]["metadata"]["end_idx"] == expected[i]["metadata"]["end_idx"]
            assert result[i]["metadata"]["chunk_idx"] == expected[i]["metadata"]["chunk_idx"]
            assert result[i]["score"] == expected[i]["score"]
            assert result[i]["metadata"]["header"] == expected[i]["metadata"]["header"]
            assert result[i]["metadata"]["parent_header"] == expected[i]["metadata"]["parent_header"]

    def test_multiple_headers(self):
        """Test merging results from different headers."""
        # Given: Results from two different headers
        results = [
            {
                "rank": 1,
                "score": 0.8,
                "content": "Header1 content",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "Header1",
                    "level": 2,
                    "parent_header": "Parent1",
                    "parent_level": 1,
                    "start_idx": 0,
                    "end_idx": 14,
                    "chunk_idx": 0,
                    "header_content_similarity": 0.7,
                    "headers_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "content": "Header2 content",
                "metadata": {
                    "doc_index": 1,
                    "doc_id": "doc2",
                    "header": "Header2",
                    "level": 2,
                    "parent_header": "Parent2",
                    "parent_level": 1,
                    "start_idx": 0,
                    "end_idx": 14,
                    "chunk_idx": 0,
                    "header_content_similarity": 0.6,
                    "headers_similarity": 0.5,
                    "content_similarity": 0.8
                }
            }
        ]
        expected = [
            {
                "rank": 1,
                "score": 0.8,
                "content": "Header1 content",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "Header1",
                    "level": 2,
                    "parent_header": "Parent1",
                    "parent_level": 1,
                    "start_idx": 0,
                    "end_idx": 14,
                    "chunk_idx": 0,
                    "header_content_similarity": 0.7,
                    "headers_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "content": "Header2 content",
                "metadata": {
                    "doc_index": 1,
                    "doc_id": "doc2",
                    "header": "Header2",
                    "level": 2,
                    "parent_header": "Parent2",
                    "parent_level": 1,
                    "start_idx": 0,
                    "end_idx": 14,
                    "chunk_idx": 0,
                    "header_content_similarity": 0.6,
                    "headers_similarity": 0.5,
                    "content_similarity": 0.8
                }
            }
        ]

        # When: Merge results
        result = merge_results(results)

        # Then: Verify results for different headers
        assert len(result) == 2
        for i in range(2):
            assert result[i]["content"] == expected[i]["content"]
            assert result[i]["metadata"]["doc_index"] == expected[i]["metadata"]["doc_index"]
            assert result[i]["score"] == expected[i]["score"]
            assert result[i]["metadata"]["header"] == expected[i]["metadata"]["header"]
            assert result[i]["metadata"]["parent_header"] == expected[i]["metadata"]["parent_header"]

    def test_empty_results(self):
        """Test merge_results with empty input."""
        # Given: An empty list of results
        results = []
        expected = []

        # When: Merge results
        result = merge_results(results)

        # Then: Verify empty output
        assert result == expected
