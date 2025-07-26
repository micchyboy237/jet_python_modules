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
    MAX_CONTENT_SIZE
)
from jet.code.markdown_types.markdown_parsed_types import HeaderDoc, MarkdownToken
import logging


@pytest.fixture
def mock_sentence_transformer():
    with patch('jet.vectors.semantic_search.header_vector_search.SentenceTransformerRegistry') as mock_registry:
        mock_model = Mock()
        mock_model.encode.side_effect = lambda x, **kwargs: np.array(
            [0.1, 0.2, 0.3]) if isinstance(x, str) else np.array([[0.1, 0.2, 0.3]] * len(x))
        mock_registry.load_model.return_value = mock_model
        yield mock_model


@pytest.fixture
def sample_header_doc():
    return {
        "doc_index": 0,
        "doc_id": "doc1",
        "header": "Test Header",
        "content": "This is a test content",
        "level": 1,
        "parent_header": "Parent Header",
        "parent_level": 0,
        "tokens": []
    }


def test_cosine_similarity():
    vec1 = np.array([1.0, 0.0])
    vec2 = np.array([0.0, 1.0])
    sim = cosine_similarity(vec1, vec2)
    assert abs(sim) < 1e-10
    vec1 = np.array([1.0, 1.0])
    vec2 = np.array([1.0, 1.0])
    sim = cosine_similarity(vec1, vec2)
    assert abs(sim - 1.0) < 1e-10
    vec1 = np.array([0.0, 0.0])
    vec2 = np.array([1.0, 1.0])
    sim = cosine_similarity(vec1, vec2)
    assert sim == 0.0


def test_collect_header_chunks(sample_header_doc):
    def custom_tokenizer(text): return len(text.split())
    expected_content = "this is a test content"
    expected_doc_index = 0
    expected_header = "test header"
    expected_parent = "parent header"
    expected_chunk = (0, expected_header, expected_content, 0, 22, 5)
    expected_chunk_custom = (0, expected_header, expected_content, 0, 22, 5)

    doc_indices, headers, parent_headers, chunks = collect_header_chunks([
                                                                         sample_header_doc])
    assert len(doc_indices) == 1
    assert doc_indices[0] == expected_doc_index
    assert headers[0] == expected_header
    assert parent_headers[0] == expected_parent
    assert len(chunks) == 1
    assert chunks[0] == expected_chunk

    doc_indices, headers, parent_headers, chunks = collect_header_chunks(
        [sample_header_doc], tokenizer=custom_tokenizer)
    assert chunks[0][5] == expected_chunk_custom[5]


def test_compute_weighted_similarity_with_content():
    query_vec = np.array([1.0, 0.0, 0.0])
    header_vec = np.array([1.0, 0.0, 0.0])
    parent_vec = np.array([0.0, 1.0, 0.0])
    content_vec = np.array([0.0, 0.0, 1.0])
    weighted_sim, header_sim, parent_sim, content_sim = compute_weighted_similarity(
        query_vec, header_vec, parent_vec, content_vec
    )
    assert abs(header_sim - 1.0) < 1e-10
    assert abs(parent_sim) < 1e-10
    assert abs(content_sim) < 1e-10
    assert abs(weighted_sim - (0.4 * 1.0 + 0.2 * 0.0 + 0.4 * 0.0)) < 1e-10


def test_compute_weighted_similarity_no_content():
    query_vec = np.array([1.0, 0.0, 0.0])
    header_vec = np.array([1.0, 0.0, 0.0])
    parent_vec = np.array([0.0, 1.0, 0.0])
    content_vec = None
    weighted_sim, header_sim, parent_sim, content_sim = compute_weighted_similarity(
        query_vec, header_vec, parent_vec, content_vec
    )
    assert abs(header_sim - 1.0) < 1e-10
    assert abs(parent_sim) < 1e-10
    assert abs(content_sim) < 1e-10
    assert abs(weighted_sim - (0.4 * 1.0 + 0.2 * 0.0)) < 1e-10


def test_search_headers(mock_sentence_transformer, sample_header_doc):
    query = "test query"
    expected_content = "this is a test content"
    expected_doc_index = 0
    top_k = 1
    def custom_tokenizer(text): return len(text.split())
    expected_num_tokens_default = 5
    expected_num_tokens_custom = 5

    results = list(search_headers([sample_header_doc], query, top_k=top_k))
    assert len(results) == 1
    assert results[0]['rank'] == 1
    assert isinstance(results[0]['score'], float)
    assert results[0]['content'] == expected_content
    assert results[0]['metadata']['doc_index'] == expected_doc_index
    assert results[0]['metadata']['chunk_idx'] == 0
    assert isinstance(results[0]['metadata']['header_similarity'], float)
    assert isinstance(results[0]['metadata']['parent_similarity'], float)
    assert isinstance(results[0]['metadata']['content_similarity'], float)
    assert results[0]['metadata']['num_tokens'] == expected_num_tokens_default

    results = list(search_headers(
        [sample_header_doc], query, top_k=top_k, tokenizer=custom_tokenizer))
    assert results[0]['metadata']['num_tokens'] == expected_num_tokens_custom

    results = list(search_headers(
        [sample_header_doc], query, top_k=top_k, split_chunks=True))
    assert len(results) == 1
    assert results[0]['content'] == expected_content
    assert results[0]['metadata']['num_tokens'] == expected_num_tokens_default


def test_search_headers_no_results(mock_sentence_transformer):
    results = list(search_headers([], "test query"))
    assert results == []


def test_search_headers_chunking(mock_sentence_transformer):
    header_doc = {
        "doc_index": 0,
        "doc_id": "doc1",
        "header": "Test Header",
        "content": "word " * 200,
        "level": 1,
        "parent_header": "Parent Header",
        "parent_level": 0,
        "tokens": []
    }
    content = "word " * 200
    def custom_tokenizer(text): return len(text.split())
    chunk_size = 200
    chunk_overlap = 50

    results_default = list(search_headers(
        [header_doc], "test query", chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    assert len(results_default) == 1
    assert results_default[0]['content'] == content.lower()
    expected_tokens_merged = len(re.findall(
        r'\b\w+\b|[^\w\s]', content.lower()))
    assert results_default[0]['metadata']['num_tokens'] == expected_tokens_merged
    assert results_default[0]['metadata']['chunk_idx'] == 0

    results_custom = list(search_headers(
        [header_doc], "test query", chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=custom_tokenizer))
    expected_tokens_merged_custom = len(content.lower().split())
    assert results_custom[0]['metadata']['num_tokens'] == expected_tokens_merged_custom

    results_split = list(search_headers(
        [header_doc], "test query", chunk_size=chunk_size, chunk_overlap=chunk_overlap, split_chunks=True))
    assert len(results_split) > 1
    assert all(r['metadata']['end_idx'] - r['metadata']
               ['start_idx'] <= chunk_size for r in results_split)
    assert [r['metadata']['chunk_idx']
            for r in results_split] == list(range(len(results_split)))


class TestMergeResults:
    def test_merge_adjacent_chunks(self):
        results = [
            {
                "rank": 1,
                "score": 0.8,
                "content": "first chunk ",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "test header",
                    "level": 1,
                    "parent_header": "parent header",
                    "parent_level": 0,
                    "start_idx": 0,
                    "end_idx": 12,
                    "chunk_idx": 0,
                    "header_similarity": 0.7,
                    "parent_similarity": 0.6,
                    "content_similarity": 0.9,
                    "num_tokens": 2
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "content": "second chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "test header",
                    "level": 1,
                    "parent_header": "parent header",
                    "parent_level": 0,
                    "start_idx": 12,
                    "end_idx": 23,
                    "chunk_idx": 1,
                    "header_similarity": 0.7,
                    "parent_similarity": 0.6,
                    "content_similarity": 0.8,
                    "num_tokens": 2
                }
            }
        ]
        expected = [
            {
                "rank": 1,
                "score": 0.75,
                "content": "first chunk second chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "test header",
                    "level": 1,
                    "parent_header": "parent header",
                    "parent_level": 0,
                    "start_idx": 0,
                    "end_idx": 23,
                    "chunk_idx": 0,
                    "header_similarity": 0.7,
                    "parent_similarity": 0.6,
                    "content_similarity": 0.85,
                    "num_tokens": 4
                }
            }
        ]
        merged = merge_results(results)
        assert len(merged) == 1
        assert merged[0]["content"] == expected[0]["content"]
        assert merged[0]["metadata"]["start_idx"] == expected[0]["metadata"]["start_idx"]
        assert merged[0]["metadata"]["end_idx"] == expected[0]["metadata"]["end_idx"]
        assert merged[0]["metadata"]["chunk_idx"] == expected[0]["metadata"]["chunk_idx"]
        assert abs(merged[0]["score"] - expected[0]["score"]) < 1e-10
        assert abs(merged[0]["metadata"]["content_similarity"] -
                   expected[0]["metadata"]["content_similarity"]) < 1e-10

    def test_non_adjacent_chunks(self):
        results = [
            {
                "rank": 1,
                "score": 0.8,
                "content": "first chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "test header",
                    "level": 1,
                    "parent_header": "parent header",
                    "parent_level": 0,
                    "start_idx": 0,
                    "end_idx": 11,
                    "chunk_idx": 0,
                    "header_similarity": 0.7,
                    "parent_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "content": "second chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "test header",
                    "level": 1,
                    "parent_header": "parent header",
                    "parent_level": 0,
                    "start_idx": 20,
                    "end_idx": 31,
                    "chunk_idx": 1,
                    "header_similarity": 0.7,
                    "parent_similarity": 0.6,
                    "content_similarity": 0.8
                }
            }
        ]
        expected = [
            {
                "rank": 1,
                "score": 0.8,
                "content": "first chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "test header",
                    "level": 1,
                    "parent_header": "parent header",
                    "parent_level": 0,
                    "start_idx": 0,
                    "end_idx": 11,
                    "chunk_idx": 0,
                    "header_similarity": 0.7,
                    "parent_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "content": "second chunk",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "test header",
                    "level": 1,
                    "parent_header": "parent header",
                    "parent_level": 0,
                    "start_idx": 20,
                    "end_idx": 31,
                    "chunk_idx": 0,
                    "header_similarity": 0.7,
                    "parent_similarity": 0.6,
                    "content_similarity": 0.8
                }
            }
        ]
        merged = merge_results(results)
        assert len(merged) == 2
        for i in range(2):
            assert merged[i]["content"] == expected[i]["content"]
            assert merged[i]["metadata"]["start_idx"] == expected[i]["metadata"]["start_idx"]
            assert merged[i]["metadata"]["end_idx"] == expected[i]["metadata"]["end_idx"]
            assert merged[i]["metadata"]["chunk_idx"] == expected[i]["metadata"]["chunk_idx"]
            assert merged[i]["score"] == expected[i]["score"]

    def test_multiple_headers(self):
        results = [
            {
                "rank": 1,
                "score": 0.8,
                "content": "header1 content",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "header1",
                    "level": 1,
                    "parent_header": "parent1",
                    "parent_level": 0,
                    "start_idx": 0,
                    "end_idx": 14,
                    "chunk_idx": 0,
                    "header_similarity": 0.7,
                    "parent_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "content": "header2 content",
                "metadata": {
                    "doc_index": 1,
                    "doc_id": "doc2",
                    "header": "header2",
                    "level": 1,
                    "parent_header": "parent2",
                    "parent_level": 0,
                    "start_idx": 0,
                    "end_idx": 14,
                    "chunk_idx": 0,
                    "header_similarity": 0.6,
                    "parent_similarity": 0.5,
                    "content_similarity": 0.8
                }
            }
        ]
        expected = [
            {
                "rank": 1,
                "score": 0.8,
                "content": "header1 content",
                "metadata": {
                    "doc_index": 0,
                    "doc_id": "doc1",
                    "header": "header1",
                    "level": 1,
                    "parent_header": "parent1",
                    "parent_level": 0,
                    "start_idx": 0,
                    "end_idx": 14,
                    "chunk_idx": 0,
                    "header_similarity": 0.7,
                    "parent_similarity": 0.6,
                    "content_similarity": 0.9
                }
            },
            {
                "rank": 2,
                "score": 0.7,
                "content": "header2 content",
                "metadata": {
                    "doc_index": 1,
                    "doc_id": "doc2",
                    "header": "header2",
                    "level": 1,
                    "parent_header": "parent2",
                    "parent_level": 0,
                    "start_idx": 0,
                    "end_idx": 14,
                    "chunk_idx": 0,
                    "header_similarity": 0.6,
                    "parent_similarity": 0.5,
                    "content_similarity": 0.8
                }
            }
        ]
        merged = merge_results(results)
        assert len(merged) == 2
        for i in range(2):
            assert merged[i]["content"] == expected[i]["content"]
            assert merged[i]["metadata"]["doc_index"] == expected[i]["metadata"]["doc_index"]
            assert merged[i]["score"] == expected[i]["score"]

    def test_empty_results(self):
        results = []
        expected = []
        merged = merge_results(results)
        assert merged == expected
