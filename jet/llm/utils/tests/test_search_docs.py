import pytest
import torch
from typing import List
from jet.llm.utils.search_docs import search_docs_with_rerank


@pytest.fixture
def sample_data():
    query = "machine learning"
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning models require large datasets.",
        "Python is a popular programming language.",
        "Neural networks are used in machine learning."
    ]
    ids = ["doc1", "doc2", "doc3", "doc4"]
    return query, documents, ids


def test_search_docs_with_rerank_basic(sample_data):
    query, documents, ids = sample_data
    expected = [
        {"id": "doc1", "rank": 1, "doc_index": 0, "score": pytest.approx(
            0.9, abs=0.1), "text": documents[0], "tokens": 12},
        {"id": "doc4", "rank": 2, "doc_index": 3, "score": pytest.approx(
            0.8, abs=0.1), "text": documents[3], "tokens": 10}
    ]
    result = search_docs_with_rerank(
        query=query,
        documents=documents,
        model="all-minilm:33m",
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        top_k=2,
        batch_size=2,
        ids=ids
    )
    assert len(result) == 2
    assert result[0]["id"] == expected[0]["id"]
    assert result[0]["rank"] == expected[0]["rank"]
    assert result[0]["doc_index"] == expected[0]["doc_index"]
    assert result[0]["score"] == pytest.approx(expected[0]["score"], abs=0.1)
    assert result[0]["text"] == expected[0]["text"]
    assert result[0]["tokens"] >= expected[0]["tokens"]
    assert result[1]["id"] == expected[1]["id"]
    assert result[1]["rank"] == expected[1]["rank"]
    assert result[1]["doc_index"] == expected[1]["doc_index"]
    assert result[1]["score"] == pytest.approx(expected[1]["score"], abs=0.1)
    assert result[1]["text"] == expected[1]["text"]
    assert result[1]["tokens"] >= expected[1]["tokens"]


def test_search_docs_with_rerank_empty_input():
    expected = []
    result = search_docs_with_rerank(
        query="",
        documents=[],
        model="all-minilm:33m",
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        top_k=2
    )
    assert result == expected


def test_search_docs_with_rerank_invalid_ids(sample_data):
    query, documents, _ = sample_data
    invalid_ids = ["doc1", "doc2"]  # Length mismatch
    with pytest.raises(ValueError):
        search_docs_with_rerank(
            query=query,
            documents=documents,
            model="all-minilm:33m",
            rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
            top_k=2,
            ids=invalid_ids
        )


def test_search_docs_with_rerank_memory_cleanup(sample_data):
    query, documents, ids = sample_data
    result = search_docs_with_rerank(
        query=query,
        documents=documents,
        model="all-minilm:33m",
        rerank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
        top_k=2,
        batch_size=2,
        ids=ids
    )
    assert len(result) <= 2
    if torch.backends.mps.is_available():
        assert torch.mps.current_allocated_memory() < 1024 * 1024 * \
            100  # Less than 100MB
