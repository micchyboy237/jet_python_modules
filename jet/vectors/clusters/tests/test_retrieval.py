import os
import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple
from jet.vectors.clusters.retrieval import VectorRetriever, RetrievalConfig


@pytest.fixture
def sample_corpus():
    """Fixture for sample corpus."""
    return [
        "Machine learning is a method of data analysis that automates model building.",
        "Supervised learning uses labeled data to train models for prediction.",
        "Unsupervised learning finds patterns in data without predefined labels.",
        "Deep learning is a subset of machine learning using neural networks."
    ]


@pytest.fixture
def retriever(sample_corpus):
    """Fixture for initialized VectorRetriever."""
    config = RetrievalConfig(top_k=None, cluster_threshold=5)
    retriever = VectorRetriever(config)
    retriever.load_or_compute_embeddings(sample_corpus)
    retriever.cluster_embeddings()
    retriever.build_index()
    return retriever


class TestVectorRetriever:
    def test_retrieval_direct_search(self, retriever, sample_corpus):
        """Test direct search for small corpora."""
        # Given
        query = "What is supervised learning in machine learning?"
        expected_results = [
            {
                "rank": 1,
                "score": pytest.approx(0.9, abs=0.1),
                "num_tokens": 10,
                "text": "Supervised learning uses labeled data to train models for prediction."
            },
            {
                "rank": 2,
                "score": pytest.approx(0.8, abs=0.1),
                "num_tokens": 12,
                "text": "Machine learning is a method of data analysis that automates model building."
            },
            {
                "rank": 3,
                "score": pytest.approx(0.7, abs=0.1),
                "num_tokens": 11,
                "text": "Deep learning is a subset of machine learning using neural networks."
            },
            {
                "rank": 4,
                "score": pytest.approx(0.6, abs=0.1),
                "num_tokens": 9,
                "text": "Unsupervised learning finds patterns in data without predefined labels."
            }
        ]
        # When
        top_chunks = retriever.search_chunks(query)
        # Then
        result = top_chunks
        assert len(result) == len(
            expected_results), f"Expected {len(expected_results)} chunks, but got {len(result)}"
        for res, exp in zip(result, expected_results):
            assert res["text"] == exp["text"], f"Expected text {exp['text']}, but got {res['text']}"
            assert res["num_tokens"] == exp[
                "num_tokens"], f"Expected num_tokens {exp['num_tokens']}, but got {res['num_tokens']}"
            assert res["rank"] == exp["rank"], f"Expected rank {exp['rank']}, but got {res['rank']}"
            assert res["score"] == pytest.approx(
                exp["score"], abs=0.1), f"Expected score {exp['score']}, but got {res['score']}"

    def test_retrieval_with_custom_args(self, retriever, sample_corpus):
        """Test retrieval with custom arguments overriding defaults."""
        # Given
        query = "What is supervised learning in machine learning?"
        top_k = 2
        cluster_threshold = 3
        expected_results = [
            {
                "rank": 1,
                "score": pytest.approx(0.9, abs=0.1),
                "num_tokens": 10,
                "text": "Supervised learning uses labeled data to train models for prediction."
            },
            {
                "rank": 2,
                "score": pytest.approx(0.8, abs=0.1),
                "num_tokens": 12,
                "text": "Machine learning is a method of data analysis that automates model building."
            }
        ]
        # When
        top_chunks = retriever.search_chunks(
            query, top_k=top_k, cluster_threshold=cluster_threshold)
        # Then
        result = top_chunks
        assert len(result) == 2, f"Expected 2 chunks, but got {len(result)}"
        for res, exp in zip(result, expected_results):
            assert res["text"] == exp["text"], f"Expected text {exp['text']}, but got {res['text']}"
            assert res["num_tokens"] == exp[
                "num_tokens"], f"Expected num_tokens {exp['num_tokens']}, but got {res['num_tokens']}"
            assert res["rank"] == exp["rank"], f"Expected rank {exp['rank']}, but got {res['rank']}"
            assert res["score"] == pytest.approx(
                exp["score"], abs=0.1), f"Expected score {exp['score']}, but got {res['score']}"

    def test_retrieval_no_cache(self, sample_corpus):
        """Test embedding computation without cache."""
        # Given a retriever with no cache file
        config = RetrievalConfig(cache_file=None)
        retriever = VectorRetriever(config)
        # When loading or computing embeddings without cache
        embeddings = retriever.load_or_compute_embeddings(
            sample_corpus, cache_file=None)
        # Then embeddings should be computed and not cached
        assert embeddings.shape[0] == len(
            sample_corpus), f"Expected {len(sample_corpus)} embeddings, got {embeddings.shape[0]}"
        assert not os.path.exists(
            "embeddings.pkl"), "Cache file should not exist"

    def test_retrieval_with_threshold(self, retriever, sample_corpus):
        """Test retrieval with a similarity threshold."""
        # Given
        query = "What is supervised learning in machine learning?"
        threshold = 0.7
        expected_results = [
            {
                "rank": 1,
                "score": pytest.approx(0.9, abs=0.1),
                "num_tokens": 10,
                "text": "Supervised learning uses labeled data to train models for prediction."
            },
            {
                "rank": 2,
                "score": pytest.approx(0.8, abs=0.1),
                "num_tokens": 12,
                "text": "Machine learning is a method of data analysis that automates model building."
            }
        ]
        # When
        top_chunks = retriever.search_chunks(query, threshold=threshold)
        # Then
        result = top_chunks
        assert len(
            result) <= 3, f"Expected at most 3 chunks, got {len(result)}"
        for res, exp in zip(result, expected_results):
            assert res["text"] == exp["text"], f"Expected text {exp['text']}, but got {res['text']}"
            assert res["num_tokens"] == exp[
                "num_tokens"], f"Expected num_tokens {exp['num_tokens']}, but got {res['num_tokens']}"
            assert res["rank"] == exp["rank"], f"Expected rank {exp['rank']}, but got {res['rank']}"
            assert res["score"] >= threshold, f"Score {res['score']} should be >= {threshold}"

    def test_retrieval_all_results(self, retriever, sample_corpus):
        """Test retrieval when top_k is None to return all results."""
        # Given
        query = "What is supervised learning in machine learning?"
        expected_min_chunks = len(sample_corpus)
        # When
        top_chunks = retriever.search_chunks(query, top_k=None)
        # Then
        result = top_chunks
        assert len(
            result) == expected_min_chunks, f"Expected {expected_min_chunks} chunks, but got {len(result)}"
        assert set(chunk["text"] for chunk in result) == set(
            sample_corpus), f"Expected all corpus chunks, but got {[chunk['text'] for chunk in result]}"
        for i, chunk in enumerate(result):
            assert chunk["rank"] == i + \
                1, f"Expected rank {i + 1}, but got {chunk['rank']}"
            assert isinstance(
                chunk["score"], float), f"Expected score to be float, got {type(chunk['score'])}"
            assert isinstance(
                chunk["num_tokens"], int), f"Expected num_tokens to be int, got {type(chunk['num_tokens'])}"

    def test_retrieval_empty_corpus(self, sample_corpus):
        """Test error handling for empty corpus."""
        # Given a retriever with default config
        config = RetrievalConfig()
        retriever = VectorRetriever(config)
        empty_corpus: List[str] = []
        # When attempting to load empty corpus
        # Then it should raise ValueError
        with pytest.raises(ValueError, match="Corpus cannot be empty"):
            retriever.load_or_compute_embeddings(empty_corpus)

    def test_retrieval_empty_query(self, retriever):
        """Test error handling for empty query."""
        # Given a retriever with initialized corpus
        query = ""
        # When retrieving with empty query
        # Then it should raise ValueError
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.search_chunks(query)

    def test_retrieval_with_dict_config(self, sample_corpus):
        """Test VectorRetriever initialization with a typed dictionary config."""
        # Given
        config_dict = {
            "min_cluster_size": 2,
            "k_clusters": 2,
            "top_k": 3,
            "cluster_threshold": 5,
            "model_name": "mxbai-embed-large",
            "cache_file": None,
            "threshold": None
        }
        retriever = VectorRetriever(config_dict)
        retriever.load_or_compute_embeddings(sample_corpus)
        retriever.cluster_embeddings()
        retriever.build_index()
        query = "What is supervised learning in machine learning?"
        expected_results = [
            {
                "rank": 1,
                "score": pytest.approx(0.9, abs=0.1),
                "num_tokens": 10,
                "text": "Supervised learning uses labeled data to train models for prediction."
            },
            {
                "rank": 2,
                "score": pytest.approx(0.8, abs=0.1),
                "num_tokens": 12,
                "text": "Machine learning is a method of data analysis that automates model building."
            },
            {
                "rank": 3,
                "score": pytest.approx(0.7, abs=0.1),
                "num_tokens": 11,
                "text": "Deep learning is a subset of machine learning using neural networks."
            }
        ]
        # When
        top_chunks = retriever.search_chunks(query)
        # Then
        result = top_chunks
        assert len(result) == len(
            expected_results), f"Expected {len(expected_results)} chunks, but got {len(result)}"
        for res, exp in zip(result, expected_results):
            assert res["text"] == exp["text"], f"Expected text {exp['text']}, but got {res['text']}"
            assert res["num_tokens"] == exp[
                "num_tokens"], f"Expected num_tokens {exp['num_tokens']}, but got {res['num_tokens']}"
            assert res["rank"] == exp["rank"], f"Expected rank {exp['rank']}, but got {res['rank']}"
            assert res["score"] == pytest.approx(
                exp["score"], abs=0.1), f"Expected score {exp['score']}, but got {res['score']}"


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up cache file if exists."""
    cache_file = "embeddings.pkl"
    if os.path.exists(cache_file):
        os.remove(cache_file)
    yield
