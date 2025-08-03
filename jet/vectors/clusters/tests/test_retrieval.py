import os
import pytest
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple
from jet.vectors.clusters.retrieval import VectorRetriever, LLMGenerator, RetrievalConfig


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
    config = RetrievalConfig(k_chunks=3, cluster_threshold=5)
    retriever = VectorRetriever(config)
    retriever.load_or_compute_embeddings(sample_corpus)
    retriever.cluster_embeddings()
    retriever.build_index()
    return retriever


@pytest.fixture
def generator():
    """Fixture for LLMGenerator."""
    return LLMGenerator()


class TestVectorRetriever:
    def test_retrieval_direct_search(self, retriever, sample_corpus):
        """Test direct search for small corpora."""
        # Given a retriever with a small corpus and a query
        query = "What is supervised learning in machine learning?"
        expected_chunks = [
            "Supervised learning uses labeled data to train models for prediction.",
            "Machine learning is a method of data analysis that automates model building.",
            "Deep learning is a subset of machine learning using neural networks."
        ]
        # When retrieving chunks
        top_chunks = retriever.retrieve_chunks(query)
        result = [chunk for chunk, _ in top_chunks]
        # Then the result should match expected chunks
        assert result == expected_chunks, f"Expected {expected_chunks}, but got {result}"

    def test_retrieval_with_custom_args(self, retriever, sample_corpus):
        """Test retrieval with custom arguments overriding defaults."""
        # Given a retriever and custom arguments
        query = "What is supervised learning in machine learning?"
        k_chunks = 2
        cluster_threshold = 3
        expected_chunks = [
            "Supervised learning uses labeled data to train models for prediction.",
            "Machine learning is a method of data analysis that automates model building."
        ]
        # When retrieving chunks with custom arguments
        top_chunks = retriever.retrieve_chunks(
            query, k_chunks=k_chunks, cluster_threshold=cluster_threshold)
        result = [chunk for chunk, _ in top_chunks]
        # Then the result should respect custom arguments (k_chunks=2)
        assert len(result) == 2, f"Expected 2 chunks, but got {len(result)}"
        assert result == expected_chunks, f"Expected {expected_chunks}, but got {result}"

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
            retriever.retrieve_chunks(query)

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


class TestLLMGenerator:
    def test_generate_response_with_chunks(self, generator, sample_corpus):
        """Test LLM response generation with valid chunks."""
        # Given a generator, query, and chunks
        query = "What is supervised learning in machine learning?"
        chunks = [
            (sample_corpus[1], 0.9512),
            (sample_corpus[2], 0.7200),
            (sample_corpus[0], 0.6416)
        ]
        expected_response_contains = (
            "Based on the provided information",
            sample_corpus[1],
            "In summary, Supervised learning uses labeled data to train models for prediction."
        )
        # When generating response
        response = generator.generate_response(query, chunks)
        # Then the response should contain expected content
        for expected in expected_response_contains:
            assert expected in response, f"Expected '{expected}' in response, but got: {response}"

    def test_generate_response_no_chunks(self, generator):
        """Test LLM response when no chunks are provided."""
        # Given a generator and query with no chunks
        query = "What is supervised learning in machine learning?"
        chunks: List[Tuple[str, float]] = []
        expected_response = "No relevant information found."
        # When generating response
        response = generator.generate_response(query, chunks)
        # Then the response should match expected
        assert response == expected_response, f"Expected '{expected_response}', but got: {response}"


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up cache file if exists."""
    cache_file = "embeddings.pkl"
    if os.path.exists(cache_file):
        os.remove(cache_file)
    yield
