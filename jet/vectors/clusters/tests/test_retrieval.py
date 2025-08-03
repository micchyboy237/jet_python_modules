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
    config = RetrievalConfig(
        k_chunks=3, cluster_threshold=5)  # Small threshold to test clustering
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
        # Given
        query = "What is supervised learning in machine learning?"
        expected_chunks = [
            "Supervised learning uses labeled data to train models for prediction.",
            "Machine learning is a method of data analysis that automates model building.",
            "Deep learning is a subset of machine learning using neural networks."
        ]

        # When
        top_chunks = retriever.retrieve_chunks(query)
        result = [chunk for chunk, _ in top_chunks]

        # Then
        assert result == expected_chunks, f"Expected {expected_chunks}, but got {result}"

    def test_retrieval_empty_corpus(self, sample_corpus):
        """Test error handling for empty corpus."""
        # Given
        config = RetrievalConfig()
        retriever = VectorRetriever(config)
        empty_corpus: List[str] = []

        # When/Then
        with pytest.raises(ValueError, match="Corpus cannot be empty"):
            retriever.load_or_compute_embeddings(empty_corpus)

    def test_retrieval_empty_query(self, retriever):
        """Test error handling for empty query."""
        # Given
        query = ""

        # When/Then
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve_chunks(query)


class TestLLMGenerator:
    def test_generate_response_with_chunks(self, generator, sample_corpus):
        """Test LLM response generation with valid chunks."""
        # Given
        query = "What is supervised learning in machine learning?"
        chunks = [
            (sample_corpus[1], 0.9512),
            (sample_corpus[2], 0.7200),
            (sample_corpus[0], 0.6416)
        ]
        expected_response_contains = (
            "Based on the provided information",
            sample_corpus[1],  # Supervised learning chunk
            "In summary, Supervised learning uses labeled data to train models for prediction."
        )

        # When
        response = generator.generate_response(query, chunks)

        # Then
        for expected in expected_response_contains:
            assert expected in response, f"Expected '{expected}' in response, but got: {response}"

    def test_generate_response_no_chunks(self, generator):
        """Test LLM response when no chunks are provided."""
        # Given
        query = "What is supervised learning in machine learning?"
        chunks: List[Tuple[str, float]] = []
        expected_response = "No relevant information found."

        # When
        response = generator.generate_response(query, chunks)

        # Then
        assert response == expected_response, f"Expected '{expected_response}', but got: {response}"


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up cache file if exists."""
    cache_file = RetrievalConfig().cache_file
    if os.path.exists(cache_file):
        os.remove(cache_file)
    yield
