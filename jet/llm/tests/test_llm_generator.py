import pytest
from typing import List, Tuple
from jet.llm.llm_generator import LLMGenerator, LLMConfig


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
def generator():
    """Fixture for LLMGenerator with default config."""
    return LLMGenerator()


class TestLLMGenerator:
    def test_generate_response_with_chunks(self, generator, sample_corpus):
        """Test LLM response generation with valid chunks."""
        # Given: A query and relevant chunks with scores
        query = "What is supervised learning in machine learning?"
        chunks = [
            (sample_corpus[1], 0.9512),
            (sample_corpus[2], 0.7200),
            (sample_corpus[0], 0.6416)
        ]
        expected_response_contains = (
            "Based on the provided information",
            sample_corpus[1],
            "In summary, Supervised learning uses labeled data to train models for prediction.",
            "(Score: 0.9512)"
        )

        # When: Generating a response
        response = generator.generate_response(query, chunks)

        # Then: The response should include expected content
        for expected in expected_response_contains:
            assert expected in response, f"Expected '{expected}' in response, but got: {response}"

    def test_generate_response_no_chunks(self, generator):
        """Test LLM response when no chunks are provided."""
        # Given: A query with no chunks
        query = "What is supervised learning in machine learning?"
        chunks: List[Tuple[str, float]] = []
        expected_response = "No relevant information found for the query."

        # When: Generating a response
        response = generator.generate_response(query, chunks)

        # Then: The response should indicate no information
        assert response == expected_response, f"Expected '{expected_response}', but got: {response}"

    def test_generate_response_conversational_tone(self, sample_corpus):
        """Test LLM response with conversational tone."""
        # Given: A generator with conversational tone config
        config = {"response_tone": "conversational",
                  "max_context_length": 1000, "include_scores": True}
        generator = LLMGenerator(config)
        query = "What is supervised learning?"
        chunks = [(sample_corpus[1], 0.9512)]
        expected_response_contains = (
            "Hey, I looked into your question",
            sample_corpus[1],
            "So, to sum it up",
            "(Score: 0.9512)"
        )

        # When: Generating a response
        response = generator.generate_response(query, chunks)

        # Then: The response should reflect conversational tone
        for expected in expected_response_contains:
            assert expected in response, f"Expected '{expected}' in response, but got: {response}"

    def test_generate_response_truncated_context(self, sample_corpus):
        """Test LLM response with context truncation."""
        # Given: A generator with a short max context length
        config = {"max_context_length": 80, "include_scores": True}
        generator = LLMGenerator(config)
        query = "What is supervised learning?"
        chunks = [(sample_corpus[1], 0.9512)]
        expected_response_contains = (
            "Based on the provided information",
            "Supervised learning uses labeled data to train models"
        )

        # When: Generating a response
        response = generator.generate_response(query, chunks)

        # Then: The response should be truncated but valid
        assert len(
            response) < 150, f"Response too long: {len(response)}, response: {response}"
        for expected in expected_response_contains:
            assert expected in response, f"Expected '{expected}' in response, but got: {response}"

    def test_generate_response_empty_query(self, generator, sample_corpus):
        """Test LLM response with empty query."""
        # Given: An empty query
        query = ""
        chunks = [(sample_corpus[1], 0.9512)]
        expected_response = "Query cannot be empty."

        # When: Generating a response
        response = generator.generate_response(query, chunks)

        # Then: The response should indicate an empty query
        assert response == expected_response, f"Expected '{expected_response}', but got: {response}"
