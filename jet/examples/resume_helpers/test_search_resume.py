import pytest
from jet.examples.resume_helpers.base import VectorSearch


@pytest.fixture
def vector_search():
    """Fixture to initialize VectorSearch with sample data."""
    vs = VectorSearch()
    resume = {
        "id": "C123",
        "text": (
            "Jethro Reuel A. Estrada. Experienced Frontend Web/Mobile Developer with extensive experience "
            "in React, React Native, and Node.js. Developed multiple enterprise web applications using "
            "React and TypeScript, leading a team of 3 developers (2021–2023). Proficient in TypeScript for "
            "type-safe React development. Led sprint planning and mentored junior developers, promoting a "
            "collaborative team culture."
        ),
        "metadata": {"candidate_id": "C123", "document_type": "resume"}
    }
    vs.preprocess_and_index([resume], chunk_size=500)
    return vs


class TestVectorSearch:
    def test_dynamic_chunking(self, vector_search):
        """Test dynamic chunking splits text appropriately."""
        # Given
        text = (
            "Jethro Reuel A. Estrada. Experienced Frontend Web/Mobile Developer. "
            "Developed multiple enterprise web applications using React. "
            "Proficient in TypeScript for type-safe React development."
        )
        expected = [
            "Jethro Reuel A. Estrada. Experienced Frontend Web/Mobile Developer.",
            "Developed multiple enterprise web applications using React. Proficient in TypeScript for type-safe React development."
        ]

        # When
        result = vector_search._dynamic_chunking(text, base_chunk_size=100)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_query_expansion(self, vector_search):
        """Test query expansion generates an enhanced query."""
        # Given
        query = "React experience"
        expected = "React experience Details about react experience in the context of job experience or skills."

        # When
        result = vector_search.query_expansion(query)

        # Then
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_search_retrieval(self, vector_search):
        """Test search retrieves relevant chunks."""
        # Given
        query = "React experience"
        expected_chunk_ids = ["C123_0", "C123_1"]

        # When
        results = vector_search.search(query, top_k=2)
        result_chunk_ids = [result["chunk"]["id"] for result in results]

        # Then
        assert len(
            result_chunk_ids) == 2, f"Expected 2 results, got {len(result_chunk_ids)}"
        assert all(
            cid in expected_chunk_ids for cid in result_chunk_ids), f"Expected {expected_chunk_ids}, got {result_chunk_ids}"

    def test_evaluate_retrieval_ndcg(self, vector_search):
        """Test NDCG evaluation of retrieval results."""
        # Given
        query = "React experience"
        relevant_chunk_ids = ["C123_0", "C123_1"]
        expected_ndcg = 1.0  # Perfect ranking for relevant chunks

        # When
        result = vector_search.evaluate_retrieval(
            query, relevant_chunk_ids, top_k=2)

        # Then
        assert abs(
            result - expected_ndcg) < 0.1, f"Expected NDCG {expected_ndcg}, got {result}"
