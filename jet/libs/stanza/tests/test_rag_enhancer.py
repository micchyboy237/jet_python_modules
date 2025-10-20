import pytest
from typing import List

from jet.libs.stanza.rag_enhancer import RAGEnhancer

@pytest.fixture
def improver():
    """Fixture to initialize and clean up RAGEnhancer."""
    yield RAGEnhancer(embedding_model='all-MiniLM-L6-v2')

def test_preprocess_documents(improver):
    # Given: Sample markdown documents
    documents: List[str] = [
        "# Title\nThis is a sentence about Apple Inc. Another about Paris.",
        "Simple text without markdown."
    ]
    expected_chunks: List[str] = [
        "This is a sentence about Apple Inc.",
        "Another about Paris.",
        "Simple text without markdown."
    ]
    expected_entity_map = {0: ['Apple Inc.'], 1: ['Paris']}

    # When: Preprocess to get chunks and entities
    result_chunks, result_entity_map = improver.preprocess_documents(documents)

    # Then: Verify exact chunks and entity map
    assert result_chunks == expected_chunks
    assert result_entity_map == expected_entity_map

def test_model_topics(improver):
    # Given: Sample chunks
    chunks: List[str] = ["Tech company Apple", "City of Paris", "Fruit apple"]
    expected_topic_map = {0: 0, 1: -1, 2: 0}  # Apple-related in one topic, Paris as outlier

    # When: Model topics
    result_topic_map = improver.model_topics(chunks)

    # Then: Verify exact topic mapping
    assert result_topic_map == expected_topic_map  # Note: Mock embeddings if topics vary

def test_retrieve_contexts(improver):
    # Given: Query and documents
    query = "Tell me about Apple Inc."
    documents: List[str] = [
        "# Tech\nApple Inc. is a company.",
        "# Travel\nParis is a city."
    ]
    expected_contexts: List[str] = ["Apple Inc. is a company."]

    # When: Retrieve top 2 contexts
    result_contexts = improver.retrieve_contexts(query, documents, top_k=2)

    # Then: Verify exact retrieved contexts
    assert result_contexts == expected_contexts


class TestSearchFeature:
    @pytest.fixture
    def improver(self):
        """Fixture to initialize RAGEnhancer for search tests."""
        yield RAGEnhancer(embedding_model='all-MiniLM-L6-v2')

    def test_search_documents_hybrid(self, improver):
        # Given: Query and mixed-format documents (markdown + plain)
        query = "Apple company"
        documents: List[str] = [
            "# Tech\nApple Inc. is a technology company.",
            "Paris is the capital of France. No relation to fruit."
        ]
        expected: List[str] = ["Apple Inc. is a technology company."]  # Top result: semantic + keyword match

        # When: Perform hybrid search
        result = improver.search_documents(query, documents, top_k=1, alpha=0.5)

        # Then: Verify exact top chunk
        assert result == expected

    def test_search_documents_keyword_only(self, improver):
        # Given: Query with exact phrase; documents favoring lexical match
        query = "Paris France"
        documents: List[str] = [
            "# Cities\nParis is the capital of France.",
            "Apple Inc. produces fruits? No, technology."
        ]
        expected: List[str] = ["Paris is the capital of France."]

        # When: Keyword-only search (alpha=0)
        result = improver.search_documents(query, documents, top_k=1, alpha=0.0)

        # Then: Verify exact top chunk via TF-IDF
        assert result == expected

    def test_search_documents_semantic_only(self, improver):
        # Given: Synonym-based query; documents with semantic overlap
        query = "tech giant"
        documents: List[str] = [
            "Microsoft is a software leader.",
            "Apple Inc. innovates in hardware and software."
        ]
        expected: List[str] = ["Apple Inc. innovates in hardware and software."]  # Assumes model ranks Apple higher semantically

        # When: Semantic-only search (alpha=1)
        result = improver.search_documents(query, documents, top_k=1, alpha=1.0)

        # Then: Verify exact top chunk via embeddings
        assert result == expected

    def test_search_documents_empty(self, improver):
        # Given: Empty documents list
        query = "test"
        documents: List[str] = []

        # When: Perform search
        result = improver.search_documents(query, documents, top_k=3)

        # Then: Verify empty result
        expected: List[str] = []
        assert result == expected
