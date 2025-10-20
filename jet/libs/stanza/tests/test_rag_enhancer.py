import pytest
from typing import List

from jet.libs.stanza.rag_enhancer import RAGContextImprover

@pytest.fixture
def improver():
    """Fixture to initialize and clean up RAGContextImprover."""
    yield RAGContextImprover(embedding_model='all-mpnet-base-v2')

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