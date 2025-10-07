import pytest
from typing import List, Tuple

from jet.adapters.bertopic.jet_examples.base.extract_topics_for_rag import extract_topics_for_rag

# Fixture to avoid repeated setup
@pytest.fixture
def topic_model_setup() -> Tuple[List[str], str]:
    docs = [
        "Climate change is accelerating due to fossil fuels and deforestation.",
        "Renewable energy sources like solar and wind are essential for sustainability."
    ]
    query = "environmental impacts"
    return docs, query

def test_extract_topics_for_rag_assigns_topics_and_probs(topic_model_setup):
    """
    Test that extract_topics_for_rag assigns topics and uses probabilities correctly.
    """
    # Given: A set of documents and a query
    docs, query = topic_model_setup
    expected_topic_count = 1  # Expect at least one topic (stochastic, so conservative)
    expected_doc_mapping = [
        {"doc_id": 0, "doc_text": docs[0][:100] + "...", "topic_id": 0, "probability": pytest.approx(0.5, abs=0.5)},
        {"doc_id": 1, "doc_text": docs[1][:100] + "...", "topic_id": 0, "probability": pytest.approx(0.5, abs=0.5)}
    ]

    # When: We extract topics and mappings
    topic_dicts, summaries, doc_mappings = extract_topics_for_rag(docs, query, top_k_topics=2, min_prob=0.1)

    # Then: Topics are assigned, and doc mappings use topics/probs correctly
    result_topic_count = len([t for t in topic_dicts if t["topic_id"] != -1])
    assert result_topic_count >= expected_topic_count, f"Expected at least {expected_topic_count} topics, got {result_topic_count}"
    
    # Verify doc mappings (focus on structure and reasonable probs)
    result_doc_mappings = [
        {"doc_id": m["doc_id"], "doc_text": m["doc_text"], "topic_id": m["topic_id"], "probability": m["probability"]}
        for m in doc_mappings
    ]
    for expected in expected_doc_mapping:
        assert any(
            r["doc_id"] == expected["doc_id"] and r["doc_text"] == expected["doc_text"]
            and r["topic_id"] >= -1 and r["probability"] >= 0.1
            for r in result_doc_mappings
        ), f"Doc {expected['doc_id']} mapping incorrect or low confidence"

# Cleanup not needed for BERTopic in-memory objects