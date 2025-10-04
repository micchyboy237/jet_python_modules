import pytest
from typing import List

from jet.libs.bertopic.jet_examples.base.extract_keywords_for_rag import extract_keywords_for_rag

@pytest.fixture
def keyword_extraction_setup() -> tuple[List[str], str]:
    docs = [
        "Climate change is accelerating due to fossil fuels and deforestation.",
        "Renewable energy sources like solar and wind are essential for sustainability."
    ]
    query = "environmental impacts"
    return docs, query

def test_extract_keywords_for_rag_uses_query_weighting(keyword_extraction_setup):
    """
    Test that extract_keywords_for_rag uses query to weight keyword extraction.
    """
    # Given: Documents and a query
    docs, query = keyword_extraction_setup
    expected_keywords_doc0 = ["climate change", "deforestation"]  # Query-relevant terms
    expected_keywords_doc1 = ["renewable energy", "sustainability"]

    # When: Extract keywords with query weighting
    results = extract_keywords_for_rag(docs, query, top_k=2, ngram_range=(1, 2))

    # Then: Verify keywords are query-relevant and scores are reasonable
    result_keywords_doc0 = [kw["keyword"] for kw in results[0]["keywords"]]
    result_keywords_doc1 = [kw["keyword"] for kw in results[1]["keywords"]]
    
    # Check that at least one expected keyword appears per doc
    assert any(kw in result_keywords_doc0 for kw in expected_keywords_doc0), \
        f"Expected query-relevant keywords {expected_keywords_doc0}, got {result_keywords_doc0}"
    assert any(kw in result_keywords_doc1 for kw in expected_keywords_doc1), \
        f"Expected query-relevant keywords {expected_keywords_doc1}, got {result_keywords_doc1}"
    
    # Verify scores are positive and doc_text is correct
    for i, result in enumerate(results):
        assert result["doc_id"] == i, f"Expected doc_id {i}, got {result['doc_id']}"
        assert result["doc_text"] == docs[i][:100] + "...", f"Doc text mismatch for doc {i}"
        assert all(kw["score"] > 0 for kw in result["keywords"]), f"Non-positive scores in doc {i}"
