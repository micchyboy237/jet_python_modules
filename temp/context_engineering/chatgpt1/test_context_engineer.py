# test_context_engineer.py
from context_engineer import Embedder, retrieve_top_k

def test_retrieve_top_k_simple():
    # Given: two short documents
    docs = [
        "Alice loves apples and lives in Berlin.",
        "Bob works at OpenAI and loves machine learning."
    ]
    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    # When: querying for 'who loves apples'
    results = retrieve_top_k("who loves apples", docs, embedder, k=2)
    # Then: top retrieved doc should point to the doc about Alice
    assert len(results) > 0
    assert results[0]["id"] == 0
    assert "apples" in results[0]["text"]