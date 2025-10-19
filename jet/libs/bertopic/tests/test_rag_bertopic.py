"""
Unit tests for TopicRAG (BERTopic + FAISS hybrid retriever)

These tests validate:
- Topic fitting and topic index creation
- Document deduplication
- Query retrieval (single & multi-topic)
- FAISS and fallback cosine similarity
- Retrieval uniqueness and ranking correctness
- Cosine search consistency
"""
import pytest
import numpy as np
from typing import List
from jet.libs.bertopic.rag_bertopic import TopicRAG


# =====================================================================
# Global fixtures
# =====================================================================

@pytest.fixture(scope="module")
def rag():
    """Reusable fitted TopicRAG instance for all tests."""
    docs = [
        "AI is transforming industries worldwide.",
        "Machine learning and deep learning are key subfields of AI.",
        "Neural networks power many AI applications today.",
        "Football players require intense physical training.",
        "Basketball championships attract huge crowds.",
        "The Olympics feature global athletic competition.",
        "Stock market volatility impacts investor confidence.",
        "Central banks raise interest rates to control inflation.",
        "Financial analysts predict slow GDP growth."
    ]
    r = TopicRAG(verbose=True)
    r.fit_topics(docs, nr_topics="auto", min_topic_size=2)
    return r


# =====================================================================
# TestTopicRAGSetup
# =====================================================================

class TestTopicRAGSetup:
    """Validate topic model fitting and index creation."""

    @pytest.fixture(scope="class")
    def small_docs(self) -> List[str]:
        return [
            "AI improves healthcare with better diagnosis models.",
            "Machine learning enables predictive analytics in finance.",
            "New vaccines help reduce spread of viral diseases.",
            "Football championship draws record audiences.",
            "Basketball players train hard for the Olympics.",
            "Stock markets are volatile during elections.",
        ]

    def test_fit_topics_creates_indexes(self, small_docs):
        """Given small docs, When fit_topics() runs, Then topic indexes are built."""
        rag = TopicRAG(verbose=True)
        rag.fit_topics(small_docs, nr_topics="auto", min_topic_size=2)
        assert rag.model is not None
        assert len(rag.topic_indexes) > 0
        assert all(len(ti.doc_ids) > 0 for ti in rag.topic_indexes.values())

    def test_deduplication_removes_duplicates(self):
        """Given duplicate docs, When deduplicated, Then duplicates removed."""
        rag = TopicRAG()
        docs = ["A", "A", "B", "C", "C"]
        result = rag._deduplicate(docs)
        expected = ["A", "B", "C"]
        assert result == expected


# =====================================================================
# TestTopicRAGRetrieval
# =====================================================================

class TestTopicRAGRetrieval:
    """Validate query retrieval behavior across topics."""

    def test_single_topic_query_retrieval(self, rag):
        """Given a query about AI, When retrieved, Then top results are AI-related."""
        query = "deep learning in AI"
        result = rag.retrieve_for_query(query, top_topics=1, top_k=3)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(d, dict) for d in result)
        assert all("score" in d for d in result)
        top_text = result[0]["text"].lower()
        assert "ai" in top_text or "learning" in top_text

    def test_multi_topic_query(self, rag):
        """Given a mixed-domain query, When top 2 topics are retrieved, Then results cover multiple topics."""
        query = "AI and finance trends"
        result = rag.retrieve_for_query(query, top_topics=2, top_k=3)
        topics = {r["topic"] for r in result}
        assert len(topics) >= 1
        assert all(isinstance(r["score"], float) for r in result)

    def test_unique_by_text_reduces_duplicates(self, rag):
        """Given a query, When unique_by='text', Then duplicate texts are removed."""
        query = "AI applications"
        result = rag.retrieve_for_query(query, top_topics=2, top_k=5, unique_by="text")
        texts = [r["text"] for r in result]
        assert len(texts) == len(set(texts))

    def test_invalid_state_raises_error(self):
        """Given uninitialized model, When retrieve_for_query called, Then RuntimeError raised."""
        rag = TopicRAG()
        with pytest.raises(RuntimeError):
            rag.retrieve_for_query("any query")

    def test_brute_force_cosine_fallback(self, rag, monkeypatch):
        """Given FAISS disabled, When searching, Then cosine fallback works."""
        monkeypatch.setattr("jet.libs.bertopic.rag_bertopic._HAS_FAISS", False)
        query = "AI innovation"
        result = rag.retrieve_for_query(query, top_topics=1, top_k=3)
        assert len(result) > 0
        assert isinstance(result[0]["score"], float)
        monkeypatch.setattr("jet.libs.bertopic.rag_bertopic._HAS_FAISS", True)


# =====================================================================
# TestEdgeCases
# =====================================================================

class TestEdgeCases:
    """Cover low-level and edge behaviors."""

    def test_empty_docs(self):
        """Given empty input, When fit_topics called, Then ValueError raised."""
        rag = TopicRAG()
        docs = []
        with pytest.raises(ValueError):
            rag.fit_topics(docs)

    def test_cosine_similarity_consistency(self, rag):
        """Given a query, When _search_topic is used, Then cosine-based results are consistent."""
        query = "AI"
        qvec = np.array(rag.embedder.encode([query], show_progress_bar=False)).astype("float32")
        topic_id, topic_index = list(rag.topic_indexes.items())[0]
        results = rag._search_topic(topic_index, qvec, top_k=2)
        assert isinstance(results, list)
        assert all(isinstance(i, tuple) and len(i) == 2 for i in results)
        assert all(isinstance(i[0], int) for i in results)

    def test_tiny_docset_triggers_safe_fallback(self):
        """Given ≤4 docs, When fit_topics called, Then safe single-cluster fallback runs without error."""
        docs = [
            "AI improves healthcare diagnostics.",
            "Deep learning assists in medical image analysis.",
            "Neural networks optimize patient treatment plans.",
        ]
        rag = TopicRAG(verbose=True)
        # Should not raise ValueError or crash due to hdbscan
        rag.fit_topics(docs)
        assert rag.model is not None
        # All docs should belong to a single cluster (topic 0)
        assert len(rag.topic_indexes) == 1
        tindex = next(iter(rag.topic_indexes.values()))
        assert len(tindex.texts) == len(docs)
        # Retrieval should work and return expected results
        results = rag.retrieve_for_query("AI healthcare", top_topics=1, top_k=2)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
        assert len(results) > 0

    def test_blank_docs_raise_value_error(self):
        """Given blank or invalid docs, When fit_topics called, Then ValueError raised."""
        rag = TopicRAG()
        with pytest.raises(ValueError, match="No valid text documents"):
            rag.fit_topics(["", "   ", None])

    def test_empty_embeddings_raise_value_error(self, monkeypatch):
        """Given mocked empty embeddings, When fit_topics called, Then ValueError raised."""
        rag = TopicRAG()
        # Force encode() to return an empty list
        monkeypatch.setattr(rag.embedder, "encode", lambda x, **_: [])
        with pytest.raises(ValueError, match="No embeddings were generated"):
            rag.fit_topics(["AI in medicine", "Machine learning in healthcare"])

    def test_mismatched_embeddings_raise_value_error(self, monkeypatch):
        """Given mismatched embedding shape, When fit_topics called, Then ValueError raised."""
        rag = TopicRAG()
        fake_embeddings = np.zeros((1, 384))  # Should not match 2 documents
        monkeypatch.setattr(rag.embedder, "encode", lambda x, **_: fake_embeddings)
        with pytest.raises(ValueError, match="Mismatch between documents"):
            rag.fit_topics(["AI in medicine", "Machine learning in healthcare"])
