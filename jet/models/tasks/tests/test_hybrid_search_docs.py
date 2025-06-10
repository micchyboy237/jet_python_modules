import pytest
import json
from jet.models.tasks.hybrid_search_docs_with_bm25 import embed_chunks_parallel, search_docs, load_documents, split_document
from unittest.mock import Mock
import numpy as np


@pytest.fixture
def mock_documents():
    """Fixture providing mock documents for testing."""
    return [
        {
            "text": "# Title\nContent paragraph.",
            "id": 0
        },
        {
            "text": "## Subtitle\nAnother paragraph.",
            "id": 1
        }
    ]


def test_load_documents(mock_documents, tmp_path):
    """Test loading documents from file."""
    file_path = tmp_path / "docs.json"
    with open(file_path, "w") as f:
        json.dump(mock_documents, f)

    expected = [
        {"text": "# Title\nContent paragraph.", "id": 0},
        {"text": "## Subtitle\nAnother paragraph.", "id": 1}
    ]
    result = load_documents(str(file_path))
    assert result == expected


def test_split_document():
    """Test document splitting into chunks.

    Expects the input '# Title\nFirst sentence. Second sentence.' to be split into two chunks
    with chunk_size=10 (words), where the first chunk includes the header and first sentence,
    and the second chunk includes the second sentence, both with the same header.
    """
    doc_text = "# Title\nFirst sentence. Second sentence."
    doc_id = 0
    expected = [
        {
            "text": "# Title First sentence.",
            "headers": ["# Title"],
            "doc_id": 0
        },
        {
            "text": "Second sentence.",
            "headers": ["# Title"],
            "doc_id": 0
        }
    ]
    result = split_document(doc_text, doc_id, chunk_size=10, overlap=0)
    assert result == expected


def test_search_docs(mock_documents, tmp_path, monkeypatch):
    """Test hybrid search with mocked models."""
    file_path = tmp_path / "docs.json"
    with open(file_path, "w") as f:
        json.dump(mock_documents, f)

    query = "Title"
    mock_embedder = Mock()
    mock_embedder.encode.side_effect = lambda x, **kwargs: np.array(
        [[0.1, 0.2], [0.3, 0.4]])
    mock_embedder.get_sentence_embedding_dimension.return_value = 384
    mock_cross_encoder = Mock()
    mock_cross_encoder.predict.return_value = [0.9, 0.8]

    monkeypatch.setattr(
        "jet.models.tasks.hybrid_search_docs.SentenceTransformer", lambda x: mock_embedder)
    monkeypatch.setattr(
        "jet.models.tasks.hybrid_search_docs.CrossEncoder", lambda x: mock_cross_encoder)

    expected = [
        {
            "rank": 1,
            "doc_id": 0,
            "combined_score": pytest.approx(0.45, 0.1),
            "rerank_score": 0.9,
            "headers": ["# Title"],
            "text": "# Title\nContent paragraph."
        }
    ]
    result = search_docs(str(file_path), query, top_k=1,
                         rerank_top_k=1, bm25_weight=0.5)
    assert len(result) == 1
    assert result[0]["doc_id"] == expected[0]["doc_id"]
    assert result[0]["text"] == expected[0]["text"]


def test_embed_chunks_performance(mock_documents, monkeypatch):
    """Test embedding performance with batching."""
    from time import time
    embedder = Mock()
    embedder.encode.side_effect = lambda x, **kwargs: np.zeros(
        (len(x) if isinstance(x, list) else 1, 384))
    embedder.get_sentence_embedding_dimension.return_value = 384
    chunk_texts = ["text " * 50] * 100  # Simulate 100 chunks
    start_time = time()
    result = embed_chunks_parallel(chunk_texts, embedder)
    elapsed = time() - start_time
    expected_shape = (100, 384)
    assert result.shape == expected_shape
    assert elapsed < 1.0  # Adjust threshold based on M1 performance
