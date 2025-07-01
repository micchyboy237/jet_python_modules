from jet.models.model_types import EmbedModelType
from jet.models.embeddings.base import load_embed_model
from jet.data.header_utils._search_headers import search_headers, cosine_similarity
from jet.data.header_types import TextNode
from jet.data.header_utils import VectorStore
import pytest
import numpy as np
from typing import List, Tuple
from jet.data.header_types import TextNode, HeaderNode
from jet.data.header_utils import prepare_for_rag, VectorStore, search_headers
from jet.models.tokenizer.base import get_tokenizer
from tokenizers import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Provide a tokenizer for tests."""
    return get_tokenizer("all-MiniLM-L6-v2")


@pytest.fixture
def default_params() -> dict:
    """Provide default parameters for chunking functions."""
    return {"chunk_size": 50, "chunk_overlap": 10, "buffer": 5}


class TestSearchHeaders:
    def test_search_single_node(self, tokenizer: Tokenizer) -> None:
        """Test searching with a single node."""
        # Given
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="Test Content",
            meta={},
            chunk_index=0,
            num_tokens=0
        )
        query = "Test Content"
        expected_node_id = node.id
        expected_min_similarity = 0.5
        vector_store = prepare_for_rag(
            [node], model="all-MiniLM-L6-v2", tokenizer=tokenizer)

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=1)

        # Then
        assert len(results) == 1
        result_node, similarity = results[0]
        assert result_node.id == expected_node_id
        assert result_node.content == "Test Content"
        assert similarity > expected_min_similarity

    def test_search_with_chunked_nodes(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test searching with chunked nodes."""
        # Given
        long_content = "This is a long sentence. " * 20
        node = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Long Header",
            content=long_content,
            level=1,
            children=[],
            chunk_index=0,
            num_tokens=0
        )
        query = "long sentence"
        expected_min_chunks = 3
        expected_min_similarity = 0.5
        vector_store = prepare_for_rag(
            [node],
            model="all-MiniLM-L6-v2",
            tokenizer=tokenizer,
            **default_params
        )

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=5)

        # Then
        assert len(results) >= expected_min_chunks
        for result_node, similarity in results:
            assert result_node.header == "Long Header"
            assert result_node.num_tokens <= default_params["chunk_size"]
            assert result_node.num_tokens > 0
            assert similarity > expected_min_similarity

    def test_search_empty_vector_store(self, tokenizer: Tokenizer) -> None:
        """Test searching with an empty vector store."""
        # Given
        vector_store = VectorStore()
        query = "Test query"

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=5)

        # Then
        assert len(results) == 0


class TestSearchHeaders2:
    @pytest.fixture
    def vector_store(self) -> VectorStore:
        """Create a vector store with sample nodes and embeddings."""
        store = VectorStore()
        nodes = [
            TextNode(
                id="node1",
                line=1,
                type="paragraph",
                header="Header 1",
                content="Header 1\nSample content one",
                parent_header="Parent Header 1",
                num_tokens=10
            ),
            TextNode(
                id="node2",
                line=2,
                type="paragraph",
                header="Header 2",
                content="Header 2\nSample content two",
                parent_header=None,
                num_tokens=8
            ),
            TextNode(
                id="node3",
                line=3,
                type="paragraph",
                header="",
                content="Sample content three",
                parent_header="Parent Header 3",
                num_tokens=7
            ),
            TextNode(
                id="node4",
                line=4,
                type="paragraph",
                header="Header 4",
                content="",
                parent_header="Parent Header 4",
                num_tokens=0
            )
        ]
        model = load_embed_model("all-MiniLM-L6-v2")
        embeddings = model.encode(
            [
                "Parent Header 1\nHeader 1\nSample content one",
                "Header 2\nSample content two",
                "Parent Header 3\nSample content three",
                "Parent Header 4\nHeader 4"
            ],
            show_progress_bar=False
        )
        for node, emb in zip(nodes, embeddings):
            store.add(node, emb)
        return store

    def test_search_headers_with_parent_header(self, vector_store: VectorStore) -> None:
        """Test search_headers with a node that has a parent header."""
        # Given
        query = "Sample query"
        model = "all-MiniLM-L6-v2"
        top_k = 4
        expected_nodes = 4
        transformer = load_embed_model(model)
        query_embedding = transformer.encode(
            [query], show_progress_bar=False)[0]

        # When
        results = search_headers(query, vector_store, model, top_k)

        # Then
        assert len(results) <= expected_nodes
        for node, score in results:
            assert isinstance(node, TextNode)
            assert isinstance(score, float)
            assert 0 <= score <= 1
            # Verify parent_header is preserved if it exists
            if node.id == "node1":
                assert node.parent_header == "Parent Header 1"
            elif node.id == "node2":
                assert node.parent_header is None
            elif node.id == "node3":
                assert node.parent_header == "Parent Header 3"
            elif node.id == "node4":
                assert node.parent_header == "Parent Header 4"

    def test_search_headers_empty_store(self, vector_store: VectorStore) -> None:
        """Test search_headers with an empty vector store."""
        # Given
        query = "Sample query"
        model = "all-MiniLM-L6-v2"
        top_k = 5
        empty_store = VectorStore()
        expected = []

        # When
        results = search_headers(query, empty_store, model, top_k)

        # Then
        assert results == expected

    def test_search_headers_similarity_calculation(self, vector_store: VectorStore) -> None:
        """Test similarity calculation with parent header and header combined."""
        # Given
        query = "Parent Header 1 Header 1 Sample content one"
        model = "all-MiniLM-L6-v2"
        top_k = 1
        transformer = load_embed_model(model)
        query_embedding = transformer.encode(
            [query], show_progress_bar=False)[0]
        content_embedding = transformer.encode(
            ["Sample content one"], show_progress_bar=False)[0]
        header_embedding = transformer.encode(
            ["Parent Header 1\nHeader 1"], show_progress_bar=False)[0]
        expected_content_sim = cosine_similarity(
            query_embedding, content_embedding)
        expected_header_sim = cosine_similarity(
            query_embedding, header_embedding)
        expected_avg_sim = (expected_content_sim + expected_header_sim) / 2

        # When
        results = search_headers(query, vector_store, model, top_k)

        # Then
        assert len(results) == 1
        node, score = results[0]
        assert node.id == "node1"
        # Allow small floating-point differences
        assert abs(score - expected_avg_sim) < 0.1

    def test_search_headers_no_parent_header(self, vector_store: VectorStore) -> None:
        """Test similarity calculation for node without parent header."""
        # Given
        query = "Header 2 Sample content two"
        model = "all-MiniLM-L6-v2"
        top_k = 1
        transformer = load_embed_model(model)
        query_embedding = transformer.encode(
            [query], show_progress_bar=False)[0]
        content_embedding = transformer.encode(
            ["Sample content two"], show_progress_bar=False)[0]
        header_embedding = transformer.encode(
            ["Header 2"], show_progress_bar=False)[0]
        expected_content_sim = cosine_similarity(
            query_embedding, content_embedding)
        expected_header_sim = cosine_similarity(
            query_embedding, header_embedding)
        expected_avg_sim = (expected_content_sim + expected_header_sim) / 2

        # When
        results = search_headers(query, vector_store, model, top_k)

        # Then
        assert len(results) == 1
        node, score = results[0]
        assert node.id == "node2"
        # Allow small floating-point differences
        assert abs(score - expected_avg_sim) < 0.1

    def test_search_headers_no_header(self, vector_store: VectorStore) -> None:
        """Test similarity calculation for node with parent header but no header."""
        # Given
        query = "Parent Header 3 Sample content three"
        model = "all-MiniLM-L6-v2"
        top_k = 1
        transformer = load_embed_model(model)
        query_embedding = transformer.encode(
            [query], show_progress_bar=False)[0]
        content_embedding = transformer.encode(
            ["Sample content three"], show_progress_bar=False)[0]
        header_embedding = transformer.encode(
            ["Parent Header 3"], show_progress_bar=False)[0]
        expected_content_sim = cosine_similarity(
            query_embedding, content_embedding)
        expected_header_sim = cosine_similarity(
            query_embedding, header_embedding)
        expected_avg_sim = (expected_content_sim + expected_header_sim) / 2

        # When
        results = search_headers(query, vector_store, model, top_k)

        # Then
        assert len(results) == 1
        node, score = results[0]
        assert node.id == "node3"
        # Allow small floating-point differences
        assert abs(score - expected_avg_sim) < 0.1

    def test_search_headers_empty_content(self, vector_store: VectorStore) -> None:
        """Test similarity score is 0 for nodes with empty content."""
        # Given
        query = "Parent Header 4 Header 4"
        model = "all-MiniLM-L6-v2"
        top_k = 4
        expected_score = 0.0

        # When
        results = search_headers(query, vector_store, model, top_k)

        # Then
        for node, score in results:
            if node.id == "node4":
                assert score == expected_score
                break
        else:
            assert False, "Node with empty content not found in results"
