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
