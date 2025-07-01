import pytest
import numpy as np
from jet.data.header_types import TextNode, HeaderNode
from jet.code.markdown_types import ContentType
from jet.data.header_utils import prepare_for_rag, VectorStore


class TestPrepareForRAG:
    def test_prepare_single_node(self):
        # Given
        expected_node = TextNode(
            id="node1",
            line=1,
            type="paragraph",  # type: ignore
            header="Test Header",
            content="Test Content",
            meta={},
            chunk_index=0
        )
        expected_embedding_shape = (384,)  # all-MiniLM-L6-v2 embedding size

        # When
        vector_store = prepare_for_rag(
            [expected_node], model="all-MiniLM-L6-v2")

        # Then
        assert len(vector_store.nodes) == 1
        assert vector_store.nodes[0] == expected_node
        assert len(vector_store.embeddings) == 1
        assert vector_store.embeddings[0].shape == expected_embedding_shape

    def test_prepare_multiple_nodes(self):
        # Given
        expected_nodes = [
            TextNode(
                id=f"node{i}",
                line=i,
                type="paragraph",  # type: ignore
                header=f"Header {i}",
                content=f"Content {i}",
                meta={},
                chunk_index=0
            ) for i in range(3)
        ]
        expected_embedding_shape = (384,)

        # When
        vector_store = prepare_for_rag(
            expected_nodes, model="all-MiniLM-L6-v2")

        # Then
        assert len(vector_store.nodes) == 3
        assert vector_store.nodes == expected_nodes
        assert len(vector_store.embeddings) == 3
        for emb in vector_store.embeddings:
            assert emb.shape == expected_embedding_shape

    def test_prepare_empty_nodes(self):
        # Given
        expected_nodes: List[TextNode] = []
        expected_embedding_shape = (0,)

        # When
        vector_store = prepare_for_rag(
            expected_nodes, model="all-MiniLM-L6-v2")

        # Then
        assert len(vector_store.nodes) == 0
        assert len(vector_store.embeddings) == 0
        assert vector_store.get_embeddings().shape == expected_embedding_shape

    def test_prepare_complex_hierarchy(self):
        # Given
        expected_nodes = [
            TextNode(
                id="header1",
                line=1,
                type="paragraph",  # type: ignore
                header="Level 1 Header",
                content="Level 1 content",
                meta={},
                chunk_index=0
            ),
            TextNode(
                id="child1",
                line=2,
                type="code",  # type: ignore
                header="Code Child",
                content="print('Hello')",
                meta={"language": "python"},
                parent_id="header1",
                parent_header="Level 1 Header",
                chunk_index=0
            ),
            TextNode(
                id="child2",
                line=3,
                type="table",  # type: ignore
                header="Table Child",
                content="| Col1 | Col2 |\n|------|------|\n| A    | B    |",
                meta={"header": ["Col1", "Col2"], "rows": [["A", "B"]]},
                parent_id="header1",
                parent_header="Level 1 Header",
                chunk_index=0
            )
        ]
        expected_embedding_shape = (384,)

        # When
        vector_store = prepare_for_rag(
            expected_nodes, model="all-MiniLM-L6-v2")

        # Then
        assert len(vector_store.nodes) == 3
        assert vector_store.nodes == expected_nodes
        assert len(vector_store.embeddings) == 3
        for emb in vector_store.embeddings:
            assert emb.shape == expected_embedding_shape

    def test_prepare_unsorted_hierarchy(self):
        # Given
        expected_nodes = [
            TextNode(
                id="header3",
                line=3,
                type="paragraph",  # type: ignore
                header="Level 3 Header",
                content="Level 3 content",
                meta={},
                parent_id="header2",
                parent_header="Level 2 Header",
                chunk_index=0
            ),
            TextNode(
                id="header1",
                line=1,
                type="paragraph",  # type: ignore
                header="Level 1 Header",
                content="Level 1 content",
                meta={},
                chunk_index=0
            ),
            TextNode(
                id="header2",
                line=2,
                type="paragraph",  # type: ignore
                header="Level 2 Header",
                content="Level 2 content",
                meta={},
                parent_id="header1",
                parent_header="Level 1 Header",
                chunk_index=0
            )
        ]
        expected_embedding_shape = (384,)

        # When
        vector_store = prepare_for_rag(
            expected_nodes, model="all-MiniLM-L6-v2")

        # Then
        assert len(vector_store.nodes) == 3
        assert vector_store.nodes == expected_nodes
        assert len(vector_store.embeddings) == 3
        for emb in vector_store.embeddings:
            assert emb.shape == expected_embedding_shape

    def test_prepare_empty_content_and_meta(self):
        # Given
        expected_node = TextNode(
            id="node1",
            line=1,
            type="paragraph",  # type: ignore
            header="Empty Header",
            content="",
            meta={},
            chunk_index=0
        )
        expected_embedding_shape = (384,)

        # When
        vector_store = prepare_for_rag(
            [expected_node], model="all-MiniLM-L6-v2")

        # Then
        assert len(vector_store.nodes) == 1
        assert vector_store.nodes[0] == expected_node
        assert len(vector_store.embeddings) == 1
        assert vector_store.embeddings[0].shape == expected_embedding_shape
