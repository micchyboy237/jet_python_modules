import pytest
import numpy as np
from jet.data.header_types import TextNode
from jet.code.markdown_types import ContentType
from jet.data.header_utils._prepare_for_rag import VectorStore, prepare_for_rag
from jet.data.header_utils._search_headers import search_headers, cosine_similarity


class TestSearchHeaders:
    def test_search_relevant_nodes(self):
        # Given
        nodes = [
            TextNode(
                id=f"node{i}",
                line=i,
                type="paragraph",  # type: ignore
                header=f"Header {i}",
                content=f"Content about topic {i}",
                meta={},
                chunk_index=0
            ) for i in range(3)
        ]
        query = "topic 1"
        vector_store = prepare_for_rag(nodes, model="all-MiniLM-L6-v2")
        expected_top_k = 3
        expected_relevant_node_id = "node1"

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=expected_top_k)

        # Then
        assert len(results) <= expected_top_k
        assert all(isinstance(node, TextNode) and isinstance(score, float)
                   for node, score in results)
        assert any(node.id == expected_relevant_node_id for node,
                   score in results)
        assert all(0 <= score <= 1 for node, score in results)
        assert any(score > 0.5 for node,
                   score in results if node.id == expected_relevant_node_id)

    def test_search_empty_vector_store(self):
        # Given
        vector_store = VectorStore()
        query = "test query"
        expected_results: List = []

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=5)

        # Then
        assert results == expected_results

    def test_search_no_relevant_nodes(self):
        # Given
        nodes = [
            TextNode(
                id="node1",
                line=1,
                type="paragraph",  # type: ignore
                header="Header",
                content="Unrelated content",
                meta={},
                chunk_index=0
            )
        ]
        query = "completely different topic"
        vector_store = prepare_for_rag(nodes, model="all-MiniLM-L6-v2")
        expected_results: List = []

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=5)

        # Then
        assert len(results) == 0 or all(score < 0.3 for _, score in results)

    def test_search_complex_hierarchy(self):
        # Given
        nodes = [
            TextNode(
                id="header1",
                line=1,
                type="paragraph",  # type: ignore
                header="Level 1 Header",
                content="Anime recommendations",
                meta={},
                chunk_index=0
            ),
            TextNode(
                id="child1",
                line=2,
                type="code",  # type: ignore
                header="Code Child",
                content="print('Isekai anime list')",
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
                content="| Anime | Genre |\n|-------|-------|\n| Sword Art | Isekai |",
                meta={"header": ["Anime", "Genre"],
                      "rows": [["Sword Art", "Isekai"]]},
                parent_id="header1",
                parent_header="Level 1 Header",
                chunk_index=0
            )
        ]
        query = "isekai anime"
        vector_store = prepare_for_rag(nodes, model="all-MiniLM-L6-v2")
        expected_top_k = 3
        expected_relevant_ids = {"header1", "child2"}

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=expected_top_k)

        # Then
        assert len(results) <= expected_top_k
        assert all(isinstance(node, TextNode) and isinstance(score, float)
                   for node, score in results)
        assert any(node.id in expected_relevant_ids for node, score in results)
        # Adjusted threshold
        assert any(score > 0.4 for node,
                   score in results if node.id in expected_relevant_ids)

    def test_search_unsorted_hierarchy(self):
        # Given
        nodes = [
            TextNode(
                id="header3",
                line=3,
                type="paragraph",  # type: ignore
                header="Level 3 Header",
                content="Deep nested anime discussion",
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
                content="Anime overview",
                meta={},
                chunk_index=0
            ),
            TextNode(
                id="header2",
                line=2,
                type="paragraph",  # type: ignore
                header="Level 2 Header",
                content="Isekai anime trends",
                meta={},
                parent_id="header1",
                parent_header="Level 1 Header",
                chunk_index=0
            )
        ]
        query = "isekai trends"
        vector_store = prepare_for_rag(nodes, model="all-MiniLM-L6-v2")
        expected_top_k = 3
        expected_relevant_id = "header2"

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=expected_top_k)

        # Then
        assert len(results) <= expected_top_k
        assert all(isinstance(node, TextNode) and isinstance(score, float)
                   for node, score in results)
        assert any(node.id == expected_relevant_id for node, score in results)
        assert any(score > 0.5 for node,
                   score in results if node.id == expected_relevant_id)

    def test_search_with_empty_content(self):
        # Given
        nodes = [
            TextNode(
                id="node1",
                line=1,
                type="paragraph",  # type: ignore
                header="Empty Header",
                content="",
                meta={},
                chunk_index=0
            )
        ]
        query = "test query"
        vector_store = prepare_for_rag(nodes, model="all-MiniLM-L6-v2")
        expected_top_k = 1

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=expected_top_k)

        # Then
        assert len(results) <= expected_top_k
        assert all(isinstance(node, TextNode) and isinstance(score, float)
                   for node, score in results)
        assert all(score < 0.3 for _, score in results)

    def test_search_with_different_meta_types(self):
        # Given
        nodes = [
            TextNode(
                id="node1",
                line=1,
                type="unordered_list",  # type: ignore
                header="List Header",
                content="Item 1\nItem 2",
                meta={"items": [{"text": "Item 1", "task_item": False}, {
                    "text": "Item 2", "task_item": False}]},
                chunk_index=0
            ),
            TextNode(
                id="node2",
                line=2,
                type="code",  # type: ignore
                header="Code Header",
                content="print('Hello')",
                meta={"language": "python"},
                chunk_index=0
            ),
            TextNode(
                id="node3",
                line=3,
                type="table",  # type: ignore
                header="Table Header",
                content="| Col1 | Col2 |\n|------|------|\n| A    | B    |",
                meta={"header": ["Col1", "Col2"], "rows": [["A", "B"]]},
                chunk_index=0
            )
        ]
        query = "list items"
        vector_store = prepare_for_rag(nodes, model="all-MiniLM-L6-v2")
        expected_top_k = 3
        expected_relevant_id = "node1"

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=expected_top_k)

        # Then
        assert len(results) <= expected_top_k
        assert all(isinstance(node, TextNode) and isinstance(score, float)
                   for node, score in results)
        assert any(node.id == expected_relevant_id for node, score in results)
        assert any(score > 0.5 for node,
                   score in results if node.id == expected_relevant_id)

    def test_search_deeply_nested_hierarchy(self):
        # Given
        nodes = [
            TextNode(
                id="header1",
                line=1,
                type="paragraph",  # type: ignore
                header="Level 1 Header",
                content="Anime overview",
                meta={},
                chunk_index=0
            ),
            TextNode(
                id="header2",
                line=2,
                type="paragraph",  # type: ignore
                header="Level 2 Header",
                content="Isekai anime trends",
                meta={},
                parent_id="header1",
                parent_header="Level 1 Header",
                chunk_index=0
            ),
            TextNode(
                id="header3",
                line=3,
                type="paragraph",  # type: ignore
                header="Level 3 Header",
                content="Detailed isekai analysis",
                meta={},
                parent_id="header2",
                parent_header="Level 2 Header",
                chunk_index=0
            ),
            TextNode(
                id="child1",
                line=4,
                type="unordered_list",  # type: ignore
                header="List Child",
                content="Isekai anime 1\nIsekai anime 2",
                meta={"items": [{"text": "Isekai anime 1", "task_item": False}, {
                    "text": "Isekai anime 2", "task_item": False}]},
                parent_id="header3",
                parent_header="Level 3 Header",
                chunk_index=0
            )
        ]
        query = "isekai analysis"
        vector_store = prepare_for_rag(nodes, model="all-MiniLM-L6-v2")
        expected_top_k = 4
        expected_relevant_ids = {"header3", "child1"}

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=expected_top_k)

        # Then
        assert len(results) <= expected_top_k
        assert all(isinstance(node, TextNode) and isinstance(score, float)
                   for node, score in results)
        assert any(node.id in expected_relevant_ids for node, score in results)
        assert any(score > 0.5 for node,
                   score in results if node.id in expected_relevant_ids)

    def test_search_oversized_content(self):
        # Given
        oversized_content = "Isekai anime content " * 100
        nodes = [
            TextNode(
                id="node1",
                line=1,
                type="paragraph",  # type: ignore
                header="Oversized Header",
                content=oversized_content,
                meta={},
                chunk_index=0
            )
        ]
        query = "isekai anime"
        vector_store = prepare_for_rag(nodes, model="all-MiniLM-L6-v2")
        expected_top_k = 1
        expected_relevant_id = "node1"

        # When
        results = search_headers(
            query, vector_store, model="all-MiniLM-L6-v2", top_k=expected_top_k)

        # Then
        assert len(results) == 1
        assert all(isinstance(node, TextNode) and isinstance(score, float)
                   for node, score in results)
        assert results[0][0].id == expected_relevant_id
        assert results[0][1] > 0.5
