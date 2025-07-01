import pytest
from typing import List, Optional
from jet.code.markdown_types import ContentType, MetaType
from jet.data.header_types import HeaderNode, TextNode, NodeType, Nodes
from jet.data.header_utils import split_and_merge_headers
from jet.models.tokenizer.base import get_tokenizer
from tokenizers import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Provide a BERT tokenizer for tests."""
    return get_tokenizer("bert-base-uncased")


@pytest.fixture
def default_params() -> dict:
    """Provide default parameters for split_and_merge_headers."""
    return {"chunk_size": 100, "chunk_overlap": 0, "buffer": 0}


def assert_nodes_equal(result: TextNode, expected: TextNode) -> None:
    """Assert that two nodes have identical attributes."""
    assert result.type == expected.type
    assert result.header == expected.header
    assert result.content == expected.content
    assert result.line == expected.line
    assert result.meta == expected.meta
    assert result.parent_id == expected.parent_id
    assert result.parent_header == expected.parent_header
    assert result.chunk_index == expected.chunk_index


class TestSplitAndMergeHeaders:
    def test_single_text_node_no_chunking(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing a single text node without chunking."""
        # Given
        node = TextNode(id="node1", line=1, type="paragraph", header="Test Header",
                        content="Short content.", meta=None, chunk_index=0)
        expected = [TextNode(id="node1", line=1, type="paragraph", header="Test Header",
                             content="Test Header\nShort content.", meta=None, chunk_index=0)]

        # When
        result = split_and_merge_headers(
            node, tokenizer=tokenizer, **default_params)

        # Then
        assert len(result) == 1
        assert_nodes_equal(result[0], expected[0])

    def test_single_text_node_with_chunking(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing a text node with chunking and overlap."""
        # Given
        content = "This is a long sentence. " * 20
        node = TextNode(id="node1", line=1, type="paragraph",
                        header="Long Content", content=content, meta=None, chunk_index=0)
        params = default_params | {"chunk_size": 50,
                                   "chunk_overlap": 10, "buffer": 5}
        expected_chunk_count = 3

        # When
        result = split_and_merge_headers(node, tokenizer=tokenizer, **params)

        # Then
        assert len(result) >= expected_chunk_count
        for i, node in enumerate(result):
            assert node.header == "Long Content"
            assert node.content.startswith("Long Content\n")
            assert node.type == "paragraph"
            assert node.line == 1
            assert node.meta is None
            assert node.chunk_index == i
            tokens = tokenizer.encode(
                node.content[len("Long Content\n"):], add_special_tokens=False).ids
            assert len(tokens) <= params["chunk_size"] - params["buffer"]

    def test_header_node_with_children(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing a header node with children."""
        # Given
        node = HeaderNode(
            id="header1", line=1, type="header", header="Main Header", content="Header content", level=1,
            children=[TextNode(id="child1", line=2, type="paragraph", header="Child Header", content="Child content.",
                               meta=None, parent_id="header1", parent_header="Main Header", chunk_index=0)]
        )
        expected = [
            TextNode(id="header1", line=1, type="paragraph", header="Main Header",
                     content="Main Header\nHeader content", meta=None, chunk_index=0),
            TextNode(id="child1", line=2, type="paragraph", header="Child Header", content="Child Header\nChild content.",
                     meta=None, parent_id="header1", parent_header="Main Header", chunk_index=0)
        ]

        # When
        result = split_and_merge_headers(
            node, tokenizer=tokenizer, **default_params)

        # Then
        assert len(result) == 2
        for res, exp in zip(result, expected):
            assert_nodes_equal(res, exp)

    def test_empty_content_node(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing an empty content node."""
        # Given
        node = TextNode(id="node1", line=1, type="paragraph",
                        header="Empty Header", content="", meta=None, chunk_index=0)
        expected = [node]

        # When
        result = split_and_merge_headers(
            node, tokenizer=tokenizer, **default_params)

        # Then
        assert len(result) == 1
        assert_nodes_equal(result[0], expected[0])

    def test_complex_nested_headers(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing complex nested headers."""
        # Given
        nodes = [
            HeaderNode(
                id="header1", line=1, type="header", header="Level 1 Header", content="Level 1 content", level=1,
                children=[
                    HeaderNode(id="header2", line=2, type="header", header="Level 3 Header", content="Level 3 content", level=3, children=[
                        TextNode(id="child1", line=3, type="paragraph", header="Child Paragraph", content="Paragraph content.",
                                 meta=None, parent_id="header2", parent_header="Level 3 Header", chunk_index=0)
                    ]),
                    HeaderNode(id="header3", line=4, type="header", header="Level 2 Header", content="Level 2 content", level=2, children=[
                        TextNode(id="child2", line=5, type="code", header="Child Code", content="print('Hello')", meta={
                                 "language": "python"}, parent_id="header3", parent_header="Level 2 Header", chunk_index=0)
                    ])
                ]
            )
        ]
        expected = [
            TextNode(id="header1", line=1, type="paragraph", header="Level 1 Header",
                     content="Level 1 Header\nLevel 1 content", meta=None, chunk_index=0),
            TextNode(id="header2", line=2, type="paragraph", header="Level 3 Header", content="Level 3 Header\nLevel 3 content",
                     meta=None, parent_id="header1", parent_header="Level 1 Header", chunk_index=0),
            TextNode(id="child1", line=3, type="paragraph", header="Child Paragraph", content="Child Paragraph\nParagraph content.",
                     meta=None, parent_id="header2", parent_header="Level 3 Header", chunk_index=0),
            TextNode(id="header3", line=4, type="paragraph", header="Level 2 Header", content="Level 2 Header\nLevel 2 content",
                     meta=None, parent_id="header1", parent_header="Level 1 Header", chunk_index=0),
            TextNode(id="child2", line=5, type="code", header="Child Code", content="Child Code\nprint('Hello')", meta={
                     "language": "python"}, parent_id="header3", parent_header="Level 2 Header", chunk_index=0)
        ]

        # When
        result = split_and_merge_headers(
            nodes, tokenizer=tokenizer, **default_params)

        # Then
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert_nodes_equal(res, exp)

    def test_multiple_children_mixed_types(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing a header with mixed-type children."""
        # Given
        nodes = [
            HeaderNode(
                id="header1", line=1, type="header", header="Main Header", content="Header content", level=1,
                children=[
                    TextNode(id="child1", line=2, type="paragraph", header="Paragraph Child", content="Paragraph content.",
                             meta=None, parent_id="header1", parent_header="Main Header", chunk_index=0),
                    TextNode(id="child2", line=3, type="code", header="Code Child", content="def func(): pass", meta={
                             "language": "python"}, parent_id="header1", parent_header="Main Header", chunk_index=0),
                    TextNode(id="child3", line=4, type="table", header="Table Child", content="| Col1 | Col2 |\n|------|------|\n| A    | B    |",
                             meta={"header": ["Col1", "Col2"], "rows": [["A", "B"]]}, parent_id="header1", parent_header="Main Header", chunk_index=0)
                ]
            )
        ]
        expected = [
            TextNode(id="header1", line=1, type="paragraph", header="Main Header",
                     content="Main Header\nHeader content", meta=None, chunk_index=0),
            TextNode(id="child1", line=2, type="paragraph", header="Paragraph Child", content="Paragraph Child\nParagraph content.",
                     meta=None, parent_id="header1", parent_header="Main Header", chunk_index=0),
            TextNode(id="child2", line=3, type="code", header="Code Child", content="Code Child\ndef func(): pass", meta={
                     "language": "python"}, parent_id="header1", parent_header="Main Header", chunk_index=0),
            TextNode(id="child3", line=4, type="table", header="Table Child", content="Table Child\n| Col1 | Col2 |\n|------|------|\n| A    | B    |",
                     meta={"header": ["Col1", "Col2"], "rows": [["A", "B"]]}, parent_id="header1", parent_header="Main Header", chunk_index=0)
        ]

        # When
        result = split_and_merge_headers(
            nodes, tokenizer=tokenizer, **default_params)

        # Then
        assert len(result) == len(expected)
        for res, exp in zip(result, expected):
            assert_nodes_equal(res, exp)

    def test_deeply_nested_with_chunking(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing deeply nested headers with chunking."""
        # Given
        content = "This is a long sentence. " * 20
        nodes = [
            HeaderNode(
                id="header1", line=1, type="header", header="Level 1 Header", content="Level 1 content", level=1,
                children=[
                    HeaderNode(id="header2", line=2, type="header", header="Level 2 Header", content="Level 2 content", level=2, children=[
                        HeaderNode(id="header3", line=3, type="header", header="Level 3 Header", content="Level 3 content", level=3, children=[
                            TextNode(id="child1", line=4, type="paragraph", header="Child Header", content=content,
                                     meta=None, parent_id="header3", parent_header="Level 3 Header", chunk_index=0)
                        ])
                    ])
                ]
            )
        ]
        params = default_params | {"chunk_size": 50,
                                   "chunk_overlap": 10, "buffer": 5}
        expected_chunk_count = 3

        # When
        result = split_and_merge_headers(nodes, tokenizer=tokenizer, **params)

        # Then
        header_nodes = [n for n in result if n.line in [1, 2, 3]]
        chunk_nodes = [n for n in result if n.line == 4]
        assert len(header_nodes) == 3
        assert len(chunk_nodes) >= expected_chunk_count
        assert header_nodes[0].header == "Level 1 Header"
        assert header_nodes[0].content == "Level 1 Header\nLevel 1 content"
        assert header_nodes[0].parent_id is None
        assert header_nodes[0].chunk_index == 0
        assert header_nodes[1].header == "Level 2 Header"
        assert header_nodes[1].parent_id == "header1"
        assert header_nodes[2].header == "Level 3 Header"
        assert header_nodes[2].parent_id == "header2"
        for i, node in enumerate(chunk_nodes):
            assert node.header == "Child Header"
            assert node.content.startswith("Child Header\n")
            assert node.type == "paragraph"
            assert node.line == 4
            assert node.parent_id == "header3"
            assert node.chunk_index == i
            tokens = tokenizer.encode(
                node.content[len("Child Header\n"):], add_special_tokens=False).ids
            assert len(tokens) <= params["chunk_size"] - params["buffer"]

    def test_edge_case_empty_children_and_oversized_header(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing a header with empty children and oversized content."""
        # Given
        content = "This is a long header content. " * 20
        nodes = [
            HeaderNode(
                id="header1", line=1, type="header", header="Main Header", content=content, level=1,
                children=[TextNode(id="child1", line=2, type="paragraph", header="Empty Child", content="",
                                   meta=None, parent_id="header1", parent_header="Main Header", chunk_index=0)]
            )
        ]
        params = default_params | {"chunk_size": 50,
                                   "chunk_overlap": 10, "buffer": 5}
        expected_chunk_count = 3

        # When
        result = split_and_merge_headers(nodes, tokenizer=tokenizer, **params)

        # Then
        header_nodes = [n for n in result if n.line == 1]
        empty_child = [n for n in result if n.line == 2]
        assert len(header_nodes) >= expected_chunk_count
        assert len(empty_child) == 1
        assert empty_child[0].header == "Empty Child"
        assert empty_child[0].content == ""
        assert empty_child[0].parent_id == "header1"
        for i, node in enumerate(header_nodes):
            assert node.header == "Main Header"
            assert node.content.startswith("Main Header\n")
            assert node.type == "paragraph"
            assert node.line == 1
            assert node.chunk_index == i
            tokens = tokenizer.encode(
                node.content[len("Main Header\n"):], add_special_tokens=False).ids
            assert len(tokens) <= params["chunk_size"] - params["buffer"]
