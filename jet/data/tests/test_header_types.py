import pytest
from typing import List, Optional
from jet.code.markdown_types import ContentType
from jet.data.header_types import Node, TextNode, HeaderNode, NodeType
from jet.data.utils import generate_unique_id
from jet.data.header_utils._base import process_node, create_text_node
from jet.models.tokenizer.base import get_tokenizer
from jet.logger import logger


@pytest.fixture
def sample_nodes() -> List[NodeType]:
    """Provide a sample hierarchy of nodes for testing."""
    header1 = HeaderNode(
        id="header1",
        line=1,
        type="header",
        header="Main Header",
        content="Main content",
        level=1,
        children=[],
        doc_id="doc1"
    )
    text1 = TextNode(
        id="text1",
        line=2,
        type="paragraph",
        header="Text Header",
        content="Text content",
        meta={},
        parent_id="header1",
        parent_header="Main Header",
        _parent_node=header1,
        doc_id="doc1"
    )
    header2 = HeaderNode(
        id="header2",
        line=3,
        type="header",
        header="Sub Header",
        content="Sub content",
        level=2,
        parent_id="header1",
        parent_header="Main Header",
        children=[],
        _parent_node=header1,
        doc_id="doc1"
    )
    header1.children.extend([text1, header2])
    return [header1, text1, header2]


class TestNodeGetText:
    def test_get_text_with_header_and_content(self) -> None:
        """Test get_text with both header and content."""
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="Test Content",
            meta={}
        )
        expected = "Test Header\nTest Content"
        result = node.get_text()
        assert result == expected

    def test_get_text_empty_header(self) -> None:
        """Test get_text with empty header."""
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="",
            content="Test Content",
            meta={}
        )
        expected = "Test Content"
        result = node.get_text()
        assert result == expected

    def test_get_text_empty_content(self) -> None:
        """Test get_text with empty content."""
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="",
            meta={}
        )
        expected = "Test Header"
        result = node.get_text()
        assert result == expected

    def test_get_text_empty_both(self) -> None:
        """Test get_text with both header and content empty."""
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="",
            content="",
            meta={}
        )
        expected = ""
        result = node.get_text()
        assert result == expected


class TestNodeGetParentHeaders:
    def test_get_parent_headers_with_parents(self, sample_nodes: List[NodeType]) -> None:
        """Test get_parent_headers with multiple parent levels."""
        node = sample_nodes[2]
        expected = ["Main Header"]
        result = node.get_parent_headers()
        assert result == expected

    def test_get_parent_headers_no_parent(self) -> None:
        """Test get_parent_headers with no parent."""
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="Test Content",
            meta={}
        )
        expected = []
        result = node.get_parent_headers()
        assert result == expected

    def test_get_parent_headers_multiple_levels(self) -> None:
        """Test get_parent_headers with multiple hierarchy levels."""
        header1 = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Root Header",
            content="Root content",
            level=1
        )
        header2 = HeaderNode(
            id="header2",
            line=2,
            type="header",
            header="Level 2 Header",
            content="Level 2 content",
            level=2,
            parent_id="header1",
            parent_header="Root Header",
            _parent_node=header1
        )
        text_node = TextNode(
            id="text1",
            line=3,
            type="paragraph",
            header="Text Header",
            content="Text content",
            meta={},
            parent_id="header2",
            parent_header="Level 2 Header",
            _parent_node=header2
        )
        header1.children = [header2]
        header2.children = [text_node]
        logger.debug(
            f"Verifying parent setup: text_node._parent_node.id={text_node._parent_node.id if text_node._parent_node else None}")
        logger.debug(
            f"header2._parent_node.id={header2._parent_node.id if header2._parent_node else None}")
        expected = ["Root Header", "Level 2 Header"]
        result = text_node.get_parent_headers()
        assert result == expected


class TestHeaderNodeGetRecursiveText:
    def test_get_recursive_text_single_header(self) -> None:
        """Test get_recursive_text with single header node."""
        node = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Main Header",
            content="Main content",
            level=1
        )
        expected = "Main Header\nMain content"
        result = node.get_recursive_text()
        assert result == expected

    def test_get_recursive_text_with_children(self, sample_nodes: List[NodeType]) -> None:
        """Test get_recursive_text with header and children."""
        node = sample_nodes[0]
        expected = (
            "Main Header\nMain content\n\n"
            "Text Header\nText content\n\n"
            "Sub Header\nSub content"
        )
        result = node.get_recursive_text()
        assert result == expected

    def test_get_recursive_text_empty_content(self) -> None:
        """Test get_recursive_text with empty content nodes."""
        header = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Main Header",
            content="",
            level=1,
            children=[
                TextNode(
                    id="text1",
                    line=2,
                    type="paragraph",
                    header="",
                    content="",
                    meta={}
                )
            ]
        )
        expected = "Main Header"
        result = header.get_recursive_text()
        assert result == expected

    def test_get_recursive_text_nested_headers(self) -> None:
        """Test get_recursive_text with nested headers."""
        header1 = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Root Header",
            content="Root content",
            level=1
        )
        header2 = HeaderNode(
            id="header2",
            line=2,
            type="header",
            header="Level 2 Header",
            content="Level 2 content",
            level=2,
            parent_id="header1",
            parent_header="Root Header",
            _parent_node=header1
        )
        text_node = TextNode(
            id="text1",
            line=3,
            type="paragraph",
            header="Text Header",
            content="Text content",
            meta={},
            parent_id="header2",
            parent_header="Level 2 Header",
            _parent_node=header2
        )
        header1.children = [header2]
        header2.children = [text_node]
        expected = (
            "Root Header\nRoot content\n\n"
            "Level 2 Header\nLevel 2 content\n\n"
            "Text Header\nText content"
        )
        result = header1.get_recursive_text()
        assert result == expected


class TestProcessNodeDocId:
    def test_process_node_single_chunk_preserve_doc_id(self, sample_nodes: List[NodeType]) -> None:
        """Test that process_node preserves doc_id for single chunk."""
        # Given a node with a specific doc_id
        node = sample_nodes[1]  # TextNode with doc_id="doc1"
        tokenizer = get_tokenizer("sentencepiece")
        chunk_size = 1000  # Large enough to avoid chunking
        chunk_overlap = 50
        buffer = 10

        # When processing the node
        result = process_node(node, tokenizer, chunk_size,
                              chunk_overlap, buffer)

        # Then the resulting node should have the same doc_id
        expected = ["doc1"]
        result_doc_ids = [n.doc_id for n in result]
        assert result_doc_ids == expected, (
            f"Expected doc_ids {expected}, but got {result_doc_ids}"
        )

    def test_process_node_multiple_chunks_preserve_doc_id(self, sample_nodes: List[NodeType]) -> None:
        """Test that process_node preserves doc_id across multiple chunks."""
        # Given a node with a specific doc_id and content that will be chunked
        node = TextNode(
            id="text1",
            line=1,
            type="paragraph",
            header="Test Header",
            content=" ".join(["word"] * 200),  # Long content to force chunking
            meta={},
            doc_id="doc1"
        )
        tokenizer = get_tokenizer("sentencepiece")
        chunk_size = 50  # Small enough to force multiple chunks
        chunk_overlap = 10
        buffer = 5

        # When processing the node
        result = process_node(node, tokenizer, chunk_size,
                              chunk_overlap, buffer)

        # Then all resulting nodes should have the same doc_id
        expected = ["doc1"] * len(result)
        result_doc_ids = [n.doc_id for n in result]
        assert result_doc_ids == expected, (
            f"Expected doc_ids {expected}, but got {result_doc_ids}"
        )
        assert len(result) > 1, "Expected multiple chunks"

    def test_process_header_node_with_children_preserve_doc_id(self, sample_nodes: List[NodeType]) -> None:
        """Test that process_node preserves doc_id for header node and its children."""
        # Given a header node with children, all with the same doc_id
        node = sample_nodes[0]  # HeaderNode with children and doc_id="doc1"
        tokenizer = get_tokenizer("sentencepiece")
        chunk_size = 1000
        chunk_overlap = 50
        buffer = 10

        # When processing the node
        result = process_node(node, tokenizer, chunk_size,
                              chunk_overlap, buffer)

        # Then all resulting nodes should have the same doc_id
        expected = ["doc1"] * len(result)
        result_doc_ids = [n.doc_id for n in result]
        assert result_doc_ids == expected, (
            f"Expected doc_ids {expected}, but got {result_doc_ids}"
        )

    def test_process_node_chunked_header_preserve_doc_id(self) -> None:
        """Test that process_node preserves doc_id when header node is chunked."""
        # Given a header node with long content that will be chunked
        node = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Main Header",
            content=" ".join(["word"] * 200),  # Long content to force chunking
            level=1,
            children=[],
            doc_id="doc1"
        )
        tokenizer = get_tokenizer("sentencepiece")
        chunk_size = 50  # Small enough to force multiple chunks
        chunk_overlap = 10
        buffer = 5

        # When processing the node
        result = process_node(node, tokenizer, chunk_size,
                              chunk_overlap, buffer)

        # Then all resulting nodes should have the same doc_id
        expected = ["doc1"] * len(result)
        result_doc_ids = [n.doc_id for n in result]
        assert result_doc_ids == expected, (
            f"Expected doc_ids {expected}, but got {result_doc_ids}"
        )
        assert len(result) > 1, "Expected multiple chunks"
