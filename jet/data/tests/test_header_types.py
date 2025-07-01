import pytest
from typing import List, Optional
from jet.code.markdown_types import ContentType
from jet.data.header_types import Node, TextNode, HeaderNode, NodeType
from jet.data.utils import generate_unique_id
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
        children=[]
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
        _parent_node=header1
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
        _parent_node=header1
    )
    header1.children.extend([text1, header2])
    return [header1, text1, header2]


class TestNodeGetText:
    def test_get_text_with_header_and_content(self) -> None:
        """Test get_text with both header and content."""
        # Given
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="Test Content",
            meta={}
        )
        expected = "Test Header\nTest Content"

        # When
        result = node.get_text()

        # Then
        assert result == expected

    def test_get_text_empty_header(self) -> None:
        """Test get_text with empty header."""
        # Given
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="",
            content="Test Content",
            meta={}
        )
        expected = "Test Content"

        # When
        result = node.get_text()

        # Then
        assert result == expected

    def test_get_text_empty_content(self) -> None:
        """Test get_text with empty content."""
        # Given
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="",
            meta={}
        )
        expected = "Test Header"

        # When
        result = node.get_text()

        # Then
        assert result == expected

    def test_get_text_empty_both(self) -> None:
        """Test get_text with both header and content empty."""
        # Given
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="",
            content="",
            meta={}
        )
        expected = ""

        # When
        result = node.get_text()

        # Then
        assert result == expected


class TestNodeGetParentHeaders:
    def test_get_parent_headers_with_parents(self, sample_nodes: List[NodeType]) -> None:
        """Test get_parent_headers with multiple parent levels."""
        # Given
        node = sample_nodes[2]
        expected = ["Main Header"]

        # When
        result = node.get_parent_headers()

        # Then
        assert result == expected

    def test_get_parent_headers_no_parent(self) -> None:
        """Test get_parent_headers with no parent."""
        # Given
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="Test Content",
            meta={}
        )
        expected = []

        # When
        result = node.get_parent_headers()

        # Then
        assert result == expected

    def test_get_parent_headers_multiple_levels(self) -> None:
        """Test get_parent_headers with multiple hierarchy levels."""
        # Given
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

        # When
        result = text_node.get_parent_headers()

        # Then
        assert result == expected


class TestHeaderNodeGetRecursiveText:
    def test_get_recursive_text_single_header(self) -> None:
        """Test get_recursive_text with single header node."""
        # Given
        node = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Main Header",
            content="Main content",
            level=1
        )
        expected = "Main Header\nMain content"

        # When
        result = node.get_recursive_text()

        # Then
        assert result == expected

    def test_get_recursive_text_with_children(self, sample_nodes: List[NodeType]) -> None:
        """Test get_recursive_text with header and children."""
        # Given
        node = sample_nodes[0]
        expected = (
            "Main Header\nMain content\n\n"
            "Text Header\nText content\n\n"
            "Sub Header\nSub content"
        )

        # When
        result = node.get_recursive_text()

        # Then
        assert result == expected

    def test_get_recursive_text_empty_content(self) -> None:
        """Test get_recursive_text with empty content nodes."""
        # Given
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

        # When
        result = header.get_recursive_text()

        # Then
        assert result == expected

    def test_get_recursive_text_nested_headers(self) -> None:
        """Test get_recursive_text with nested headers."""
        # Given
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

        # When
        result = header1.get_recursive_text()

        # Then
        assert result == expected
