import pytest
from typing import List, Optional
from jet.code.markdown_types import ContentType, MetaType
from jet.data.header_types import HeaderNode, TextNode, NodeType, Nodes
from jet.data.utils import generate_unique_id
from jet.data.header_utils import create_text_node, chunk_content, process_node
from jet.models.tokenizer.base import get_tokenizer
from tokenizers import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Provide a tokenizer for tests."""
    return get_tokenizer("all-MiniLM-L6-v2")


@pytest.fixture
def default_params() -> dict:
    """Provide default parameters for chunking functions."""
    return {"chunk_size": 100, "chunk_overlap": 0, "buffer": 0}


class TestCreateTextNode:
    def test_create_text_node_from_text_node(self, default_params: dict) -> None:
        """Test creating a TextNode from a TextNode with correct attributes."""
        # Given
        input_node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="Test Content",
            meta={"key": "value"},
            chunk_index=0,
            num_tokens=0
        )
        expected = TextNode(
            id=generate_unique_id(),
            line=1,
            type="paragraph",
            header="Test Header",
            content="Test Header\nTest Content",
            meta={"key": "value"},
            parent_id="parent1",
            parent_header="Parent Header",
            chunk_index=1,
            num_tokens=0
        )

        # When
        result = create_text_node(
            node=input_node,
            content=f"{input_node.header}\n{input_node.content}",
            chunk_index=1,
            parent_id="parent1",
            parent_header="Parent Header"
        )

        # Then
        assert result.type == expected.type
        assert result.header == expected.header
        assert result.content == expected.content
        assert result.line == expected.line
        assert result.meta == expected.meta
        assert result.parent_id == expected.parent_id
        assert result.parent_header == expected.parent_header
        assert result.chunk_index == expected.chunk_index

    def test_create_text_node_from_header_node(self, default_params: dict) -> None:
        """Test creating a TextNode from a HeaderNode with correct attributes."""
        # Given
        input_node = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Main Header",
            content="Header Content",
            level=1,
            children=[],
            chunk_index=0,
            num_tokens=0
        )
        expected = TextNode(
            id=generate_unique_id(),
            line=1,
            type="paragraph",
            header="Main Header",
            content="Main Header\nHeader Content",
            meta=None,
            parent_id="parent1",
            parent_header="Parent Header",
            chunk_index=0,
            num_tokens=0
        )

        # When
        result = create_text_node(
            node=input_node,
            content=f"{input_node.header}\n{input_node.content}",
            chunk_index=0,
            parent_id="parent1",
            parent_header="Parent Header"
        )

        # Then
        assert result.type == expected.type
        assert result.header == expected.header
        assert result.content == expected.content
        assert result.line == expected.line
        assert result.meta == expected.meta
        assert result.parent_id == expected.parent_id
        assert result.parent_header == expected.parent_header
        assert result.chunk_index == expected.chunk_index


class TestChunkContent:
    def test_chunk_content_no_chunking(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test chunk_content with content that doesn't require chunking."""
        # Given
        content = "Short content."
        expected = [content]

        # When
        result = chunk_content(content, tokenizer, **default_params)

        # Then
        assert result == expected

    def test_chunk_content_with_chunking(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test chunk_content with content that requires chunking."""
        # Given
        content = "This is a long sentence. " * 20
        params = default_params | {"chunk_size": 50,
                                   "chunk_overlap": 10, "buffer": 5}
        expected_chunk_count = 3

        # When
        result = chunk_content(content, tokenizer, **params)

        # Then
        assert len(result) >= expected_chunk_count
        for chunk in result:
            token_ids = tokenizer.encode(chunk, add_special_tokens=False)
            token_ids = [tid for tid in token_ids if tid != 0]
            assert len(token_ids) <= params["chunk_size"] - params["buffer"]

    def test_chunk_content_empty(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test chunk_content with empty content."""
        # Given
        content = ""
        expected = []

        # When
        result = chunk_content(content, tokenizer, **default_params)

        # Then
        assert result == expected


class TestProcessNode:
    def test_process_text_node_no_chunking(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing a text node without chunking."""
        # Given
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Test Header",
            content="Short content.",
            meta=None,
            chunk_index=0,
            num_tokens=0
        )
        content = "Test Header\nShort content."
        token_ids = tokenizer.encode(content, add_special_tokens=False)
        token_ids = [tid for tid in token_ids if tid != 0]
        expected = [TextNode(
            id=generate_unique_id(),
            line=1,
            type="paragraph",
            header="Test Header",
            content=content,
            meta=None,
            chunk_index=0,
            num_tokens=len(token_ids)
        )]

        # When
        result = process_node(node, tokenizer, **default_params)

        # Then
        assert len(result) == 1
        assert result[0].type == expected[0].type
        assert result[0].header == expected[0].header
        assert result[0].content == expected[0].content
        assert result[0].line == expected[0].line
        assert result[0].meta == expected[0].meta
        assert result[0].chunk_index == expected[0].chunk_index
        assert result[0].num_tokens == expected[0].num_tokens

    def test_process_header_node_with_children(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing a header node with children."""
        # Given
        node = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Main Header",
            content="Header content",
            level=1,
            children=[
                TextNode(
                    id="child1",
                    line=2,
                    type="paragraph",
                    header="Child Header",
                    content="Child content.",
                    meta=None,
                    parent_id="header1",
                    parent_header="Main Header",
                    chunk_index=0,
                    num_tokens=0
                )
            ],
            chunk_index=0,
            num_tokens=0
        )
        expected = [
            TextNode(
                id=generate_unique_id(),
                line=1,
                type="paragraph",
                header="Main Header",
                content="Main Header\nHeader content",
                meta=None,
                chunk_index=0,
                num_tokens=len([tid for tid in tokenizer.encode(
                    "Main Header\nHeader content", add_special_tokens=False) if tid != 0])
            ),
            TextNode(
                id=generate_unique_id(),
                line=2,
                type="paragraph",
                header="Child Header",
                content="Child Header\nChild content.",
                meta=None,
                parent_id="header1",
                parent_header="Main Header",
                chunk_index=0,
                num_tokens=len([tid for tid in tokenizer.encode(
                    "Child Header\nChild content.", add_special_tokens=False) if tid != 0])
            )
        ]

        # When
        result = process_node(node, tokenizer, **default_params)

        # Then
        assert len(result) == 2
        for res, exp in zip(result, expected):
            assert res.type == exp.type
            assert res.header == exp.header
            assert res.content == exp.content
            assert res.line == exp.line
            assert res.meta == exp.meta
            assert res.parent_id == exp.parent_id
            assert res.parent_header == exp.parent_header
            assert res.chunk_index == exp.chunk_index
            assert res.num_tokens == exp.num_tokens

    def test_process_empty_content_node(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing a text node with empty content."""
        # Given
        node = TextNode(
            id="node1",
            line=1,
            type="paragraph",
            header="Empty Header",
            content="",
            meta=None,
            chunk_index=0,
            num_tokens=0
        )
        expected = [node]

        # When
        result = process_node(node, tokenizer, **default_params)

        # Then
        assert len(result) == 1
        assert result[0].type == expected[0].type
        assert result[0].header == expected[0].header
        assert result[0].content == expected[0].content
        assert result[0].line == expected[0].line
        assert result[0].meta == expected[0].meta
        assert result[0].chunk_index == expected[0].chunk_index
        assert result[0].num_tokens == 0

    def test_process_header_node_with_chunking(self, tokenizer: Tokenizer, default_params: dict) -> None:
        """Test processing a header node with content requiring chunking."""
        # Given
        content = "This is a long sentence. " * 20
        node = HeaderNode(
            id="header1",
            line=1,
            type="header",
            header="Main Header",
            content=content,
            level=1,
            children=[],
            chunk_index=0,
            num_tokens=0
        )
        params = default_params | {"chunk_size": 50,
                                   "chunk_overlap": 10, "buffer": 5}
        expected_chunk_count = 3

        # When
        result = process_node(node, tokenizer, **params)

        # Then
        assert len(result) >= expected_chunk_count
        for i, node in enumerate(result):
            assert node.header == "Main Header"
            assert node.content.startswith("Main Header\n")
            assert node.type == "paragraph"
            assert node.line == 1
            assert node.chunk_index == i
            token_ids = tokenizer.encode(
                node.content, add_special_tokens=False)
            token_ids = [tid for tid in token_ids if tid != 0]
            assert node.num_tokens == len(token_ids)
            assert node.num_tokens <= params["chunk_size"] - params["buffer"]
            assert node.num_tokens > 0
