import pytest
from typing import List, Optional
from jet.data.header_types import HeaderNode, TextNode, NodeType
from jet.data.header_utils._base import process_node, chunk_content, create_text_node
from jet.models.tokenizer.base import get_tokenizer, count_tokens
from jet.code.markdown_types import ContentType
from tokenizers import Tokenizer


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Provide a tokenizer for tests."""
    return get_tokenizer("all-MiniLM-L6-v2")


class TestProcessNode:
    def test_single_header_node_no_chunking(self, tokenizer):
        # Given: A single HeaderNode with content that doesn't need chunking
        node = HeaderNode(
            id="header1",
            doc_id="doc1",
            line=1,
            type="header",
            header="Test Header",
            content="Short content",
            level=1
        )
        chunk_size = 1000
        chunk_overlap = 50
        buffer = 10

        # When: Processing the node
        result = process_node(node, tokenizer, chunk_size,
                              chunk_overlap, buffer)

        # Then: Expect one TextNode with same doc_id
        expected = [
            TextNode(
                # ID will be generated, so we check other fields
                id=result[0].id,
                doc_id="doc1",
                line=1,
                type="header",
                header="Test Header",
                content="Test Header\nShort content",
                chunk_index=0,
                num_tokens=None  # To be calculated later
            )
        ]
        expected[0].num_tokens = count_tokens(
            tokenizer, expected[0].get_text())

        assert len(result) == 1
        assert result[0].doc_id == expected[0].doc_id
        assert result[0].header == expected[0].header
        assert result[0].content == expected[0].content
        assert result[0].chunk_index == expected[0].chunk_index
        assert result[0].num_tokens == expected[0].num_tokens

    def test_header_node_with_chunking(self, tokenizer):
        # Given: A HeaderNode with content requiring chunking
        long_content = "Word " * 200  # Creates content that needs splitting
        node = HeaderNode(
            id="header2",
            doc_id="doc2",
            line=1,
            type="header",
            header="Long Header",
            content=long_content,
            level=1
        )
        chunk_size = 50
        chunk_overlap = 10
        buffer = 5

        # When: Processing the node
        result = process_node(node, tokenizer, chunk_size,
                              chunk_overlap, buffer)

        # Then: Expect multiple TextNodes with same doc_id
        assert len(result) > 1
        expected_doc_id = "doc2"
        for i, node in enumerate(result):
            assert node.doc_id == expected_doc_id
            assert node.header == "Long Header"
            assert node.chunk_index == i
            assert node.num_tokens <= chunk_size - buffer
            assert "Word" in node.content  # Ensure content chunks are non-empty

    def test_text_node_with_chunking(self, tokenizer):
        # Given: A TextNode with content requiring chunking
        long_content = "Paragraph " * 150
        node = TextNode(
            id="text1",
            doc_id="doc3",
            line=1,
            type="paragraph",
            header="Text Header",
            content=long_content
        )
        chunk_size = 50
        chunk_overlap = 10
        buffer = 5

        # When: Processing the node
        result = process_node(node, tokenizer, chunk_size,
                              chunk_overlap, buffer)

        # Then: Expect multiple TextNodes with same doc_id
        assert len(result) > 1
        expected_doc_id = "doc3"
        for i, node in enumerate(result):
            assert node.doc_id == expected_doc_id
            assert node.header == "Text Header"
            assert node.chunk_index == i
            assert node.num_tokens <= chunk_size - buffer
            assert "Paragraph" in node.content

    def test_node_with_children(self, tokenizer):
        # Given: A HeaderNode with children
        parent = HeaderNode(
            id="parent1",
            doc_id="doc4",
            line=1,
            type="header",
            header="Parent Header",
            content="Parent content",
            level=1,
            children=[
                TextNode(
                    id="child1",
                    doc_id="doc4",
                    line=2,
                    type="paragraph",
                    header="Child Header",
                    content="Child content"
                )
            ]
        )
        chunk_size = 1000
        chunk_overlap = 50
        buffer = 10

        # When: Processing the node
        result = process_node(node=parent, tokenizer=tokenizer, chunk_size=chunk_size,
                              chunk_overlap=chunk_overlap, buffer=buffer)

        # Then: Expect TextNodes for parent and child with same doc_id
        expected = [
            TextNode(
                id=result[0].id,
                doc_id="doc4",
                line=1,
                type="header",
                header="Parent Header",
                content="Parent Header\nParent content",
                chunk_index=0
            ),
            TextNode(
                id=result[1].id,
                doc_id="doc4",
                line=2,
                type="paragraph",
                header="Child Header",
                content="Child Header\nChild content",
                chunk_index=0
            )
        ]
        assert len(result) == 2
        for i, node in enumerate(result):
            assert node.doc_id == expected[i].doc_id
            assert node.header == expected[i].header
            assert node.content == expected[i].content
            assert node.chunk_index == expected[i].chunk_index

    def test_empty_content_node(self, tokenizer):
        # Given: A TextNode with empty content
        node = TextNode(
            id="text2",
            doc_id="doc5",
            line=1,
            type="paragraph",
            header="Empty Header",
            content=""
        )
        chunk_size = 50
        chunk_overlap = 10
        buffer = 5

        # When: Processing the node
        result = process_node(node, tokenizer, chunk_size,
                              chunk_overlap, buffer)

        # Then: Expect single TextNode with same doc_id and empty content
        expected = [
            TextNode(
                id=node.id,
                doc_id="doc5",
                line=1,
                type="paragraph",
                header="Empty Header",
                content="",
                chunk_index=0,
                num_tokens=0
            )
        ]
        assert len(result) == 1
        assert result[0].doc_id == expected[0].doc_id
        assert result[0].header == expected[0].header
        assert result[0].content == expected[0].content
        assert result[0].num_tokens == expected[0].num_tokens
