import pytest
from typing import List, Optional
from jet.data.header_types import HeaderNode, TextNode, NodeType
from jet.data.header_utils._base import process_node, chunk_content, create_text_node
from jet.models.tokenizer.base import get_tokenizer, count_tokens
from jet.code.markdown_types import ContentType
from tokenizers import Tokenizer
from jet.models.model_types import ModelType


@pytest.fixture
def model_name_or_tokenizer() -> str:
    """Provide a model name for tests."""
    return "sentence-transformers/all-MiniLM-L6-v2"


class TestProcessNode:
    def test_single_header_node_no_chunking(self, model_name_or_tokenizer):
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
        expected = [
            TextNode(
                id="header1",  # Will be replaced by generated ID
                doc_id="doc1",
                line=1,
                type="header",
                header="Test Header",
                content="Test Header\nShort content",
                chunk_index=0,
                num_tokens=5  # Expected tokens based on tokenizer output
            )
        ]

        # When: Processing the node
        result = process_node(node, model_name_or_tokenizer,
                              chunk_size, chunk_overlap, buffer)

        # Then: Expect one TextNode with correct fields
        assert len(result) == 1, "Expected exactly one TextNode"
        assert result[0].doc_id == expected[0].doc_id, "Doc ID should match"
        assert result[0].header == expected[0].header, "Header should match"
        assert result[0].content == expected[0].content, "Content should match"
        assert result[0].chunk_index == expected[0].chunk_index, "Chunk index should be 0"
        assert result[0].num_tokens == expected[
            0].num_tokens, f"Expected {expected[0].num_tokens} tokens, got {result[0].num_tokens}"

    def test_header_node_with_chunking(self, model_name_or_tokenizer):
        # Given: A HeaderNode with content requiring chunking
        long_content = "Word " * 200  # Approx 200 tokens
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
        tokenizer = get_tokenizer(model_name_or_tokenizer)
        full_text = f"Long Header\n{long_content}"
        expected_token_count = len(tokenizer.encode(
            full_text, add_special_tokens=False).ids)
        expected_chunks = max(1, (expected_token_count + chunk_size - buffer - 1) //
                              # Account for header tokens
                              (chunk_size - buffer - 10))

        # When: Processing the node
        result = process_node(node, model_name_or_tokenizer,
                              chunk_size, chunk_overlap, buffer)

        # Then: Expect multiple TextNodes with same doc_id
        assert len(
            result) >= expected_chunks, f"Expected at least {expected_chunks} chunks, got {len(result)}"
        expected_doc_id = "doc2"
        for i, node in enumerate(result):
            assert node.doc_id == expected_doc_id, f"Chunk {i} doc_id should be {expected_doc_id}"
            assert node.header == "Long Header", f"Chunk {i} header should be Long Header"
            assert node.chunk_index == i, f"Chunk {i} index should be {i}"
            assert node.num_tokens <= chunk_size - \
                buffer, f"Chunk {i} num_tokens {node.num_tokens} exceeds limit {chunk_size - buffer}"
            assert "Word" in node.content, f"Chunk {i} content should contain 'Word'"

    def test_text_node_with_chunking(self, model_name_or_tokenizer):
        # Given: A TextNode with content requiring chunking
        long_content = "Paragraph " * 150  # Approx 150 tokens
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
        tokenizer = get_tokenizer(model_name_or_tokenizer)
        full_text = f"Text Header\n{long_content}"
        expected_token_count = len(tokenizer.encode(
            full_text, add_special_tokens=False).ids)
        expected_chunks = max(
            1, (expected_token_count + chunk_size - buffer - 1) // (chunk_size - buffer - 10))

        # When: Processing the node
        result = process_node(node, model_name_or_tokenizer,
                              chunk_size, chunk_overlap, buffer)

        # Then: Expect multiple TextNodes with same doc_id
        assert len(
            result) >= expected_chunks, f"Expected at least {expected_chunks} chunks, got {len(result)}"
        expected_doc_id = "doc3"
        for i, node in enumerate(result):
            assert node.doc_id == expected_doc_id, f"Chunk {i} doc_id should be {expected_doc_id}"
            assert node.header == "Text Header", f"Chunk {i} header should be Text Header"
            assert node.chunk_index == i, f"Chunk {i} index should be {i}"
            assert node.num_tokens <= chunk_size - \
                buffer, f"Chunk {i} num_tokens {node.num_tokens} exceeds limit {chunk_size - buffer}"
            assert "Paragraph" in node.content, f"Chunk {i} content should contain 'Paragraph'"

    def test_node_with_children(self, model_name_or_tokenizer):
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
        expected = [
            TextNode(
                id="parent1",  # Will be replaced by generated ID
                doc_id="doc4",
                line=1,
                type="header",
                header="Parent Header",
                content="Parent Header\nParent content",
                chunk_index=0,
                num_tokens=5  # Expected based on tokenizer
            ),
            TextNode(
                id="child1",  # Will be replaced by generated ID
                doc_id="doc4",
                line=2,
                type="paragraph",
                header="Child Header",
                content="Child Header\nChild content",
                chunk_index=0,
                num_tokens=5  # Expected based on tokenizer
            )
        ]

        # When: Processing the node
        result = process_node(node=parent, model_name_or_tokenizer=model_name_or_tokenizer, chunk_size=chunk_size,
                              chunk_overlap=chunk_overlap, buffer=buffer)

        # Then: Expect TextNodes for parent and child with same doc_id
        assert len(result) == 2, f"Expected 2 nodes, got {len(result)}"
        for i, node in enumerate(result):
            assert node.doc_id == expected[i].doc_id, f"Node {i} doc_id should be {expected[i].doc_id}"
            assert node.header == expected[i].header, f"Node {i} header should be {expected[i].header}"
            assert node.content == expected[i].content, f"Node {i} content should be {expected[i].content}"
            assert node.chunk_index == expected[
                i].chunk_index, f"Node {i} chunk_index should be {expected[i].chunk_index}"
            assert node.num_tokens == expected[
                i].num_tokens, f"Node {i} num_tokens should be {expected[i].num_tokens}"

    def test_empty_content_node(self, model_name_or_tokenizer):
        # Given: A TextNode with empty content
        node = TextNode(
            id="text2",
            doc_id="doc5",
            line=1,
            type="paragraph",
            header="",
            content=""
        )
        chunk_size = 50
        chunk_overlap = 10
        buffer = 5
        expected = [
            TextNode(
                id="text2",
                doc_id="doc5",
                line=1,
                type="paragraph",
                header="",
                content="",
                chunk_index=0,
                num_tokens=0
            )
        ]

        # When: Processing the node
        result = process_node(node, model_name_or_tokenizer,
                              chunk_size, chunk_overlap, buffer)

        # Then: Expect single TextNode with empty content and zero tokens
        assert len(result) == 1, f"Expected 1 node, got {len(result)}"
        assert result[0].doc_id == expected[0].doc_id, "Doc ID should match"
        assert result[0].header == expected[0].header, "Header should be empty"
        assert result[0].content == expected[0].content, "Content should be empty"
        assert result[0].num_tokens == expected[0].num_tokens, "Num tokens should be 0"

    def test_node_without_header(self, model_name_or_tokenizer):
        # Given: A TextNode with content but no header
        node = TextNode(
            id="text3",
            doc_id="doc6",
            line=1,
            type="paragraph",
            header="",
            content="Content without header"
        )
        chunk_size = 1000
        chunk_overlap = 50
        buffer = 10
        expected = [
            TextNode(
                id="text3",  # Will be replaced by generated ID
                doc_id="doc6",
                line=1,
                type="paragraph",
                header="",
                content="Content without header",
                chunk_index=0,
                num_tokens=4  # Expected based on tokenizer
            )
        ]

        # When: Processing the node
        result = process_node(node, model_name_or_tokenizer,
                              chunk_size, chunk_overlap, buffer)

        # Then: Expect single TextNode with correct content and token count
        assert len(result) == 1, f"Expected 1 node, got {len(result)}"
        assert result[0].doc_id == expected[0].doc_id, "Doc ID should match"
        assert result[0].header == expected[0].header, "Header should be empty"
        assert result[0].content == expected[0].content, "Content should match"
        assert result[0].num_tokens == expected[
            0].num_tokens, f"Expected {expected[0].num_tokens} tokens, got {result[0].num_tokens}"

    def test_invalid_tokenizer(self):
        # Given: A HeaderNode with an invalid model name
        node = HeaderNode(
            id="header3",
            doc_id="doc7",
            line=1,
            type="header",
            header="Invalid Tokenizer Test",
            content="Test content",
            level=1
        )
        chunk_size = 1000
        chunk_overlap = 50
        buffer = 10
        invalid_model = "invalid-model-name"
        expected = [
            TextNode(
                id="header3",  # Will be replaced by generated ID
                doc_id="doc7",
                line=1,
                type="header",
                header="Invalid Tokenizer Test",
                content="Invalid Tokenizer Test\nTest content",
                chunk_index=0,
                num_tokens=0  # No valid tokenizer, so num_tokens should be 0
            )
        ]

        # When: Processing the node with an invalid tokenizer
        result = process_node(node, invalid_model,
                              chunk_size, chunk_overlap, buffer)

        # Then: Expect single TextNode with zero tokens
        assert len(result) == 1, f"Expected 1 node, got {len(result)}"
        assert result[0].doc_id == expected[0].doc_id, "Doc ID should match"
        assert result[0].header == expected[0].header, "Header should match"
        assert result[0].content == expected[0].content, "Content should match"
        assert result[0].num_tokens == expected[0].num_tokens, "Num tokens should be 0"
