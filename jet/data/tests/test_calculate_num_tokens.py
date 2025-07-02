import pytest
from typing import List, Union
from tokenizers import Tokenizer
from jet.data.header_docs import HeaderDocs
from jet.data.header_types import HeaderNode, TextNode, Nodes
from jet.models.model_types import ModelType
from jet.models.tokenizer.base import get_tokenizer, TokenizerWrapper


@pytest.fixture
def tokenizer():
    """Fixture to create a tokenizer for sentence-transformers/all-MiniLM-L6-v2."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = get_tokenizer(model_name)
    return TokenizerWrapper(tokenizer, remove_pad_tokens=True, add_special_tokens=False)


class TestCalculateNumTokens:
    def test_empty_header_docs(self, tokenizer):
        """Test calculate_num_tokens with empty HeaderDocs."""
        # Given
        header_docs = HeaderDocs(root=[], tokens=[])
        expected_token_counts = []

        # When
        header_docs.calculate_num_tokens(tokenizer)
        result = [node.num_tokens for node in header_docs.as_nodes()]

        # Then
        assert result == expected_token_counts, "Empty HeaderDocs should have no token counts"

    def test_single_text_node(self, tokenizer):
        """Test calculate_num_tokens with a single TextNode."""
        # Given
        text_node = TextNode(
            id="text1",
            line=1,
            type="paragraph",
            header="Sample text",
            content="This is a sample paragraph.",
            meta={}
        )
        header_docs = HeaderDocs(root=[text_node], tokens=[])
        # "Sample text\nThis is a sample paragraph." -> ~9 tokens
        expected_token_counts = [8]

        # When
        header_docs.calculate_num_tokens(tokenizer)
        result = [node.num_tokens for node in header_docs.as_nodes()]

        # Then
        assert result == expected_token_counts, "Single TextNode should have correct token count"
        assert text_node.num_tokens == 8, "TextNode should have 9 tokens"

    def test_nested_header_with_children(self, tokenizer):
        """Test calculate_num_tokens with a HeaderNode containing children."""
        # Given
        header_node = HeaderNode(
            id="header1",
            line=1,
            level=1,
            header="Main Header",
            content="Header content",
            children=[
                TextNode(
                    id="text1",
                    line=2,
                    type="paragraph",
                    header="Child text",
                    content="Child paragraph content.",
                    meta={}
                ),
                HeaderNode(
                    id="header2",
                    line=3,
                    level=2,
                    header="Sub Header",
                    content="Sub header content",
                    children=[
                        TextNode(
                            id="text2",
                            line=4,
                            type="paragraph",
                            header="Sub child text",
                            content="Sub child paragraph.",
                            meta={}
                        )
                    ]
                )
            ]
        )
        header_docs = HeaderDocs(root=[header_node], tokens=[])
        # Approximate token counts based on model
        expected_token_counts = [4, 6, 5, 7]
        # Main Header\nHeader content: ~6 tokens
        # Child text\nChild paragraph content.: ~8 tokens
        # Sub Header\nSub header content: ~5 tokens
        # Sub child text\nSub child paragraph.: ~7 tokens

        # When
        header_docs.calculate_num_tokens(tokenizer)
        result = [node.num_tokens for node in header_docs.as_nodes()]

        # Then
        assert result == expected_token_counts, "Nested structure should have correct token counts"
        assert header_node.num_tokens == 4, "Main Header should have 4 tokens"
        assert header_node.children[0].num_tokens == 6, "First child TextNode should have 6 tokens"
        assert header_node.children[1].num_tokens == 5, "Sub Header should have 5 tokens"
        assert header_node.children[1].children[0].num_tokens == 7, "Sub child TextNode should have 7 tokens"

    def test_multiple_root_nodes(self, tokenizer):
        """Test calculate_num_tokens with multiple root nodes."""
        # Given
        nodes = [
            HeaderNode(
                id="header1",
                line=1,
                level=1,
                header="Header 1",
                content="Content 1"
            ),
            TextNode(
                id="text1",
                line=2,
                type="paragraph",
                header="Text 1",
                content="Paragraph 1",
                meta={}
            ),
            HeaderNode(
                id="header2",
                line=3,
                level=1,
                header="Header 2",
                content="Content 2"
            )
        ]
        header_docs = HeaderDocs(root=nodes, tokens=[])
        expected_token_counts = [4, 4, 4]
        # Header 1\nContent 1: ~5 tokens
        # Text 1\nParagraph 1: ~6 tokens
        # Header 2\nContent 2: ~5 tokens

        # When
        header_docs.calculate_num_tokens(tokenizer)
        result = [node.num_tokens for node in header_docs.as_nodes()]

        # Then
        assert result == expected_token_counts, "Multiple root nodes should have correct token counts"
        assert nodes[0].num_tokens == 4, "Header 1 should have 4 tokens"
        assert nodes[1].num_tokens == 4, "Text 1 should have 4 tokens"
        assert nodes[2].num_tokens == 4, "Header 2 should have 4 tokens"

    def test_model_name_input(self, tokenizer):
        """Test calculate_num_tokens with model name input."""
        # Given
        text_node = TextNode(
            id="text1",
            line=1,
            type="paragraph",
            header="Sample text",
            content="This is a sample paragraph.",
            meta={}
        )
        header_docs = HeaderDocs(root=[text_node], tokens=[])
        expected_token_counts = [8]  # Same as single_text_node

        # When
        header_docs.calculate_num_tokens(
            "sentence-transformers/all-MiniLM-L6-v2")
        result = [node.num_tokens for node in header_docs.as_nodes()]

        # Then
        assert result == expected_token_counts, "Model name input should correctly assign token counts"
        assert text_node.num_tokens == 8, "TextNode should have 9 tokens"

    def test_empty_node_content(self, tokenizer):
        """Test calculate_num_tokens with nodes having empty content."""
        # Given
        header_node = HeaderNode(
            id="header1",
            line=1,
            level=1,
            header="",
            content="",
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
        header_docs = HeaderDocs(root=[header_node], tokens=[])
        expected_token_counts = [0, 0]

        # When
        header_docs.calculate_num_tokens(tokenizer)
        result = [node.num_tokens for node in header_docs.as_nodes()]

        # Then
        assert result == expected_token_counts, "Empty content nodes should have zero tokens"
        assert header_node.num_tokens == 0, "Header with empty content should have 0 tokens"
        assert header_node.children[0].num_tokens == 0, "TextNode with empty content should have 0 tokens"
