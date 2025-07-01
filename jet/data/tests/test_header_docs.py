import pytest
from typing import List, Union
from jet.code.markdown_types import MarkdownToken, ContentType
from jet.data.header_types import HeaderNode, TextNode, NodeType
from jet.data.header_docs import HeaderDocs
from jet.data.utils import generate_unique_id


@pytest.fixture
def sample_markdown() -> str:
    """Provide sample markdown content for testing."""
    return """
# Main Header
Main content

## Sub Header
Sub content

Paragraph content
"""


@pytest.fixture
def sample_tokens() -> List[MarkdownToken]:
    """Provide sample tokens for testing."""
    return [
        {"type": "header", "content": "Main Header\nMain content",
            "level": 1, "meta": {}, "line": 1},
        {"type": "header", "content": "Sub Header\nSub content",
            "level": 2, "meta": {}, "line": 4},
        {"type": "paragraph", "content": "Paragraph content",
            "level": None, "meta": {}, "line": 7}
    ]


class TestHeaderDocsFromTokens:
    def test_from_tokens_simple_hierarchy(self, sample_tokens: List[MarkdownToken]) -> None:
        """Test creating HeaderDocs from tokens with simple hierarchy."""
        # Given
        expected_nodes = [
            HeaderNode(
                id=sample_tokens[0]["content"],
                line=1,
                type="header",
                header="Main Header",
                content="Main content",
                level=1,
                children=[
                    HeaderNode(
                        id=sample_tokens[1]["content"],
                        line=4,
                        type="header",
                        header="Sub Header",
                        content="Sub content",
                        level=2,
                        parent_id=sample_tokens[0]["content"],
                        parent_header="Main Header",
                        children=[
                            TextNode(
                                id=sample_tokens[2]["content"],
                                line=7,
                                type="paragraph",
                                header="Paragraph content",
                                content="Paragraph content",
                                meta={},
                                parent_id=sample_tokens[1]["content"],
                                parent_header="Sub Header"
                            )
                        ]
                    )
                ]
            )
        ]

        # When
        result = HeaderDocs.from_tokens(sample_tokens)

        # Then
        assert len(result.root) == 1
        assert result.root[0].header == "# Main Header"
        assert result.root[0].content == "Main content"
        assert len(result.root[0].children) == 1
        assert result.root[0].children[0].header == "## Sub Header"
        assert result.root[0].children[0].children[0].header == "Paragraph content"
        assert result.tokens == sample_tokens

    def test_from_tokens_empty_tokens(self) -> None:
        """Test creating HeaderDocs from empty token list."""
        # Given
        tokens: List[MarkdownToken] = []
        expected_nodes: List[NodeType] = []

        # When
        result = HeaderDocs.from_tokens(tokens)

        # Then
        assert result.root == expected_nodes
        assert result.tokens == tokens

    def test_from_tokens_only_text(self) -> None:
        """Test creating HeaderDocs with only text tokens."""
        # Given
        tokens = [
            {"type": "paragraph", "content": "Text content",
                "level": None, "meta": {}, "line": 1}
        ]
        expected_nodes = [
            TextNode(
                id="text1",
                line=1,
                type="paragraph",
                header="Text content",
                content="Text content",
                meta={}
            )
        ]

        # When
        result = HeaderDocs.from_tokens(tokens)

        # Then
        assert len(result.root) == 1
        assert result.root[0].header == "Text content"
        assert result.root[0].content == "Text content"
        assert result.root[0].type == "paragraph"
        assert result.tokens == tokens


class TestHeaderDocsFromString:
    def test_from_string_valid_markdown(self, sample_markdown: str) -> None:
        """Test creating HeaderDocs from valid markdown string."""
        # Given
        expected_headers = ["# Main Header", "## Sub Header"]
        expected_content = ["Main content", "Sub content", "Paragraph content"]

        # When
        result = HeaderDocs.from_string(sample_markdown)

        # Then
        assert len(result.root) == 1
        assert result.root[0].header == "# Main Header"
        assert result.root[0].content == "Main content"
        assert len(result.root[0].children) == 1
        assert result.root[0].children[0].header == "## Sub Header"
        assert result.root[0].children[0].content == "Sub content\nParagraph content"

    def test_from_string_empty_markdown(self) -> None:
        """Test creating HeaderDocs from empty markdown string."""
        # Given
        markdown = ""
        expected_nodes: List[NodeType] = []

        # When
        result = HeaderDocs.from_string(markdown)

        # Then
        assert result.root == expected_nodes
        assert result.tokens == []

    def test_from_string_path(self, tmp_path) -> None:
        """Test creating HeaderDocs from markdown file path."""
        # Given
        markdown_file = tmp_path / "test.md"
        markdown_file.write_text("# Header\nContent")
        expected_header = "# Header"
        expected_content = "Content"

        # When
        result = HeaderDocs.from_string(markdown_file)

        # Then
        assert len(result.root) == 1
        assert result.root[0].header == expected_header
        assert result.root[0].content == expected_content


class TestHeaderDocsAsMethods:
    def test_as_texts(self, sample_tokens: List[MarkdownToken]) -> None:
        """Test as_texts method with hierarchical content."""
        # Given
        docs = HeaderDocs.from_tokens(sample_tokens)
        expected = ["# Main Header\nMain content", "## Sub Header\nSub content",
                    "Paragraph content\nParagraph content"]

        # When
        result = docs.as_texts()

        # Then
        assert result == expected

    def test_as_nodes(self, sample_tokens: List[MarkdownToken]) -> None:
        """Test as_nodes method with hierarchical content."""
        # Given
        docs = HeaderDocs.from_tokens(sample_tokens)
        expected_node_count = 3

        # When
        result = docs.as_nodes()

        # Then
        assert len(result) == expected_node_count
        assert any(n.header == "# Main Header" for n in result)
        assert any(n.header == "## Sub Header" for n in result)
        assert any(n.header == "Paragraph content" for n in result)

    def test_as_tree(self, sample_tokens: List[MarkdownToken]) -> None:
        """Test as_tree method with hierarchical content."""
        # Given
        docs = HeaderDocs.from_tokens(sample_tokens)
        expected_root_count = 1
        expected_child_count = 1
        expected_grandchild_count = 1

        # When
        result = docs.as_tree()

        # Then
        assert len(result["root"]) == expected_root_count
        assert result["root"][0]["type"] == "header"
        assert result["root"][0]["content"] == "Main content"
        assert len(result["root"][0]["children"]) == expected_child_count
        assert len(result["root"][0]["children"][0]
                   ["children"]) == expected_grandchild_count


class TestHeaderDocsHeaderPreservation:
    def test_from_tokens_preserves_hashtags(self) -> None:
        """Test that header nodes preserve hashtag prefixes in header field."""
        # Given: A list of tokens with headers of different levels
        tokens = [
            {
                "type": "header",
                "content": "# Main Header\nMain content",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "## Sub Header\nSub content",
                "level": 2,
                "meta": {},
                "line": 4
            },
            {
                "type": "paragraph",
                "content": "Paragraph content",
                "level": None,
                "meta": {},
                "line": 7
            }
        ]
        expected_headers = ["# Main Header",
                            "## Sub Header", "Paragraph content"]

        # When: HeaderDocs is created from tokens
        result = HeaderDocs.from_tokens(tokens)

        # Then: Headers in nodes should retain their hashtag prefixes
        result_headers = [node.header for node in result.as_nodes()]
        assert result_headers == expected_headers, (
            f"Expected headers {expected_headers}, but got {result_headers}"
        )

    def test_from_tokens_multiple_header_levels(self) -> None:
        """Test header preservation with multiple header levels and edge cases."""
        # Given: Tokens with various header levels and edge cases
        tokens = [
            {
                "type": "header",
                "content": "# Top Level\nContent",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "### Deep Header\nDeep content",
                "level": 3,
                "meta": {},
                "line": 4
            },
            {
                "type": "header",
                "content": "#### Empty Content\n",
                "level": 4,
                "meta": {},
                "line": 7
            }
        ]
        expected_headers = ["# Top Level",
                            "### Deep Header", "#### Empty Content"]

        # When: HeaderDocs is created from tokens
        result = HeaderDocs.from_tokens(tokens)

        # Then: All headers should retain their hashtag prefixes
        result_headers = [node.header for node in result.as_nodes()]
        assert result_headers == expected_headers, (
            f"Expected headers {expected_headers}, but got {result_headers}"
        )

    def test_as_texts_preserves_hashtags(self) -> None:
        """Test that as_texts method preserves hashtag prefixes in headers."""
        # Given: Tokens with headers
        tokens = [
            {
                "type": "header",
                "content": "# Main Header\nMain content",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "## Sub Header\nSub content",
                "level": 2,
                "meta": {},
                "line": 4
            }
        ]
        expected_texts = ["# Main Header\nMain content",
                          "## Sub Header\nSub content"]

        # When: HeaderDocs is created and as_texts is called
        docs = HeaderDocs.from_tokens(tokens)
        result = docs.as_texts()

        # Then: Texts should include headers with hashtags
        assert result == expected_texts, (
            f"Expected texts {expected_texts}, but got {result}"
        )
