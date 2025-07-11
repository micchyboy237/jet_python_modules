import pytest
from typing import List

from jet.code.markdown_types.base_markdown_analysis_types import Analysis, HeaderCounts, TokenSequential
from jet.code.markdown_utils._markdown_analyzer import BASE_DEFAULTS, base_analyze_markdown


class TestBaseAnalyzeMarkdown:
    def test_tokens_sequential_with_line(self):
        # Given
        markdown = "# Header1\nParagraph"
        expected_tokens = [
            TokenSequential(
                checked=None,
                content="Header1",
                id=1,
                type="header",
                url=None,
                line=1
            ),
            TokenSequential(
                checked=None,
                content="Paragraph",
                id=2,
                type="paragraph",
                url=None,
                line=2
            )
        ]
        expected_analysis = Analysis(
            headers=1,
            paragraphs=1,
            blockquotes=0,
            code_blocks=0,
            ordered_lists=0,
            unordered_lists=0,
            tables=0,
            html_blocks=0,
            html_inline_count=0,
            words=2,
            characters=8,
            header_counts=HeaderCounts(h1=1, h2=0, h3=0, h4=0, h5=0, h6=0),
            text_links=0,
            image_links=0
        )

        # When
        result = base_analyze_markdown(markdown)

        # Then
        assert result["analysis"] == expected_analysis
        assert result["tokens_sequential"] == expected_tokens
        assert all(isinstance(token["line"], int)
                   for token in result["tokens_sequential"])

    def test_empty_input_uses_defaults(self):
        # Given
        markdown = ""
        expected = BASE_DEFAULTS.copy()

        # When
        result = base_analyze_markdown(markdown)

        # Then
        assert result == expected
        assert result["tokens_sequential"] == []
