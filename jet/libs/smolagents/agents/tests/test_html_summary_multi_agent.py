from unittest.mock import MagicMock

import pytest
from jet.libs.smolagents.agents.html_summary_multi_agent import (
    HTMLDOMTools,
    ScalableHTMLMultiAgentSummarizer,
)

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def sample_html():
    return """
    <html>
        <body>
            <header>
                <h1 class="title">Test Page</h1>
            </header>
            <section id="content">
                <p>Hello World</p>
                <a href="https://example.com">Example</a>
            </section>
            <footer>
                <p>Copyright 2025</p>
            </footer>
        </body>
    </html>
    """


@pytest.fixture
def dom_tools(sample_html):
    return HTMLDOMTools(sample_html)


@pytest.fixture
def summarizer(monkeypatch):
    """
    Mock all agent .run() calls so tests are deterministic.
    """

    instance = ScalableHTMLMultiAgentSummarizer(enable_cache=True)

    mock_subtree = MagicMock()
    mock_subtree.run.return_value = "STRUCTURED_SUBTREE"

    mock_merge = MagicMock()
    mock_merge.run.side_effect = lambda prompt: "MERGED_STRUCTURE"

    mock_manager = MagicMock()
    mock_manager.run.return_value = "FINAL_SUMMARY"

    instance.subtree_agent = mock_subtree
    instance.merge_agent = mock_merge
    instance.manager = mock_manager

    return instance


# ============================================================
# DOM TOOL TESTS
# ============================================================


class TestHTMLDOMTools:
    def test_iter_semantic_sections(self, dom_tools):
        # Given
        # HTML contains header, section, footer

        # When
        result = dom_tools.iter_semantic_sections()

        # Then
        expected = 3
        assert len(result) == expected

    def test_serialize_node_structure(self, dom_tools):
        # Given
        body = dom_tools.parser.css_first("body")

        # When
        result = dom_tools.serialize_node(body)

        # Then
        expected_tag = "body"
        assert result["tag"] == expected_tag
        assert "children" in result

    def test_serialize_preserves_attributes(self, dom_tools):
        # Given
        header = dom_tools.parser.css_first("h1")

        # When
        result = dom_tools.serialize_node(header)

        # Then
        expected_class = "title"
        assert result["attributes"]["class"] == expected_class

    def test_text_extraction(self, dom_tools):
        # Given
        paragraph = dom_tools.parser.css_first("p")

        # When
        result = dom_tools.serialize_node(paragraph)

        # Then
        expected_text = "Hello World"
        assert result["text"] == expected_text


# ============================================================
# SUMMARIZER TESTS
# ============================================================


class TestScalableHTMLMultiAgentSummarizer:
    def test_summarize_runs_pipeline(self, summarizer, sample_html):
        # Given
        # Agents are mocked

        # When
        result = summarizer.summarize(sample_html)

        # Then
        expected = "FINAL_SUMMARY"
        assert result == expected

    def test_subtree_agent_called(self, summarizer, sample_html):
        # Given

        # When
        summarizer.summarize(sample_html)

        # Then
        assert summarizer.subtree_agent.run.called is True

    def test_merge_agent_called(self, summarizer, sample_html):
        # Given

        # When
        summarizer.summarize(sample_html)

        # Then
        assert summarizer.merge_agent.run.called is True

    def test_cache_usage(self, summarizer, sample_html):
        # Given
        summarizer.summarize(sample_html)

        initial_calls = summarizer.subtree_agent.run.call_count

        # When (run again, should use cache)
        summarizer.summarize(sample_html)

        # Then
        expected_calls = initial_calls
        assert summarizer.subtree_agent.run.call_count == expected_calls

    def test_empty_html(self, summarizer):
        # Given
        empty_html = "<html></html>"

        # When
        result = summarizer.summarize(empty_html)

        # Then
        expected = "FINAL_SUMMARY"
        assert result == expected
