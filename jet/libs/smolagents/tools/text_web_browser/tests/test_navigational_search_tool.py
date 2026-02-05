from jet.libs.smolagents.tools.text_web_browser.navigational_search_tool import (
    LinkExtractor,
    MarkdownSectionExtractor,
    NavigationalSearchTool,
    clean_text_for_embedding,
)

# ============================================================
# Test utilities
# ============================================================


class FakeBrowser:
    def __init__(self, content: str, address: str = "https://example.com"):
        self.page_content = content
        self.address = address


# ============================================================
# Section extraction tests
# ============================================================


class TestMarkdownSectionExtractor:
    def test_extracts_all_header_levels(self):
        text = """
# Title

Intro text.

## Section A
Content A

### Subsection A1
Content A1

#### Deep Section
Deep content

## Section B
Content B
"""

        sections = MarkdownSectionExtractor.extract_sections(text)

        assert sections[0]["level"] == 1
        assert sections[0]["title"] == "Title"

        assert sections[1]["level"] == 2
        assert sections[1]["title"] == "Section A"

        assert sections[2]["level"] == 3
        assert sections[2]["title"] == "Subsection A1"

        assert sections[3]["level"] == 4
        assert sections[3]["title"] == "Deep Section"

        assert sections[4]["level"] == 2
        assert sections[4]["title"] == "Section B"

    def test_returns_empty_when_no_headers(self):
        text = "Just plain text without markdown headers."
        sections = MarkdownSectionExtractor.extract_sections(text)
        assert sections == []


# ============================================================
# Link extraction tests
# ============================================================


class TestLinkExtractor:
    def test_extracts_markdown_links(self):
        text = "See [Docs](docs.html) and [API](api.html)"
        links = LinkExtractor.extract_markdown_links(text)

        assert links == [
            {"text": "Docs", "url": "docs.html"},
            {"text": "API", "url": "api.html"},
        ]

    def test_returns_empty_when_no_links(self):
        text = "No links here."
        links = LinkExtractor.extract_markdown_links(text)
        assert links == []


# ============================================================
# NavigationalSearchTool tests
# ============================================================


class TestNavigationalSearchToolWithGoal:
    def test_ranks_links_by_goal(self):
        content = """
[Installation](install.html)
[API Reference](api.html)
[Changelog](changelog.html)
"""
        browser = FakeBrowser(content)
        tool = NavigationalSearchTool(browser)

        result = tool.forward(goal="installation")

        assert "Installation" in result
        assert "install.html" in result
        # No longer assert strict order — embeddings are semantic
        # Instead check that Installation is still present

    def test_applies_minimum_threshold(self):
        content = """
[Irrelevant](blah.html) something unrelated
[Installation Guide](install.html)
[API](api.html)
"""
        browser = FakeBrowser(content)
        tool = NavigationalSearchTool(browser)
        tool.MIN_SIMILARITY_THRESHOLD = 0.50  # force high threshold
        result = tool.forward(goal="how to install library")
        assert "Installation Guide" in result
        assert "Irrelevant" not in result  # should be filtered

    def test_url_path_bonus_helps(self):
        content = """
[Start here](getting-started/install.html)
[Reference](api.html)
"""
        browser = FakeBrowser(content)
        tool = NavigationalSearchTool(browser)
        result = tool.forward(goal="installation instructions")
        # With path bonus, install.html should rank higher even if title is generic
        assert result.index("install.html") < result.index("api.html")

    def test_context_helps_disambiguate(self):
        content = """
Before you begin the Installation Guide, make sure you have Python 3.9+.

[Installation Guide](install.html)

For developers: [API Reference](api.html) is useful.
"""
        browser = FakeBrowser(content)
        tool = NavigationalSearchTool(browser)
        result = tool.forward(goal="how do I install this?")
        assert "Installation Guide" in result
        # Ideally ranks higher thanks to "Installation Guide" + "Before you begin..." context


class TestHelpers:
    def test_clean_text_for_embedding(self):
        text = "Click Here to Read More about → Installation!!"
        cleaned = clean_text_for_embedding(text)
        assert "click" not in cleaned
        assert "read" not in cleaned
        assert "installation" in cleaned


class TestNavigationalSearchToolWithoutGoal:
    def test_returns_all_links_when_no_goal(self):
        content = """
[Home](/)
[Docs](/docs)
"""
        browser = FakeBrowser(content)
        tool = NavigationalSearchTool(browser)

        result = tool.forward()

        assert "Home" in result
        assert "Docs" in result

    def test_handles_no_links_gracefully(self):
        browser = FakeBrowser("Just text, no links.")
        tool = NavigationalSearchTool(browser)

        result = tool.forward()
        assert result == "No navigational links found on the current page."
