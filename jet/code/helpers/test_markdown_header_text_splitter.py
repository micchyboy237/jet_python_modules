import unittest
from langchain_core.documents import Document
from jet.code.helpers.markdown_header_text_splitter import MarkdownHeaderTextSplitter


class TestMarkdownHeaderTextSplitter(unittest.TestCase):

    def setUp(self):
        """Setup test splitter with markdown and numbered headers."""
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=True
        )

    def test_markdown_headers(self):
        """Test splitting on markdown headers."""
        md_text = """# Header 1
        Some content.

        ## Header 2
        More content.

        ### Header 3
        Even more content."""

        result = self.splitter.split_text(md_text)

        expected = [
            Document(page_content="Some content.",
                     metadata={"h1": "Header 1"}),
            Document(page_content="More content.", metadata={
                     "h1": "Header 1", "h2": "Header 2"}),
            Document(page_content="Even more content.", metadata={
                     "h1": "Header 1", "h2": "Header 2", "h3": "Header 3"})
        ]

        self.assertEqual(result, expected)

    def test_numbered_headers(self):
        """Test that numbered headers are correctly treated as headers."""
        md_text = """1. Introduction
        This is an introduction.

        2. Methods
        Explanation of methods.

        3. Conclusion
        Final summary."""

        result = self.splitter.split_text(md_text)

        expected = [
            Document(page_content="This is an introduction.",
                     metadata={"1.": "Introduction"}),
            Document(page_content="Explanation of methods.",
                     metadata={"2.": "Methods"}),
            Document(page_content="Final summary.",
                     metadata={"3.": "Conclusion"})
        ]

        self.assertEqual(result, expected)

    def test_mixed_markdown_and_numbered_headers(self):
        """Test correct handling of markdown and numbered headers together."""
        md_text = """# Overview
        General content.

        1. First Topic
        Details on first topic.

        ## Subsection
        Subsection details.

        2. Second Topic
        More details."""

        result = self.splitter.split_text(md_text)

        expected = [
            Document(page_content="General content.",
                     metadata={"h1": "Overview"}),
            Document(page_content="Details on first topic.", metadata={
                     "h1": "Overview", "1.": "First Topic"}),
            Document(page_content="Subsection details.", metadata={
                     "h1": "Overview", "1.": "First Topic", "h2": "Subsection"}),
            Document(page_content="More details.", metadata={
                     "h1": "Overview", "2.": "Second Topic"})
        ]

        self.assertEqual(result, expected)

    def test_strip_headers_false(self):
        """Ensure headers remain in content when strip_headers=False."""
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2")],
            strip_headers=False
        )

        md_text = """# Introduction
        Introduction content.

        1. Topic One
        Discussion."""

        result = splitter.split_text(md_text)

        expected = [
            Document(page_content="# Introduction\nIntroduction content.",
                     metadata={"h1": "Introduction"}),
            Document(page_content="1. Topic One\nDiscussion.",
                     metadata={"1.": "Topic One"})
        ]

        self.assertEqual(result, expected)

    def test_empty_text(self):
        """Check empty input handling."""
        md_text = ""
        result = self.splitter.split_text(md_text)
        self.assertEqual(result, [])

    def test_whitespace_only(self):
        """Check handling of input with only whitespace."""
        md_text = "   \n   \n  "
        result = self.splitter.split_text(md_text)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
