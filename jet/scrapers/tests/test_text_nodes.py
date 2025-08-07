import pytest
from typing import List
from jet.scrapers.text_nodes import extract_text_nodes, BaseNode
from pyquery import PyQuery as pq


@pytest.fixture
def sample_html():
    """Fixture providing a sample HTML string for tests."""
    return """
    <div id="main" class="container">
        <h1 id="header" class="title">Welcome</h1>
        <p id="intro" class="text">Hello, world!</p>
        <script>console.log('test');</script>
        <style>.hidden { display: none; }</style>
        <nav id="menu">Navigation</nav>
    </div>
    """


@pytest.fixture
def temp_html_file(tmp_path):
    """Fixture creating a temporary HTML file."""
    file_path = tmp_path / "test.html"
    content = """
    <div id="content">
        <p id="p1" class="text">File content</p>
    </div>
    """
    file_path.write_text(content)
    return str(file_path)


class TestExtractTextNodes:
    """Tests for the extract_text_nodes function."""

    def test_extract_text_from_html_string(self, sample_html: str):
        """Test extracting text nodes from an HTML string, excluding specified tags."""
        # Given: An HTML string with various elements
        html = sample_html
        excludes = ["nav", "footer", "script", "style"]
        # When: Extracting text nodes with exclusions
        expected = [
            BaseNode(
                tag="h1",
                text="Welcome",
                depth=1,
                raw_depth=3,
                id="header",
                class_names=["title"],
                line=pq(
                    '<h1 id="header" class="title">Welcome</h1>').outer_html().__hash__() % 1000,
                html='<h1 id="header" class="title">Welcome</h1>'
            ),
            BaseNode(
                tag="p",
                text="Hello, world!",
                depth=1,
                raw_depth=3,
                id="intro",
                class_names=["text"],
                line=pq(
                    '<p id="intro" class="text">Hello, world!</p>').outer_html().__hash__() % 1000,
                html='<p id="intro" class="text">Hello, world!</p>'
            )
        ]
        result = extract_text_nodes(html, excludes=excludes, timeout_ms=100)
        # Then: Expect correct text nodes extracted
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.tag == e.tag
            assert r.text == e.text
            assert r.depth == e.depth
            assert r.raw_depth == e.raw_depth
            assert r.id == e.id
            assert r.class_names == e.class_names
            assert r.get_html() == e.get_html()

    def test_extract_text_from_file_path(self, temp_html_file: str):
        """Test extracting text nodes from a file path."""
        # Given: A file path to an HTML file
        file_path = temp_html_file
        excludes = ["nav", "footer", "script", "style"]
        # When: Extracting text nodes from the file
        expected = [
            BaseNode(
                tag="p",
                text="File content",
                depth=1,
                raw_depth=3,
                id="p1",
                class_names=["text"],
                line=pq(
                    '<p id="p1" class="text">File content</p>').outer_html().__hash__() % 1000,
                html='<p id="p1" class="text">File content</p>'
            )
        ]
        result = extract_text_nodes(
            file_path, excludes=excludes, timeout_ms=100)
        # Then: Expect correct text nodes extracted
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.tag == e.tag
            assert r.text == e.text
            assert r.depth == e.depth
            assert r.raw_depth == e.raw_depth
            assert r.id == e.id
            assert r.class_names == e.class_names
            assert r.get_html() == e.get_html()

    def test_exclude_elements(self, sample_html: str):
        """Test that excluded tags are not included in the extracted nodes."""
        # Given: An HTML string with excludable tags
        html = sample_html
        excludes = ["nav", "script", "style"]
        # When: Extracting text nodes with exclusions
        expected = [
            BaseNode(
                tag="h1",
                text="Welcome",
                depth=1,
                raw_depth=3,
                id="header",
                class_names=["title"],
                line=pq(
                    '<h1 id="header" class="title">Welcome</h1>').outer_html().__hash__() % 1000,
                html='<h1 id="header" class="title">Welcome</h1>'
            ),
            BaseNode(
                tag="p",
                text="Hello, world!",
                depth=1,
                raw_depth=3,
                id="intro",
                class_names=["text"],
                line=pq(
                    '<p id="intro" class="text">Hello, world!</p>').outer_html().__hash__() % 1000,
                html='<p id="intro" class="text">Hello, world!</p>'
            )
        ]
        result = extract_text_nodes(html, excludes=excludes, timeout_ms=100)
        # Then: Expect only non-excluded nodes
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.tag not in excludes
            assert r.tag == e.tag
            assert r.text == e.text
            assert r.id == e.id
            assert r.depth == e.depth
            assert r.raw_depth == e.raw_depth
            assert r.class_names == e.class_names
            assert r.get_html() == e.get_html()

    def test_no_text_nodes(self):
        """Test handling HTML with no valid text nodes after exclusions."""
        # Given: An HTML string with only excludable tags
        html = """
        <script>console.log('test');</script>
        <style>.hidden { display: none; }</style>
        <nav>Navigation</nav>
        """
        excludes = ["nav", "footer", "script", "style"]
        # When: Extracting text nodes
        expected: List[BaseNode] = []
        result = extract_text_nodes(html, excludes=excludes, timeout_ms=100)
        # Then: Expect empty result
        assert result == expected

    def test_invalid_id_fallback(self, sample_html: str):
        """Test that nodes with invalid IDs get a fallback ID."""
        # Given: An HTML string with an invalid ID
        html = '<p id="invalid@id" class="text">Invalid ID test</p>'
        excludes = ["nav", "footer", "script", "style"]
        # When: Extracting text nodes
        expected = [
            BaseNode(
                tag="p",
                text="Invalid ID test",
                depth=1,
                raw_depth=2,
                id="node_1_0",
                class_names=["text"],
                line=pq(
                    '<p id="invalid@id" class="text">Invalid ID test</p>').outer_html().__hash__() % 1000,
                html='<p id="invalid@id" class="text">Invalid ID test</p>'
            )
        ]
        result = extract_text_nodes(html, excludes=excludes, timeout_ms=100)
        # Then: Expect node with fallback ID
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.tag == e.tag
            assert r.text == e.text
            assert r.depth == e.depth
            assert r.raw_depth == e.raw_depth
            assert r.id == e.id
            assert r.class_names == e.class_names
            assert r.get_html() == e.get_html()

    def test_html_with_doctype_and_root(self):
        """Test extracting text nodes from HTML with DOCTYPE, html root, and comments."""
        # Given: An HTML string with DOCTYPE, html root, and a comment
        html = """
        <!DOCTYPE html>
        <html>
            <body>
                <!-- This is a comment -->
                <h1 id="header" class="title">Welcome</h1>
                <p id="intro" class="text">Hello, world!</p>
            </body>
        </html>
        """
        excludes = ["nav", "footer", "script", "style"]
        # When: Extracting text nodes
        expected = [
            BaseNode(
                tag="h1",
                text="Welcome",
                depth=1,
                raw_depth=2,
                id="header",
                class_names=["title"],
                line=pq(
                    '<h1 id="header" class="title">Welcome</h1>').outer_html().__hash__() % 1000,
                html='<h1 id="header" class="title">Welcome</h1>'
            ),
            BaseNode(
                tag="p",
                text="Hello, world!",
                depth=1,
                raw_depth=2,
                id="intro",
                class_names=["text"],
                line=pq(
                    '<p id="intro" class="text">Hello, world!</p>').outer_html().__hash__() % 1000,
                html='<p id="intro" class="text">Hello, world!</p>'
            )
        ]
        result = extract_text_nodes(html, excludes=excludes, timeout_ms=100)
        # Then: Expect correct text nodes ignoring DOCTYPE, html root, and comments
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert r.tag == e.tag
            assert r.text == e.text
            assert r.depth == e.depth
            assert r.raw_depth == e.raw_depth
            assert r.id == e.id
            assert r.class_names == e.class_names
            assert r.get_html() == e.get_html()
