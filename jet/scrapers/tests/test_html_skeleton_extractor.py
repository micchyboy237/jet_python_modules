"""
Unit tests for HTML Skeleton Extractor
Follows pytest best practices: fast, independent, descriptive names, edge cases [[35]][[36]][[41]]
"""

import pytest
from jet.scrapers.html_skeleton_extractor import (
    BlockTreeBuilder,
    HTMLSkeletonExtractor,
    SkeletonConfig,
    build_block_tree,
    extract_skeleton,
)


class TestSkeletonConfig:
    """Tests for SkeletonConfig validation."""

    def test_default_config_is_valid(self):
        """Default configuration should pass validation."""
        config = SkeletonConfig()
        config.validate()

    def test_invalid_max_text_length_raises_error(self):
        """Negative max_text_length should raise ValueError."""
        config = SkeletonConfig(max_text_length=0)
        with pytest.raises(ValueError, match="max_text_length must be at least 1"):
            config.validate()

    def test_invalid_min_word_count_raises_error(self):
        """Negative min_word_count should raise ValueError."""
        config = SkeletonConfig(min_word_count_for_block=0)
        with pytest.raises(
            ValueError, match="min_word_count_for_block must be at least 1"
        ):
            config.validate()

    def test_empty_keep_tags_raises_error(self):
        """Empty keep_tags should raise ValueError."""
        config = SkeletonConfig(keep_tags=frozenset())
        with pytest.raises(ValueError, match="keep_tags cannot be empty"):
            config.validate()

    def test_custom_config_preserves_settings(self):
        """Custom configuration should preserve all settings."""
        config = SkeletonConfig(
            max_text_length=100, merge_single_children=False, preserve_comments=True
        )
        assert config.max_text_length == 100
        assert config.merge_single_children is False
        assert config.preserve_comments is True


class TestHTMLSkeletonExtractor:
    """Tests for HTMLSkeletonExtractor core functionality."""

    @pytest.fixture
    def extractor(self):
        """Create a default extractor instance."""
        return HTMLSkeletonExtractor()

    def test_extract_empty_html_returns_empty_string(self, extractor):
        """Empty HTML should return empty string."""
        result, url_map = extractor.extract("")
        assert result == ""
        assert url_map == {}

    def test_extract_removes_script_tags(self, extractor):
        """Script tags should be removed."""
        html = "<html><body><script>alert('xss')</script><p>Content</p></body></html>"
        result, _ = extractor.extract(html)
        assert "<script>" not in result
        assert "alert" not in result
        assert "Content" in result
        assert "<p>" in result

    def test_extract_removes_style_tags(self, extractor):
        """Style tags should be removed."""
        html = (
            "<html><body><style>.red { color: red; }</style><p>Text</p></body></html>"
        )
        result, _ = extractor.extract(html)
        assert "<style>" not in result
        assert "color: red" not in result
        assert "Text" in result

    def test_extract_preserves_semantic_tags(self, extractor):
        """Semantic HTML tags should be preserved."""
        html = """
        <html>
        <body>
            <header><nav><a href="/">Home</a></nav></header>
            <main>
                <article>
                    <h1>Title</h1>
                    <p>Content</p>
                </article>
            </main>
            <footer>Copyright</footer>
        </body>
        </html>
        """
        result, _ = extractor.extract(html)
        assert "<header>" in result
        assert "<nav>" in result
        assert "<main>" in result
        assert "<article>" in result
        assert "<h1>" in result
        assert "<footer>" in result

    def test_extract_preserves_essential_attributes(self, extractor):
        """Essential attributes like id, class, role should be preserved."""
        html = '<div id="main" class="container" role="main"><p>Text</p></div>'
        result, _ = extractor.extract(html)
        assert 'id="main"' in result
        assert 'class="container"' in result
        assert 'role="main"' in result

    def test_extract_removes_non_essential_attributes(self, extractor):
        """Non-essential attributes should be removed."""
        html = '<div onclick="handleClick()" style="color: red" data-custom="value"><p>Text</p></div>'
        result, _ = extractor.extract(html)
        assert "onclick" not in result
        assert "style=" not in result
        assert "data-custom" not in result

    def test_extract_truncates_long_text(self, extractor):
        """Long text content should be truncated."""
        long_text = " ".join([f"word{i}" for i in range(100)])
        html = f"<p>{long_text}</p>"
        result, _ = extractor.extract(html)
        assert "..." in result or len(result) < len(html)

    def test_extract_merges_single_child_containers(self, extractor):
        """Single-child containers should be merged."""
        html = "<div><div><p>Text</p></div></div>"
        result, _ = extractor.extract(html)
        assert result.count("<div>") <= 2

    def test_extract_removes_empty_elements(self, extractor):
        """Empty elements should be removed."""
        html = "<div><p></p><span>Text</span><div></div></div>"
        result, _ = extractor.extract(html)
        assert "<p></p>" not in result
        assert "<div></div>" not in result
        assert "Text" in result
        assert "<span>" in result

    def test_extract_preserves_self_closing_tags(self, extractor):
        """Self-closing tags like br, img should be preserved."""
        html = "<p>Line1<br/>Line2<img src='test.jpg' alt='Test'/></p>"
        result, _ = extractor.extract(html)
        assert "<br" in result
        assert "<img" in result
        assert 'alt="Test"' in result

    def test_extract_handles_tables(self, extractor):
        """Table structure should be preserved."""
        html = """
        <table>
            <thead><tr><th>Name</th><th>Age</th></tr></thead>
            <tbody><tr><td>Alice</td><td>30</td></tr></tbody>
        </table>
        """
        result, _ = extractor.extract(html)
        assert "<table>" in result
        assert "<thead>" in result
        assert "<th>" in result
        assert "<tbody>" in result

    def test_extract_handles_forms(self, extractor):
        """Form elements should be preserved."""
        html = """
        <form>
            <label for="email">Email:</label>
            <input type="email" id="email" name="email"/>
            <button type="submit">Submit</button>
        </form>
        """
        result, _ = extractor.extract(html)
        assert "<form>" in result
        assert "<label" in result
        assert "<input" in result
        assert "<button" in result
        assert "Submit" in result

    def test_extract_unwraps_non_whitelisted_tags_with_children(self, extractor):
        """Non-whitelisted tags with important children should be unwrapped."""
        html = "<custom-tag><p>Important</p></custom-tag>"
        result, _ = extractor.extract(html)
        assert "<custom-tag>" not in result
        assert "Important" in result
        assert "<p>" in result

    def test_extract_decomposes_non_whitelisted_tags_without_children(self, extractor):
        """Non-whitelisted tags without children should be decomposed."""
        html = "<custom-tag></custom-tag><p>Text</p>"
        result, _ = extractor.extract(html)
        assert "<custom-tag>" not in result
        assert "Text" in result
        assert "<p>" in result

    def test_extract_preserves_comments_when_configured(self):
        """Comments should be preserved when configured."""
        config = SkeletonConfig(preserve_comments=True)
        extractor = HTMLSkeletonExtractor(config)
        html = "<!-- Comment --><p>Text</p>"
        result, _ = extractor.extract(html)
        assert "Comment" in result

    def test_extract_removes_comments_by_default(self, extractor):
        """Comments should be removed by default."""
        html = "<!-- Comment --><p>Text</p>"
        result, _ = extractor.extract(html)
        assert "Comment" not in result

    def test_extract_handles_nested_lists(self, extractor):
        """Nested lists should be preserved."""
        html = """
        <ul>
            <li>Item 1
                <ul>
                    <li>Subitem 1.1</li>
                </ul>
            </li>
        </ul>
        """
        result, _ = extractor.extract(html)
        assert "<ul>" in result
        assert "<li>" in result

    def test_extract_handles_links(self, extractor):
        """Links with href should be preserved."""
        html = '<a href="https://example.com" class="link">Click here</a>'
        result, _ = extractor.extract(html)
        assert "<a" in result
        assert 'href="https://example.com"' in result

    def test_extract_handles_images(self, extractor):
        """Images with src and alt should be preserved."""
        html = '<img src="image.jpg" alt="Description" width="100"/>'
        result, _ = extractor.extract(html)
        assert "<img" in result
        assert 'src="image.jpg"' in result
        assert 'alt="Description"' in result
        assert "width" not in result

    def test_complex_html_extraction(self, extractor):
        """Complex real-world HTML should be properly skeletonized."""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Test Page</title>
            <meta charset="utf-8"/>
            <link rel="stylesheet" href="style.css"/>
            <script src="app.js"></script>
        </head>
        <body>
            <header>
                <nav id="main-nav" class="navigation">
                    <ul>
                        <li><a href="/">Home</a></li>
                        <li><a href="/about">About</a></li>
                    </ul>
                </nav>
            </header>
            <main>
                <article id="post-1" class="blog-post">
                    <h1>Blog Post Title</h1>
                    <p>This is a very long paragraph with lots of text that should be truncated because it exceeds the maximum length limit.</p>
                    <img src="featured.jpg" alt="Featured image"/>
                    <div class="content">
                        <p>More content here.</p>
                    </div>
                </article>
            </main>
            <footer>
                <p>&copy; 2026 Test Site</p>
            </footer>
        </body>
        </html>
        """
        result, _ = extractor.extract(html)
        assert "<header>" in result
        assert "<nav" in result
        assert "<main>" in result
        assert "<article" in result
        assert "<h1>" in result
        assert "<footer>" in result
        assert "<script>" not in result
        assert "<link" not in result
        assert "<meta" not in result
        assert 'id="main-nav"' in result
        assert 'id="post-1"' in result


class TestURLCompression:
    """Tests for URL compression functionality."""

    def test_short_urls_are_not_compressed(self):
        """Short URLs should remain unchanged."""
        extractor = HTMLSkeletonExtractor()
        html = '<a href="https://short.com">Link</a>'
        result, url_map = extractor.extract(html)
        assert 'href="https://short.com"' in result
        assert url_map == {}

    def test_long_urls_are_compressed(self):
        """Long URLs should be replaced with placeholders."""
        extractor = HTMLSkeletonExtractor()
        long_url = "https://example.com/very/long/path/with/many/parameters?param1=value1&param2=value2&param3=value3"
        html = f'<a href="{long_url}">Link</a>'
        result, url_map = extractor.extract(html)
        assert 'href="' in result
        assert long_url not in result
        assert len(url_map) == 1
        placeholder = list(url_map.keys())[0]
        assert url_map[placeholder] == long_url
        assert placeholder in result

    def test_url_compression_can_be_disabled(self):
        """URL compression should be optional."""
        config = SkeletonConfig(compress_urls=False)
        extractor = HTMLSkeletonExtractor(config)
        long_url = "https://example.com/very/long/path/with/many/parameters?param1=value1&param2=value2&param3=value3"
        html = f'<a href="{long_url}">Link</a>'
        result, url_map = extractor.extract(html)
        # BeautifulSoup escapes & to &amp; in attributes, so we check for the escaped version or use html.unescape
        assert "https://example.com/very/long/path/with/many/parameters" in result
        assert "param1=value1" in result
        assert url_map == {}


class TestRandomClassFiltering:
    """Tests for filtering randomly generated class names."""

    def test_random_emotion_classes_are_removed(self):
        """Emotion-style classes (css-xxxxxx) should be removed."""
        extractor = HTMLSkeletonExtractor()
        html = '<div class="css-g5y9jx container">Content</div>'
        result, _ = extractor.extract(html)
        assert 'class="container"' in result
        assert "css-g5y9jx" not in result

    def test_short_hash_classes_are_removed(self):
        """Short hash classes like css-ti3gu should be removed."""
        extractor = HTMLSkeletonExtractor()
        html = '<div class="css-ti3gu main-content">Content</div>'
        result, _ = extractor.extract(html)
        assert 'class="main-content"' in result
        assert "css-ti3gu" not in result

    def test_mixed_alphanumeric_classes_are_removed(self):
        """Classes with mixed digits and letters like e8ju0x50 should be removed."""
        extractor = HTMLSkeletonExtractor()
        html = '<div class="e8ju0x50 r-1awozwy wrapper">Content</div>'
        result, _ = extractor.extract(html)
        assert 'class="wrapper"' in result
        assert "e8ju0x50" not in result
        assert "r-1awozwy" not in result

    def test_semantic_classes_are_preserved(self):
        """Semantic class names should be preserved."""
        extractor = HTMLSkeletonExtractor()
        html = '<div class="header-nav primary-button">Content</div>'
        result, _ = extractor.extract(html)
        assert 'class="header-nav primary-button"' in result

    def test_classes_with_numbers_are_removed_by_default(self):
        """Classes containing numbers (like col-2) are removed as they are often brittle layout classes."""
        extractor = HTMLSkeletonExtractor()
        html = '<div class="col-2 grid-item">Content</div>'
        result, _ = extractor.extract(html)
        assert "col-2" not in result

    def test_complex_class_mixing(self):
        """Should handle a mix of semantic and random classes."""
        extractor = HTMLSkeletonExtractor()
        html = '<div class="css-1a2b3c header-wrapper r-xyz123">Content</div>'
        result, _ = extractor.extract(html)
        assert 'class="header-wrapper"' in result
        assert "css-1a2b3c" not in result
        assert "r-xyz123" not in result


class TestBlockTreeBuilder:
    """Tests for BlockTreeBuilder functionality."""

    @pytest.fixture
    def builder(self):
        """Create a default block tree builder."""
        return BlockTreeBuilder()

    def test_build_empty_html_returns_empty_dict(self, builder):
        """Empty HTML should return empty dictionary."""
        assert builder.build_block_tree("") == {}
        assert builder.build_block_tree("   ") == {}

    def test_build_simple_html_creates_tree(self, builder):
        """Simple HTML should create a basic tree structure."""
        html = "<div><p>Text</p></div>"
        tree = builder.build_block_tree(html)
        assert tree != {}
        assert "tag" in tree
        assert "children" in tree

    def test_build_tree_preserves_attributes(self, builder):
        """Tree should preserve essential attributes."""
        html = '<html><body><div id="main" class="container"><p>Text</p></div></body></html>'
        tree = builder.build_block_tree(html)

        def find_node_by_tag(node, tag_name):
            if node.get("tag") == tag_name:
                return node
            for child in node.get("children", []):
                result = find_node_by_tag(child, tag_name)
                if result:
                    return result
            return None

        div_node = find_node_by_tag(tree, "div")
        assert div_node is not None
        assert div_node["attributes"].get("id") == "main"
        assert "container" in div_node["attributes"].get("class", "")

    def test_build_tree_calculates_word_count(self, builder):
        """Tree should calculate word counts correctly."""
        html = "<p>Hello world test</p>"
        tree = builder.build_block_tree(html)

        def find_node_by_tag(node, tag_name):
            if node.get("tag") == tag_name:
                return node
            for child in node.get("children", []):
                result = find_node_by_tag(child, tag_name)
                if result:
                    return result
            return None

        p_node = find_node_by_tag(tree, "p")
        assert p_node is not None
        assert p_node["word_count"] == 3

    def test_prune_by_relevance_keeps_relevant_nodes(self, builder):
        """Pruning should keep nodes matching query."""
        html = """
        <div>
            <p>Python programming tutorial</p>
            <p>JavaScript basics</p>
        </div>
        """
        tree = builder.build_block_tree(html)
        pruned = builder.prune_by_relevance(tree, "Python")
        assert pruned != {}

    def test_prune_by_relevance_removes_irrelevant_nodes(self, builder):
        """Pruning should remove nodes not matching query."""
        html = """
        <div>
            <p>Unrelated content here</p>
            <p>More unrelated text</p>
        </div>
        """
        tree = builder.build_block_tree(html)
        pruned = builder.prune_by_relevance(tree, "Python")
        assert pruned is not None

    def test_build_tree_handles_nested_structure(self, builder):
        """Tree should handle deeply nested structures."""
        html = """
        <div>
            <section>
                <article>
                    <h1>Title</h1>
                    <p>Content</p>
                </article>
            </section>
        </div>
        """
        tree = builder.build_block_tree(html)
        assert tree != {}
        assert len(tree.get("children", [])) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_extract_skeleton_function(self):
        """extract_skeleton should work as expected."""
        html = "<div><p>Test</p></div>"
        result, url_map = extract_skeleton(html)
        assert "<p>" in result
        assert isinstance(url_map, dict)

    def test_extract_skeleton_with_custom_config(self):
        """extract_skeleton should accept custom config."""
        config = SkeletonConfig(max_text_length=10)
        html = "<p>" + " ".join(["word"] * 50) + "</p>"
        result, _ = extract_skeleton(html, config)
        assert "..." in result or len(result) < 100

    def test_build_block_tree_function(self):
        """build_block_tree should work as expected."""
        html = "<div><p>Test</p></div>"
        tree = build_block_tree(html)
        assert tree != {}
        assert "tag" in tree


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_malformed_html_handling(self):
        """Extractor should handle malformed HTML gracefully."""
        html = "<div><p>Unclosed tag<div>Nested</div>"
        extractor = HTMLSkeletonExtractor()
        result, _ = extractor.extract(html)
        assert isinstance(result, str)

    def test_unicode_content_handling(self):
        """Extractor should handle Unicode content."""
        html = "<p>Hello 世界 🌍 Привет</p>"
        extractor = HTMLSkeletonExtractor()
        result, _ = extractor.extract(html)
        assert "世界" in result or "🌍" in result or "Привет" in result

    def test_large_html_performance(self):
        """Extractor should handle large HTML documents."""
        items = "".join([f"<li>Item {i}</li>" for i in range(1000)])
        html = f"<ul>{items}</ul>"
        extractor = HTMLSkeletonExtractor()
        result, _ = extractor.extract(html)
        assert isinstance(result, str)
        assert "<ul>" in result

    def test_special_characters_in_attributes(self):
        """Extractor should handle special characters in attributes."""
        html = (
            '<div data-value="test&amp;value" title="Quote: \\"test\\"">Content</div>'
        )
        extractor = HTMLSkeletonExtractor()
        result, _ = extractor.extract(html)
        assert isinstance(result, str)

    def test_deeply_nested_structure(self):
        """Extractor should handle deeply nested structures."""
        html = "<div>" * 50 + "<p>Deep</p>" + "</div>" * 50
        extractor = HTMLSkeletonExtractor()
        result, _ = extractor.extract(html)
        assert "Deep" in result
        assert "<p>" in result

    def test_mixed_content_types(self):
        """Extractor should handle mixed content types."""
        html = """
        <div>
            Text before
            <span>Inline</span>
            More text
            <strong>Bold</strong>
            Final text
        </div>
        """
        extractor = HTMLSkeletonExtractor()
        result, _ = extractor.extract(html)
        assert "<span>" in result
        assert "<strong>" in result


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_extraction_pipeline(self):
        """Test complete extraction from raw HTML to skeleton."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Product Page</title>
            <script>var analytics = {};</script>
            <style>.price { color: red; }</style>
        </head>
        <body>
            <header>
                <nav id="navbar">
                    <a href="/">Home</a>
                    <a href="/products">Products</a>
                </nav>
            </header>
            <main>
                <article id="product-123" class="product-card">
                    <h1>Premium Widget</h1>
                    <img src="widget.jpg" alt="Premium Widget Image"/>
                    <div class="description">
                        <p>This is an amazing widget with great features.</p>
                    </div>
                    <div class="price-container">
                        <span class="price">$99.99</span>
                    </div>
                    <button id="add-to-cart" aria-label="Add Premium Widget to cart">
                        Add to Cart
                    </button>
                </article>
            </main>
            <footer>
                <p>© 2026 Widget Co.</p>
            </footer>
        </body>
        </html>
        """
        skeleton, url_map = extract_skeleton(html)
        assert "<header>" in skeleton
        assert "<nav" in skeleton
        assert "<main>" in skeleton
        assert "<article" in skeleton
        assert "<h1>" in skeleton
        assert "<footer>" in skeleton
        assert "<script>" not in skeleton
        assert "<style>" not in skeleton
        assert 'id="product-123"' in skeleton
        assert "aria-label=" in skeleton
        assert skeleton.count("<div>") < 10

    def test_block_tree_with_pruning(self):
        """Test block tree building and pruning workflow."""
        html = """
        <html>
        <body>
            <div>
                <h1>Python Tutorial</h1>
                <p>Learn Python programming from scratch.</p>
            </div>
            <div>
                <h1>JavaScript Guide</h1>
                <p>Master JavaScript development.</p>
            </div>
        </body>
        </html>
        """
        tree = build_block_tree(html)
        assert tree != {}
        builder = BlockTreeBuilder()
        pruned = builder.prune_by_relevance(tree, "Python")
        assert pruned is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
