from __future__ import annotations

import logging

from jet.scrapers.utils.replace_links import replace_links


class TestReplaceLinks:
    """Tests for replace_links root-relative link replacement."""

    # --- Valid Replacement Cases ---

    def test_single_root_relative_link(self):
        text = "Visit /about for more info"
        result = replace_links(text, "https://example.com")
        assert result == "Visit https://example.com/about for more info"

    def test_multiple_root_relative_links(self):
        text = "See /products and /blog/post-1 or /contact"
        result = replace_links(text, "https://shop.example.com")
        expected = (
            "See https://shop.example.com/products and "
            "https://shop.example.com/blog/post-1 or "
            "https://shop.example.com/contact"
        )
        assert result == expected

    def test_preserves_absolute_urls_untouched(self):
        text = "External https://other.com/page and /local-path"
        result = replace_links(text, "https://example.com")
        expected = "External https://other.com/page and https://example.com/local-path"
        assert result == expected

    def test_base_url_path_is_ignored(self):
        """Root-relative links resolve against domain root, not base path."""
        text = "/dashboard"
        result = replace_links(text, "https://example.com/docs/index.html")
        assert result == "https://example.com/dashboard"

    def test_link_with_query_and_fragment(self):
        text = "/search?q=test&page=2#results"
        result = replace_links(text, "https://example.com")
        assert result == "https://example.com/search?q=test&page=2#results"

    def test_root_only_link(self):
        text = "Go to / for home"
        result = replace_links(text, "https://example.com")
        assert result == "Go to https://example.com/ for home"

    def test_http_scheme_preserved_from_base(self):
        text = "/api/v1/users"
        result = replace_links(text, "http://localhost:8080/app")
        assert result == "http://localhost:8080/api/v1/users"

    def test_base_with_port(self):
        text = "/health"
        result = replace_links(text, "https://example.com:9443/internal")
        assert result == "https://example.com:9443/health"

    # --- HTML-like Context ---

    def test_href_attribute_replacement(self):
        text = '<a href="/products/123">Product</a>'
        result = replace_links(text, "https://shop.example.com")
        assert result == '<a href="https://shop.example.com/products/123">Product</a>'

    def test_src_attribute_replacement(self):
        text = '<img src="/static/logo.png">'
        result = replace_links(text, "https://cdn.example.com")
        assert result == '<img src="https://cdn.example.com/static/logo.png">'

    def test_mixed_html_content(self):
        text = (
            "<div>\n"
            '  <a href="/about">About</a>\n'
            '  <a href="https://external.com">External</a>\n'
            '  <img src="/img/banner.jpg">\n'
            "  Visit /help for support\n"
            "</div>"
        )
        result = replace_links(text, "https://example.com")
        assert "https://example.com/about" in result
        assert "https://external.com" in result  # Untouched
        assert "https://example.com/img/banner.jpg" in result
        assert "https://example.com/help" in result

    # --- Stop Characters & Boundary Safety ---

    def test_stops_at_quote_boundary(self):
        text = '<a href="/path">link</a>'
        result = replace_links(text, "https://example.com")
        # Should NOT capture beyond the closing quote
        assert 'href="https://example.com/path"' in result
        assert '">link' in result

    def test_stops_at_bracket_boundary(self):
        text = "Check [/wiki/article] for details"
        result = replace_links(text, "https://example.com")
        assert "https://example.com/wiki/article" in result
        assert "]" in result  # Bracket preserved outside link

    def test_stops_at_whitespace(self):
        text = "/first /second"
        result = replace_links(text, "https://example.com")
        assert result == "https://example.com/first https://example.com/second"

    def test_does_not_double_prefix_existing_absolute_url(self):
        """Negative lookbehind prevents matching / inside https://host/path."""
        text = "https://example.com/existing/path"
        result = replace_links(text, "https://other.com")
        assert result == "https://example.com/existing/path"

    # --- Edge Cases & Invalid Inputs ---

    def test_empty_text_returns_empty_string(self):
        assert replace_links("", "https://example.com") == ""

    def test_none_text_returns_empty_string(self):
        assert replace_links(None, "https://example.com") == ""  # type: ignore[arg-type]

    def test_whitespace_only_text(self):
        assert replace_links("   \n\t ", "https://example.com") == "   \n\t "

    def test_no_root_relative_links_in_text(self):
        text = "No links here, just plain text and https://absolute.com/url"
        result = replace_links(text, "https://example.com")
        assert result == text

    def test_invalid_base_url_returns_original_text(self, caplog):
        text = "/some/path"
        with caplog.at_level(logging.WARNING):
            result = replace_links(text, "not-a-valid-url")
        assert result == text
        assert (
            "invalid base" in caplog.text.lower()
            or "missing scheme" in caplog.text.lower()
        )

    def test_base_without_scheme_returns_original(self, caplog):
        text = "/path"
        with caplog.at_level(logging.WARNING):
            result = replace_links(text, "example.com/path")
        assert result == text

    def test_base_without_host_returns_original(self, caplog):
        text = "/path"
        with caplog.at_level(logging.WARNING):
            result = replace_links(text, "https://")
        assert result == text

    def test_empty_base_returns_original(self, caplog):
        text = "/path"
        with caplog.at_level(logging.WARNING):
            result = replace_links(text, "")
        assert result == text

    # --- Consistency with scrape_links ---

    def test_replaced_links_match_scrape_links_output(self):
        """Verify that replace_links produces the same absolute URLs
        that scrape_links would extract when given the same base."""
        from jet.scrapers.utils.scrape_links import scrape_links

        text = "/products /about https://external.com/x"
        base = "https://shop.example.com"

        replaced = replace_links(text, base)
        scraped = scrape_links(replaced, base)

        # All root-relative links should now be absolute and present in scraped output
        assert "https://shop.example.com/products" in scraped
        assert "https://shop.example.com/about" in scraped
        assert "https://external.com/x" in scraped

    # --- Logging Verification ---

    def test_logs_replacement_count(self, caplog):
        text = "/a /b /c"
        with caplog.at_level(logging.INFO):
            replace_links(text, "https://example.com")
        assert "3" in caplog.text
        assert "root-relative link" in caplog.text.lower()

    def test_logs_zero_replacements(self, caplog):
        text = "no links here"
        with caplog.at_level(logging.INFO):
            replace_links(text, "https://example.com")
        assert "0" in caplog.text
