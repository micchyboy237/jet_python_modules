from jet.scrapers.utils import scrape_links


class TestScrapeLinks:
    def test_empty_text(self):
        assert scrape_links("") == []
        assert scrape_links("   \n\t ") == []

    def test_no_links(self):
        text = """
        Hello world this is a test
        without any urls or paths
        """
        assert scrape_links(text) == []

    def test_absolute_http_https_links(self):
        text = """
        Visit https://example.com
        or http://test.org/path?query=123#frag
        Also www.google.com is not matched (no protocol)
        https://api.github.com/users/octocat/repos
        """
        expected = [
            "https://example.com",
            "http://test.org/path?query=123#frag",
            "https://api.github.com/users/octocat/repos",
        ]
        assert scrape_links(text) == expected

    def test_relative_paths_without_base(self):
        text = """
        Go to /about
        /blog/post-123?sort=desc
        /assets/style.css
        /?ref=footer
        """
        expected = [
            "/about",
            "/blog/post-123?sort=desc",
            "/assets/style.css",
            "/?ref=footer",
        ]
        assert scrape_links(text) == expected

    def test_relative_paths_with_base_url(self):
        base = "https://example.com/docs"
        text = """
        See /faq
        /images/logo.png
        /api/v1/users/42
        https://other.com/external
        """
        expected = [
            "https://example.com/faq",
            "https://example.com/images/logo.png",
            "https://example.com/api/v1/users/42",
            "https://other.com/external",
        ]
        assert scrape_links(text, base) == expected

    def test_base_url_without_trailing_slash(self):
        base = "https://example.com"
        text = "/contact /team"
        expected = ["https://example.com/contact", "https://example.com/team"]
        assert scrape_links(text, base) == expected

    def test_duplicates_are_removed(self):
        text = """
        /products
        https://shop.com
        /products
        /products?color=blue
        https://shop.com
        """
        expected = ["/products", "https://shop.com", "/products?color=blue"]
        assert scrape_links(text) == expected

        with_base = scrape_links(text, "https://shop.com")
        assert with_base == [
            "https://shop.com/products",
            "https://shop.com",
            "https://shop.com/products?color=blue",
        ]

    def test_ignores_invalid_or_malicious_looking_urls(self):
        text = """
        https://example.com
        javascript:alert(1)
        data:text/html,<script>bad</script>
        /normal/path
        https://exa mple.com  (space)
        https://example.com"><script>alert(1)</script>
        """
        expected = ["https://example.com", "/normal/path"]
        assert scrape_links(text) == expected

    def test_very_long_path(self):
        long_path = "/".join(["segment" + str(i) for i in range(200)])
        text = f"Link: /{long_path}"
        result = scrape_links(text)
        assert len(result) == 1
        assert result[0].startswith("/")

    def test_anchor_and_query_only(self):
        text = """
        #section
        ?page=3
        /page#top
        /?sort=asc#results
        """
        expected = ["/page#top", "/?sort=asc#results"]
        assert scrape_links(text) == expected

    def test_base_url_is_itself_not_included(self):
        base = "https://example.com/"
        text = """
        https://example.com
        https://example.com/
        /dashboard
        https://example.com/other
        """
        expected = ["https://example.com/dashboard", "https://example.com/other"]
        assert scrape_links(text, base) == expected

    def test_complex_realistic_html_like_text(self):
        text = """
        <div class="content">
            <a href="/products/123">Product</a>
            <a href="https://cdn.example.com/img.jpg">Image</a>
            <a href="https://google.com">Google</a>
            <img src="/static/logo.png">
            Check /help for more info
        </div>
        """
        expected_without_base = [
            "/products/123",
            "https://cdn.example.com/img.jpg",
            "https://google.com",
            "/static/logo.png",
            "/help",
        ]
        assert scrape_links(text) == expected_without_base

        base = "https://shop.example.com"
        expected_with_base = [
            "https://shop.example.com/products/123",
            "https://cdn.example.com/img.jpg",
            "https://google.com",
            "https://shop.example.com/static/logo.png",
            "https://shop.example.com/help",
        ]
        assert scrape_links(text, base) == expected_with_base

    def test_brackets_act_as_stop_characters_for_links(self):
        """Brackets stop link capture, preventing broken IPv6 parse errors in urljoin.

        The regex stops capturing at [ and ] characters, so:
        - Absolute URLs with brackets (https://[::1/path) are excluded entirely
        - Relative paths with brackets (/path/[brackets]) are truncated to the valid portion
        """
        base = "https://example.com"
        text = """
        https://[::1/path
        /valid/path
        /path/with/[brackets]
        /another/valid
        """
        # https://[::1/path → excluded (no valid domain before [)
        # /valid/path → captured fully
        # /path/with/[brackets] → captures /path/with/ (stops at [)
        # /another/valid → captured fully
        expected = [
            "https://example.com/valid/path",
            "https://example.com/path/with/",
            "https://example.com/another/valid",
        ]
        assert scrape_links(text, base) == expected
