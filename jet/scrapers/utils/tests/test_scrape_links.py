# test_scrape_links.py
import unittest

from jet.scrapers.utils import scrape_links


class TestScrapeLinks(unittest.TestCase):
    def test_empty_text(self):
        self.assertEqual(scrape_links(""), [])
        self.assertEqual(scrape_links("   \n\t "), [])

    def test_no_links(self):
        text = """
        Hello world this is a test
        without any urls or paths
        """
        self.assertEqual(scrape_links(text), [])

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
        self.assertEqual(scrape_links(text), expected)

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
        self.assertEqual(scrape_links(text), expected)

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
        self.assertEqual(scrape_links(text, base), expected)

    def test_base_url_without_trailing_slash(self):
        base = "https://example.com"
        text = "/contact /team"
        expected = ["https://example.com/contact", "https://example.com/team"]
        self.assertEqual(scrape_links(text, base), expected)

    def test_duplicates_are_removed(self):
        text = """
        /products
        https://shop.com
        /products
        /products?color=blue
        https://shop.com
        """
        expected = ["/products", "https://shop.com", "/products?color=blue"]
        self.assertEqual(scrape_links(text), expected)

        with_base = scrape_links(text, "https://shop.com")
        self.assertEqual(
            with_base,
            [
                "https://shop.com/products",
                "https://shop.com",
                "https://shop.com/products?color=blue",
            ],
        )

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
        self.assertEqual(scrape_links(text), expected)

    def test_very_long_path(self):
        long_path = "/".join(["segment" + str(i) for i in range(200)])
        text = f"Link: /{long_path}"
        result = scrape_links(text)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].startswith("/"))

    def test_anchor_and_query_only(self):
        text = """
        #section
        ?page=3
        /page#top
        /?sort=asc#results
        """
        expected = ["/page#top", "/?sort=asc#results"]
        self.assertEqual(scrape_links(text), expected)

    def test_base_url_is_itself_not_included(self):
        base = "https://example.com/"
        text = """
        https://example.com
        https://example.com/
        /dashboard
        https://example.com/other
        """
        expected = ["https://example.com/dashboard", "https://example.com/other"]
        self.assertEqual(scrape_links(text, base), expected)

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
        self.assertEqual(scrape_links(text), expected_without_base)

        base = "https://shop.example.com"
        expected_with_base = [
            "https://shop.example.com/products/123",
            "https://cdn.example.com/img.jpg",
            "https://google.com",
            "https://shop.example.com/static/logo.png",
            "https://shop.example.com/help",
        ]
        self.assertEqual(scrape_links(text, base), expected_with_base)


if __name__ == "__main__":
    unittest.main()
