import re

from jet.utils.url_utils import remove_links

class TestRemoveLinks:
    def test_sample_text(self):
        """Given the sample text with various links, When remove_links is called, Then only non-URL paths and text remain."""
        text_with_sample_links = """
/
/db "Database" /threads/ "Threads" /sample-with-param?q=test /sample-with-fragment#test
https://thefilibusterblog.com/es/upcoming-isekai-anime-releases-for-2025-latest-announcements/
https://fyuu.net/new-isekai-anime-2025#content
https://www.facebook.com
https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F%2F&src=sdkpreparse https://twitter.com/intent/tweet?text=Every%20New%20Isekai%20Anime%20Announced%20For%202025%20%28So%20Far%29&url=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F
https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F&title=Every%20New%20Isekai%20Anime%20Announced%20For%202025%20%28So%20Far%29&source=gamerant.com&summary=Isekai%20anime%20is%20inescapable%2C%20with%20each%20season%20containing%20a%20couple%20of%20shows.%20Here%20are%20the%202025%20isekai%20anime%20series%20announced%20so%20far. https://bsky.app/intent/compose?text=https%3A%2F%2Fgamerant.com%2Fnew-isekai-anime-2025%2F
Searxng link: http://jethros-macbook-air.local:8888/search?q=Top+10+isekai+anime+2025+with+release+date%2C+synopsis%2C+number+of+episode%2C+airing+status&format=json&pageno=1&safesearch=2&language=en&engines=google%2Cbrave%2Cduckduckgo%2Cbing%2Cyahoo
"""
        expected = """
/
"Database" "Threads"
Searxng link:
"""
        result = remove_links(text_with_sample_links)
        # Normalize whitespace for comparison
        result_normalized = re.sub(r'\s+', ' ', result.strip())
        expected_normalized = re.sub(r'\s+', ' ', expected.strip())
        assert result_normalized == expected_normalized

    def test_single_slash_preserved(self):
        """Given text with single '/', When remove_links is called, Then '/' is preserved."""
        result = remove_links("/")
        expected = "/"
        assert result == expected

    def test_path_preserved(self):
        """Given text with paths, When remove_links is called, Then paths are removed but surrounding text preserved."""
        result = remove_links('/db "Database" /threads/')
        expected = '"Database" '
        result_normalized = re.sub(r'\s+', ' ', result.strip())
        expected_normalized = re.sub(r'\s+', ' ', expected.strip())
        assert result_normalized == expected_normalized

    def test_full_urls_removed(self):
        """Given text with full URLs, When remove_links is called, Then URLs are removed."""
        result = remove_links("Visit https://example.com and http://test.org")
        expected = "Visit  and "
        assert result == expected

    def test_complex_url_with_params(self):
        """Given URL with query params and fragments, When remove_links is called, Then entire URL is removed."""
        url = "https://example.com/path?param=value&test=123#fragment"
        result = remove_links(url)
        expected = ""
        assert result == expected

    def test_mixed_content(self):
        """Given mixed text with URLs and normal text, When remove_links is called, Then only URLs removed."""
        text = "Check this https://site.com/page and this /path but keep normal text."
        result = remove_links(text)
        expected = "Check this  and this  but keep normal text."
        assert result == expected

    def test_no_urls(self):
        """Given text with no URLs, When remove_links is called, Then text unchanged."""
        text = "Just normal text without any links or paths."
        result = remove_links(text)
        expected = text
        assert result == expected

    def test_empty_string(self):
        """Given empty string, When remove_links is called, Then empty string returned."""
        result = remove_links("")
        expected = ""
        assert result == expected

    def test_only_whitespace(self):
        """Given only whitespace, When remove_links is called, Then whitespace preserved."""
        result = remove_links("   \n\t  ")
        expected = "   \n\t  "
        assert result == expected

    def test_path_with_query(self):
        """Given path with query params, When remove_links is called, Then entire path removed."""
        result = remove_links("/sample-with-param?q=test")
        expected = ""
        assert result == expected