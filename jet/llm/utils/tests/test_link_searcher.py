import unittest
import time
from typing import List, TypedDict, Optional
from jet.llm.utils.link_searcher import Link, LinkSearchResult, search_links


class TestSearchLinks(unittest.TestCase):
    def setUp(self):
        self.links: List[Link] = [
            {'url': 'https://example.com/Python-Programming?tutorial=True#Introduction',
                'text': 'Python tutorial'},
            {'url': 'https://example.com/data-science?category=ML#Overview',
                'text': 'Machine learning guide'},
            {'url': 'https://example.com/coding-bootcamp?level=beginner'},
            {'url': 'https://example.com/post/12345?session=xyz#comments',
                'text': 'User comments'},
            {'url': 'https://pydata.org/python-data-analysis#Tutorial'},
            {'url': 'https://example.com/special%20chars?key=value%20test#frag%20ment',
                'text': 'Special chars test'},
            {'url': 'https://example.com'},
            {'url': 'https://example.com/path/?#empty'},
            {'url': 'http:/malformed-url'},
            {'url': 'https://example.com/duplicate?key=value&key=other#frag',
                'text': 'Duplicate params'},
            {'url': 'https://example.com/long-path/' +
                'x' * 1000 + '?key=' + 'y' * 1000},
            {'url': 'https://example.com/%C3%BCber?text=%C3%A4pfel#caf%C3%A9',
                'text': 'German text über äpfel café'}
        ]

    def test_empty_query(self):
        """Test empty or whitespace query"""
        result = search_links(self.links, "")
        expected: List[LinkSearchResult] = []
        self.assertEqual(result, expected)
        result = search_links(self.links, "   ")
        self.assertEqual(result, expected)

    def test_cased_paths(self):
        """Test case-insensitive matching"""
        result = search_links(self.links, "python")
        self.assertTrue(any(
            r['url'] == 'https://example.com/Python-Programming?tutorial=True#Introduction' for r in result))
        self.assertTrue(any(
            r['url'] == 'https://pydata.org/python-data-analysis#Tutorial' for r in result))
        self.assertTrue(all(0 <= r['score'] <= 1 for r in result))

    def test_url_decoding_special_chars(self):
        """Test decoding of percent-encoded special characters"""
        result = search_links(self.links, "chars")
        self.assertTrue(any(
            r['url'] == 'https://example.com/special%20chars?key=value%20test#frag%20ment' for r in result))
        self.assertTrue(any(r['text'] == 'Special chars test' for r in result))
        self.assertTrue(all(0 <= r['score'] <= 1 for r in result))

    def test_url_decoding_non_ascii(self):
        """Test decoding of non-ASCII characters"""
        result = search_links(self.links, "über")
        self.assertTrue(any(
            r['url'] == 'https://example.com/%C3%BCber?text=%C3%A4pfel#caf%C3%A9' for r in result))
        self.assertTrue(
            any(r['text'] == 'German text über äpfel café' for r in result))
        result = search_links(self.links, "äpfel")
        self.assertTrue(any(
            r['url'] == 'https://example.com/%C3%BCber?text=%C3%A4pfel#caf%C3%A9' for r in result))
        result = search_links(self.links, "café")
        self.assertTrue(any(
            r['url'] == 'https://example.com/%C3%BCber?text=%C3%A4pfel#caf%C3%A9' for r in result))
        self.assertTrue(all(0 <= r['score'] <= 1 for r in result))

    def test_text_context(self):
        """Test using optional text context"""
        result = search_links(self.links, "tutorial")
        self.assertTrue(any(
            r['url'] == 'https://example.com/Python-Programming?tutorial=True#Introduction' and r['text'] == 'Python tutorial' for r in result))
        self.assertTrue(all(0 <= r['score'] <= 1 for r in result))

    def test_semantic_search(self):
        """Test semantic understanding"""
        result = search_links(self.links, "programming")
        self.assertTrue(any(
            r['url'] == 'https://example.com/Python-Programming?tutorial=True#Introduction' for r in result))
        self.assertTrue(all(0 <= r['score'] <= 1 for r in result))

    def test_typo_handling(self):
        """Test typo handling"""
        result = search_links(self.links, "pythn")
        self.assertTrue(any(
            r['url'] == 'https://example.com/Python-Programming?tutorial=True#Introduction' for r in result))
        self.assertTrue(all(0 <= r['score'] <= 1 for r in result))

    def test_noisy_params(self):
        """Test exclusion of noisy parameters"""
        result = search_links(self.links, "xyz")
        self.assertFalse(any(
            r['url'] == 'https://example.com/post/12345?session=xyz#comments' for r in result))
        result = search_links(self.links, "comments")
        self.assertTrue(any(
            r['url'] == 'https://example.com/post/12345?session=xyz#comments' and r['text'] == 'User comments' for r in result))
        self.assertTrue(all(0 <= r['score'] <= 1 for r in result))

    def test_duplicate_query_params(self):
        """Test duplicate query parameters"""
        result = search_links(self.links, "value other")
        self.assertTrue(any(
            r['url'] == 'https://example.com/duplicate?key=value&key=other#frag' for r in result))
        self.assertTrue(all(0 <= r['score'] <= 1 for r in result))

    def test_empty_components(self):
        """Test URLs with missing components"""
        result = search_links(self.links, "example.com")
        self.assertTrue(any(r['url'] == 'https://example.com' for r in result))
        result = search_links(self.links, "empty")
        self.assertTrue(
            any(r['url'] == 'https://example.com/path/?#empty' for r in result))
        self.assertTrue(all(0 <= r['score'] <= 1 for r in result))

    def test_performance(self):
        """Test performance with 1000 links"""
        large_links: List[Link] = [
            {'url': f'https://example.com/post/{i}?tag=test{i}#section{i}', 'text': f'Post {i} content'} for i in range(1000)]
        start_time = time.time()
        result = search_links(large_links, "test")
        duration = time.time() - start_time
        self.assertLess(duration, 0.5, f"Search took too long: {duration}s")
        self.assertEqual(len(result), 1000)
        self.assertTrue(all(0 <= r['score'] <= 1 for r in result))


if __name__ == '__main__':
    unittest.main()
