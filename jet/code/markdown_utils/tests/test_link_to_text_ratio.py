import unittest
from unittest.mock import patch
from jet.code.markdown_utils._markdown_analyzer import link_to_text_ratio


class TestLinkToTextRatio(unittest.TestCase):
    @patch('jet.code.markdown_utils._preprocessors.clean_markdown_links')
    def test_empty_input(self, mock_clean):
        """Test handling of empty input string."""
        mock_clean.return_value = ""
        result = link_to_text_ratio("")
        self.assertEqual(result, {
            'ratio': 0.0,
            'is_link_heavy': False,
            'link_chars': 0,
            'total_chars': 0,
            'cleaned_text_length': 0
        })

    @patch('jet.code.markdown_utils._preprocessors.clean_markdown_links')
    def test_no_links(self, mock_clean):
        """Test text with no links."""
        input_text = "This is plain text without links."
        mock_clean.return_value = "This is plain text without links"
        result = link_to_text_ratio(input_text, threshold=0.5)
        total_chars = len("Thisisplaintextwithoutlinks")
        self.assertEqual(result, {
            'ratio': 0.0,
            'is_link_heavy': False,
            'link_chars': 0,
            'total_chars': total_chars,
            'cleaned_text_length': total_chars
        })

    @patch('jet.code.markdown_utils._preprocessors.clean_markdown_links')
    def test_only_links(self, mock_clean):
        """Test text consisting only of links."""
        input_text = "[link](http://example.com)"
        mock_clean.return_value = "link"
        result = link_to_text_ratio(input_text, threshold=0.5)
        total_chars = len("[link](http://example.com)")
        cleaned_length = len("link")
        link_chars = total_chars - cleaned_length
        ratio = link_chars / total_chars
        self.assertEqual(result, {
            'ratio': ratio,
            'is_link_heavy': True,
            'link_chars': link_chars,
            'total_chars': total_chars,
            'cleaned_text_length': cleaned_length
        })
        self.assertGreater(result['ratio'], 0.5)

    @patch('jet.code.markdown_utils._preprocessors.clean_markdown_links')
    def test_mixed_content(self, mock_clean):
        """Test text with both links and regular text."""
        input_text = "Text with [link](http://example.com) and more text."
        mock_clean.return_value = "Text with link and more text"
        result = link_to_text_ratio(input_text, threshold=0.3)
        total_chars = len("Textwith[link](http://example.com)andmoretext")
        cleaned_length = len("Textwithlinkandmoretext")
        link_chars = total_chars - cleaned_length
        ratio = link_chars / total_chars
        self.assertEqual(result, {
            'ratio': ratio,
            'is_link_heavy': True,
            'link_chars': link_chars,
            'total_chars': total_chars,
            'cleaned_text_length': cleaned_length
        })

    @patch('jet.code.markdown_utils._preprocessors.clean_markdown_links')
    def test_custom_threshold(self, mock_clean):
        """Test with a custom threshold value."""
        input_text = "Some [link](http://example.com) text."
        mock_clean.return_value = "Some link text"
        result = link_to_text_ratio(input_text, threshold=0.1)
        total_chars = len("Some[link](http://example.com)text")
        cleaned_length = len("Somelinktext")
        link_chars = total_chars - cleaned_length
        ratio = link_chars / total_chars
        self.assertEqual(result, {
            'ratio': ratio,
            'is_link_heavy': True,
            'link_chars': link_chars,
            'total_chars': total_chars,
            'cleaned_text_length': cleaned_length
        })

    @patch('jet.code.markdown_utils._preprocessors.clean_markdown_links')
    def test_whitespace_handling(self, mock_clean):
        """Test handling of excessive whitespace."""
        input_text = "Text  with [link](http://example.com)  spaces."
        mock_clean.return_value = "Text with link spaces"
        result = link_to_text_ratio(input_text, threshold=0.5)
        total_chars = len("Textwith[link](http://example.com)spaces")
        cleaned_length = len("Textwithlinkspaces")
        link_chars = total_chars - cleaned_length
        ratio = link_chars / total_chars
        self.assertEqual(result, {
            'ratio': ratio,
            'is_link_heavy': True,  # Ratio=0.55 meets threshold=0.5
            'link_chars': link_chars,
            'total_chars': total_chars,
            'cleaned_text_length': cleaned_length
        })


if __name__ == '__main__':
    unittest.main()
