import pytest
from jet.code.markdown_utils._preprocessors import link_to_text_ratio


class TestLinkToTextRatio:
    def test_single_text_link(self):
        # Given a markdown text with a single text link
        input_text = "Visit [Google](https://www.google.com) now!"
        # When calculating the link-to-text ratio
        expected = {
            # 'Visit' (5) + 'Google' (6) + 'now' (3) = 14; link chars = 26 - 14 = 12
            'ratio': 20 / 26,
            'is_link_heavy': True,  # ratio > 0.5
            'link_chars': 12,
            'total_chars': 26,
            'cleaned_text_length': 14
        }
        # Then the result should match the expected dictionary
        result = link_to_text_ratio(input_text)
        assert result == expected

    def test_single_image_link(self):
        # Given a markdown text with a single image link
        input_text = "Image ![alt](https://example.com/image.jpg)"
        # When calculating the link-to-text ratio
        expected = {
            # 'Image' (5) + 'alt' (3) = 8; link chars = 30 - 5 = 25
            'ratio': 25 / 30,
            'is_link_heavy': True,  # ratio > 0.5
            'link_chars': 25,
            'total_chars': 30,
            'cleaned_text_length': 5
        }
        # Then the result should match the expected dictionary
        result = link_to_text_ratio(input_text)
        assert result == expected

    def test_empty_alt_link(self):
        # Given a markdown text with an empty alt link
        input_text = "[ ](https://twitter.com/animesoulking)"
        # When calculating the link-to-text ratio
        expected = {
            'ratio': 0.0,  # All chars are link-related, but cleaned text is the URL
            'is_link_heavy': False,
            'link_chars': 0,
            'total_chars': 20,  # Alphanumeric chars in 'https twitter com animesoulking'
            'cleaned_text_length': 20
        }
        # Then the result should match the expected dictionary
        result = link_to_text_ratio(input_text)
        assert result == expected

    def test_no_links(self):
        # Given a markdown text with no links
        input_text = "Just plain text here!!!"
        # When calculating the link-to-text ratio
        expected = {
            'ratio': 0.0,
            'is_link_heavy': False,
            'link_chars': 0,
            'total_chars': 15,  # Alphanumeric chars in 'Justplaintexthere'
            'cleaned_text_length': 15
        }
        # Then the result should match the expected dictionary
        result = link_to_text_ratio(input_text)
        assert result == expected
