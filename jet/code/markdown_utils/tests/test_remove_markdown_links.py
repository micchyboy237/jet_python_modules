from jet.code.markdown_utils._preprocessors import remove_markdown_links

class TestRemoveMarkdownLinks:
    def test_single_text_link(self):
        # Given: A string with a single markdown text link
        input_text = "Visit [Google](https://www.google.com) now"
        expected = "Visit Google now"
        
        # When: We call remove_markdown_links
        result = remove_markdown_links(input_text)
        
        # Then: The link is replaced with its label
        assert result == expected

    def test_single_image_link(self):
        # Given: A string with a single markdown image link
        input_text = "Image ![alt](https://example.com/image.jpg)"
        expected = "Image alt"
        
        # When: We call remove_markdown_links
        result = remove_markdown_links(input_text)
        
        # Then: The image link is replaced with its alt text
        assert result == expected

    def test_mixed_text_and_image_links(self):
        # Given: A string with both text and image links
        input_text = "Check [site](https://site.com) and ![img](image.png)"
        expected = "Check site and img"
        
        # When: We call remove_markdown_links
        result = remove_markdown_links(input_text)
        
        # Then: Both text and image links are replaced with their labels
        assert result == expected

    def test_multiline_text_with_links(self):
        # Given: A multiline string with text and image links
        input_text = """Line 1 with [link1](http://link1.com)
Line 2 with ![img](img.jpg)
Line 3 with [link2](http://link2.com)"""
        expected = """Line 1 with link1
Line 2 with img
Line 3 with link2"""
        
        # When: We call remove_markdown_links
        result = remove_markdown_links(input_text)
        
        # Then: All links are replaced with their labels, preserving line structure
        assert result == expected

    def test_empty_string(self):
        # Given: An empty string
        input_text = ""
        expected = ""
        
        # When: We call remove_markdown_links
        result = remove_markdown_links(input_text)
        
        # Then: The result is an empty string
        assert result == expected

    def test_no_links(self):
        # Given: A string with no links
        input_text = "Just plain text here"
        expected = "Just plain text here"
        
        # When: We call remove_markdown_links
        result = remove_markdown_links(input_text)
        
        # Then: The text remains unchanged
        assert result == expected

    def test_empty_label(self):
        # Given: A string with a text link with empty label
        input_text = "Link [](https://example.com)"
        expected = "Link "
        
        # When: We call remove_markdown_links
        result = remove_markdown_links(input_text)
        
        # Then: The link is removed entirely
        assert result == expected

    def test_empty_image_alt(self):
        # Given: A string with an image link with empty alt text
        input_text = "Image ![ ](https://example.com/image.jpg)"
        expected = "Image "
        
        # When: We call remove_markdown_links
        result = remove_markdown_links(input_text)
        
        # Then: The image link is removed entirely
        assert result == expected

    def test_nested_brackets(self):
        # Given: A string with nested brackets in a link label
        input_text = "See [nested [brackets]](https://example.com)"
        expected = "See nested [brackets]"
        
        # When: We call remove_markdown_links
        result = remove_markdown_links(input_text)
        
        # Then: The link is replaced with its full label, including nested brackets
        assert result == expected

    def test_multiple_links_same_line(self):
        # Given: A string with multiple links on the same line
        input_text = "[link1](http://link1.com) and ![img](image.png) and [link2](http://link2.com)"
        expected = "link1 and img and link2"
        
        # When: We call remove_markdown_links
        result = remove_markdown_links(input_text)
        
        # Then: All links are replaced with their labels
        assert result == expected