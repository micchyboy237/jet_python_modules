import pytest

from jet.code.markdown_utils._preprocessors import clean_markdown_links


class TestCleanMarkdownLinks:
    def test_single_text_link(self):
        # Given: A string with a single markdown text link
        input_text = "Visit [Google](https://www.google.com) now"
        expected = "Visit Google now"

        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: The text link should be replaced with the display text
        assert result == expected

    def test_single_image_link(self):
        # Given: A string with a single markdown image link
        input_text = "Image ![alt](https://example.com/image.jpg)"
        expected = "Image"

        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: The image link should be removed
        assert result == expected

    def test_mixed_text_and_image_links(self):
        # Given: A string with both text and image links
        input_text = "Check [site](https://site.com) and ![img](image.png)"
        expected = "Check site and"

        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: Text link should be replaced with display text, image link removed
        assert result == expected

    def test_multiline_text_with_links(self):
        # Given: Multiline text with multiple links
        input_text = """Line 1 with [link1](http://link1.com)
        Line 2 with ![img](img.jpg)
        Line 3 with [link2](http://link2.com)"""
        expected = """Line 1 with link1
        Line 2 with
        Line 3 with link2"""

        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: All text links should be replaced with display text, image links removed
        assert result == expected

    def test_empty_string(self):
        # Given: An empty string
        input_text = ""
        expected = ""

        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: The result should be an empty string
        assert result == expected

    def test_no_links(self):
        # Given: Text without any markdown links
        input_text = "Just plain text here"
        expected = "Just plain text here"

        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: The text should remain unchanged
        assert result == expected

    def test_nested_brackets(self):
        # Given: Text with nested brackets in link text
        input_text = "See [nested [brackets]](https://example.com)"
        expected = "See nested [brackets]"

        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: The link should be replaced with the display text
        assert result == expected

    def test_multiple_links_same_line(self):
        # Given: Multiple links on the same line
        input_text = "[link1](http://link1.com) and [link2](http://link2.com)"
        expected = "link1 and link2"

        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: All links should be replaced with their display text
        assert result == expected

    def test_special_characters_in_url(self):
        # Given: A link with special characters in the URL
        input_text = "Go to [site](https://site.com?query=1¶m=2)"
        expected = "Go to site"

        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: The link should be replaced with the display text
        assert result == expected

    def test_within_list_item(self):
        input_text = "* [Go to twitter](https://twitter.com/animesoulking)"
        expected = "* Go to twitter"

        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: The link should be replaced with the display text
        assert result == expected

    def test_empty_alt(self):
        # Given a markdown text with an empty alt text link
        input_text = "[ ](https://twitter.com/animesoulking)"
        # When the link is processed
        expected = "https://twitter.com/animesoulking"
        # Then the result should be the URL
        result = clean_markdown_links(input_text)
        assert result == expected

    def test_empty_alt_with_space_in_between_bracket_and_parenthesis(self):
        # Given a markdown text with an empty alt text link with space between brackets and parenthesis
        input_text = "[ ] (https://twitter.com/animesoulking)"
        # When the link is processed
        expected = "https://twitter.com/animesoulking"
        # Then the result should be the URL
        result = clean_markdown_links(input_text)
        assert result == expected

    def test_multiline_markdown_with_various_elements(self):
        # Given: Complex multiline markdown with various elements
        input_text = """
Sample title

# Project Overview
Welcome to our **project**! This is an `introduction` to our work, featuring a [website](https://project.com).

![Project Logo](https://project.com/logo.png)

> **Note**: Always check the [docs](https://docs.project.com) for updates.

## Features
- [ ] Task 1: Implement login
- [x] Task 2: Add dashboard
- Task 3: Optimize performance

### Technical Details
```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

#### API Endpoints
| Endpoint       | Method | Description           |
|----------------|--------|-----------------------|
| /api/users     | GET    | Fetch all users       |
| /api/users/{id}| POST   | Create a new user     |

##### Inline Code
Use `print("Hello")` for quick debugging.

###### Emphasis
*Italic*, **bold**, and ***bold italic*** text are supported.

<div class="alert">This is an HTML block.</div>
<span class="badge">New</span> inline HTML.

[^1]: This is a footnote reference.
[^1]: Footnote definition here.

## Unordered list
- List item 1
    - Nested item
- List item 2
- List item 3

## Ordered list
1. Ordered item 1
2. Ordered item 2
3. Ordered item 3

## Inline HTML
<span class="badge">New</span> inline HTML
"""
        expected = """
Sample title

# Project Overview
Welcome to our **project**! This is an `introduction` to our work, featuring a website.



> **Note**: Always check the docs for updates.

## Features
- [ ] Task 1: Implement login
- [x] Task 2: Add dashboard
- Task 3: Optimize performance

### Technical Details
```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

#### API Endpoints
| Endpoint | Method | Description |
|----------------|--------|-----------------------|
| /api/users | GET | Fetch all users |
| /api/users/{id}| POST | Create a new user |

##### Inline Code
Use `print("Hello")` for quick debugging.

###### Emphasis
*Italic*, **bold**, and ***bold italic*** text are supported.

<div class="alert">This is an HTML block.</div>
<span class="badge">New</span> inline HTML.

[^1]: This is a footnote reference.
[^1]: Footnote definition here.

## Unordered list
- List item 1
    - Nested item
- List item 2
- List item 3

## Ordered list
1. Ordered item 1
2. Ordered item 2
3. Ordered item 3

## Inline HTML
<span class="badge">New</span> inline HTML
"""
        # When: We clean the markdown links
        result = clean_markdown_links(input_text)

        # Then: Text links should be replaced, image links removed, and newlines preserved
        assert result == expected
