import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import pytest

from jet.code.markdown_analyzer import analyze_markdown, MarkdownAnalysisResult


@pytest.fixture
def sample_md_content():
    """Provide sample markdown content for tests."""
    return """
# Project Overview
Welcome to our **project**! This is an `introduction` to our work, featuring a [website](https://project.com).

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

- List item 1
  - Nested item
- List item 2

1. Ordered list
2. Another item
"""


class TestAnalyzeMarkdown:
    """Tests for the analyze_markdown function."""

    def test_given_string_content_then_analyzes_and_returns_results(self, sample_md_content):
        """Given markdown content as a string, it should analyze and return results."""
        expected_results: MarkdownAnalysisResult = {
            "headers": [
                {"level": 1, "text": "Project Overview"},
                {"level": 2, "text": "Features"},
                {"level": 3, "text": "Technical Details"},
                {"level": 4, "text": "API Endpoints"},
                {"level": 5, "text": "Inline Code"},
                {"level": 6, "text": "Emphasis"}
            ],
            "paragraphs": [
                {"text": "Welcome to our **project**! This is an `introduction` to our work, featuring a [website](https://project.com)."},
                {"text": "Use `print(\"Hello\")` for quick debugging."},
                {"text": "*Italic*, **bold**, and ***bold italic*** text are supported."},
                {"text": "[^1]: This is a footnote reference.\n[^1]: Footnote definition here."}
            ],
            "blockquotes": [
                {"text": "**Note**: Always check the [docs](https://docs.project.com) for updates."}
            ],
            "code_blocks": [
                {"language": "python",
                    "content": 'def greet(name: str) -> str:\n    return f"Hello, {name}!"'}
            ],
            "lists": [
                [
                    {"text": "Task 1: Implement login", "type": "unordered",
                        "task_item": True, "checked": False, "level": None},
                    {"text": "Task 2: Add dashboard", "type": "unordered",
                        "task_item": True, "checked": True, "level": None},
                    {"text": "Task 3: Optimize performance",
                        "type": "unordered", "task_item": False, "level": None}
                ],
                [
                    {"text": "List item", "type": "unordered",
                        "task_item": False, "level": None},
                    {"text": "Nested", "type": "unordered",
                        "task_item": False, "level": 2},
                    {"text": "List item 2", "type": "unordered",
                        "task_item": False, "level": None}
                ],
                [
                    {"text": "Ordered list", "type": "ordered",
                        "task_item": False, "level": None},
                    {"text": "Another item", "type": "ordered",
                        "task_item": False, "level": None}
                ]
            ],
            "tables": [
                [
                    {"Endpoint": "/api/users", "Method": "GET",
                        "Description": "Fetch all users"},
                    {"Endpoint": "/api/users/{id}", "Method": "POST",
                        "Description": "Create a new user"}
                ]
            ],
            "links": [
                {"text": "website", "url": "https://project.com"},
                {"text": "docs", "url": "https://docs.project.com"}
            ],
            "footnotes": [
                {"ref": "1", "text": "This is a footnote reference."},
                {"ref": "1", "text": "Footnote definition here."}
            ],
            "inline_code": ["introduction", 'print("Hello")'],
            "emphasis": [
                {"text": "project", "type": "bold"},
                {"text": "Italic", "type": "italic"},
                {"text": "bold", "type": "bold"},
                {"text": "bold italic", "type": "bold_italic"}
            ],
            "task_items": [
                "Task 1: Implement login",
                "Task 2: Add dashboard",
                "Task 3: Optimize performance"
            ],
            "html_blocks": ['<div class="alert">This is an HTML block.</div>'],
            "html_inline": ['<span class="badge">New</span>'],
            "tokens_sequential": [
                {"type": "header", "content": "Project Overview"},
                {"type": "paragraph",
                    "content": "Welcome to our **project**! This is an `introduction` to our work, featuring a [website](https://project.com)."},
                {"type": "italic", "content": "project"},
                {"type": "inline_code", "content": "introduction"},
                {"type": "link", "content": "website",
                    "url": "https://project.com"},
                {"type": "blockquote",
                    "content": "**Note**: Always check the [docs](https://docs.project.com) for updates."},
                {"type": "italic", "content": "Note"},
                {"type": "link", "content": "docs",
                    "url": "https://docs.project.com"},
                {"type": "header", "content": "Features"},
                {"type": "unordered_list",
                    "content": "Task 1: Implement login\nTask 2: Add dashboard\nTask 3: Optimize performance"},
                {"type": "task_item", "content": "Task 1: Implement login",
                    "checked": False},
                {"type": "task_item", "content": "Task 2: Add dashboard", "checked": True},
                {"type": "list_item",
                    "content": "Task 3: Optimize performance", "checked": None},
                {"type": "header", "content": "Technical Details"},
                {"type": "code_block",
                    "content": 'def greet(name: str) -> str:\n    return f"Hello, {name}!"'},
                {"type": "header", "content": "API Endpoints"},
                {"type": "table", "content": ""},
                {"type": "header", "content": "Inline Code"},
                {"type": "paragraph",
                    "content": 'Use `print("Hello")` for quick debugging.'},
                {"type": "inline_code", "content": 'print("Hello")'},
                {"type": "header", "content": "Emphasis"},
                {"type": "paragraph",
                    "content": "*Italic*, **bold**, and ***bold italic*** text are supported."},
                {"type": "italic", "content": "Italic"},
                {"type": "italic", "content": "bold"},
                {"type": "italic", "content": "*bold italic"},
                {"type": "html_block", "content": '<div class="alert">This is an HTML block.</div>\n<span class="badge">New</span>'},
                {"type": "paragraph",
                    "content": "[^1]: This is a footnote reference.\n[^1]: Footnote definition here."},
                {"type": "unordered_list",
                    "content": "List item\nNested\nList item 2"},
                {"type": "list_item", "content": "List item", "checked": None},
                {"type": "list_item", "content": "Nested", "checked": None},
                {"type": "list_item", "content": "List item 2", "checked": None},
                {"type": "ordered_list", "content": "Ordered list\nAnother item"},
                {"type": "list_item", "content": "Ordered list", "checked": None},
                {"type": "list_item", "content": "Another item", "checked": None}
            ],
            "word_count": {"word_count": 47},
            "char_count": [374],
            "analysis": {"summary": "Comprehensive markdown with diverse elements"}
        }
        mock_analyzer = Mock()
        mock_analyzer.identify_headers.return_value = expected_results["headers"]
        mock_analyzer.identify_paragraphs.return_value = expected_results["paragraphs"]
        mock_analyzer.identify_blockquotes.return_value = expected_results["blockquotes"]
        mock_analyzer.identify_code_blocks.return_value = expected_results["code_blocks"]
        mock_analyzer.identify_lists.return_value = [
            [
                {"text": "Task 1: Implement login",
                    "task_item": True, "checked": False},
                {"text": "Task 2: Add dashboard",
                    "task_item": True, "checked": True},
                {"text": "Task 3: Optimize performance", "task_item": False}
            ],
            [
                {"text": "List item", "task_item": False},
                {"text": "Nested", "task_item": False},
                {"text": "List item 2", "task_item": False}
            ],
            [
                {"text": "Ordered list", "task_item": False},
                {"text": "Another item", "task_item": False}
            ]
        ]
        mock_analyzer.identify_tables.return_value = expected_results["tables"]
        mock_analyzer.identify_links.return_value = expected_results["links"]
        mock_analyzer.identify_footnotes.return_value = expected_results["footnotes"]
        mock_analyzer.identify_inline_code.return_value = expected_results["inline_code"]
        mock_analyzer.identify_emphasis.return_value = expected_results["emphasis"]
        mock_analyzer.identify_task_items.return_value = expected_results["task_items"]
        mock_analyzer.identify_html_blocks.return_value = expected_results["html_blocks"]
        mock_analyzer.identify_html_inline.return_value = expected_results["html_inline"]
        mock_analyzer.get_tokens_sequential.return_value = expected_results["tokens_sequential"]
        mock_analyzer.count_words.return_value = expected_results["word_count"]["word_count"]
        mock_analyzer.count_characters.return_value = expected_results["char_count"][0]
        mock_analyzer.analyse.return_value = expected_results["analysis"]
        with patch("jet.code.markdown_analyzer.MarkdownAnalyzer", return_value=mock_analyzer):
            result = analyze_markdown(sample_md_content)
            assert result == expected_results

    def test_given_file_path_then_analyzes_and_returns_results(self):
        """Given a file path, it should read the file, analyze, and return results."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write("""
        # Project Overview
        Welcome to our **project**! Featuring a [website](https://project.com).

        > **Note**: Check [docs](https://docs.project.com).

        ## Features
        - [ ] Task 1: Implement login
        - [x] Task 2: Add dashboard

        ### Technical Details
        ```python
        def greet():
        pass
        ```

        #### API Endpoints
        | Endpoint | Method | Description |
        |----------|--------|-------------|
        | /api/users | GET | Fetch users |

        ##### Inline Code
        Use `print()`.

        ###### Emphasis
        *Italic* text.

        <div class="alert">HTML block.</div>
        <span class="badge">Inline</span>

        [^1]: Footnote.
        [^1]: Definition.

        - List item
        - Nested
        1. Ordered item
        """)
            temp_md_path = Path(temp_file.name)
        expected_results: MarkdownAnalysisResult = {
            "headers": [
                {"level": 1, "text": "Project Overview"},
                {"level": 2, "text": "Features"},
                {"level": 3, "text": "Technical Details"},
                {"level": 4, "text": "API Endpoints"},
                {"level": 5, "text": "Inline Code"},
                {"level": 6, "text": "Emphasis"}
            ],
            "paragraphs": [
                {"text": "Welcome to our **project**! Featuring a [website](https://project.com)."},
                {"text": "Use `print()`."},
                {"text": "*Italic* text."},
                {"text": "[^1]: Footnote.\n[^1]: Definition."}
            ],
            "blockquotes": [
                {"text": "**Note**: Check [docs](https://docs.project.com)."}
            ],
            "code_blocks": [
                {"language": "python", "content": "def greet():\n    pass"}
            ],
            "lists": [
                [
                    {"text": "Task 1: Implement login", "type": "unordered",
                        "task_item": True, "checked": False, "level": None},
                    {"text": "Task 2: Add dashboard", "type": "unordered",
                        "task_item": True, "checked": True, "level": None}
                ],
                [
                    {"text": "List item", "type": "unordered",
                        "task_item": False, "level": None},
                    {"text": "Nested", "type": "unordered",
                        "task_item": False, "level": 2},
                    {"text": "Ordered item", "type": "ordered",
                        "task_item": False, "level": None}
                ]
            ],
            "tables": [
                [
                    {"Endpoint": "/api/users", "Method": "GET",
                        "Description": "Fetch users"}
                ]
            ],
            "links": [
                {"text": "website", "url": "https://project.com"},
                {"text": "docs", "url": "https://docs.project.com"}
            ],
            "footnotes": [
                {"ref": "1", "text": "Footnote."},
                {"ref": "1", "text": "Definition."}
            ],
            "inline_code": ["print()"],
            "emphasis": [
                {"text": "project", "type": "bold"},
                {"text": "Italic", "type": "italic"}
            ],
            "task_items": [
                "Task 1: Implement login",
                "Task 2: Add dashboard"
            ],
            "html_blocks": ['<div class="alert">HTML block.</div>'],
            "html_inline": ['<span class="badge">Inline</span>'],
            "tokens_sequential": [
                {"type": "header", "content": "Project Overview"},
                {"type": "paragraph",
                    "content": "Welcome to our **project**! Featuring a [website](https://project.com)."},
                {"type": "italic", "content": "project"},
                {"type": "link", "content": "website",
                    "url": "https://project.com"},
                {"type": "blockquote",
                    "content": "**Note**: Check [docs](https://docs.project.com)."},
                {"type": "italic", "content": "Note"},
                {"type": "link", "content": "docs",
                    "url": "https://docs.project.com"},
                {"type": "header", "content": "Features"},
                {"type": "unordered_list",
                    "content": "Task 1: Implement login\nTask 2: Add dashboard"},
                {"type": "task_item", "content": "Task 1: Implement login",
                    "checked": False},
                {"type": "task_item", "content": "Task 2: Add dashboard", "checked": True},
                {"type": "header", "content": "Technical Details"},
                {"type": "code_block", "content": "def greet():\n    pass"},
                {"type": "header", "content": "API Endpoints"},
                {"type": "table", "content": ""},
                {"type": "header", "content": "Inline Code"},
                {"type": "paragraph", "content": "Use `print()`."},
                {"type": "inline_code", "content": "print()"},
                {"type": "header", "content": "Emphasis"},
                {"type": "paragraph", "content": "*Italic* text."},
                {"type": "italic", "content": "Italic"},
                {"type": "html_block",
                    "content": '<div class="alert">HTML block.</div>\n<span class="badge">Inline</span>'},
                {"type": "paragraph",
                    "content": "[^1]: Footnote.\n[^1]: Definition."},
                {"type": "unordered_list", "content": "List item\nNested"},
                {"type": "list_item", "content": "List item", "checked": None},
                {"type": "list_item", "content": "Nested", "checked": None},
                {"type": "ordered_list", "content": "Ordered item"},
                {"type": "list_item", "content": "Ordered item", "checked": None}
            ],
            "word_count": {"word_count": 30},
            "char_count": [260],
            "analysis": {"summary": "Markdown with multiple elements"}
        }
        mock_analyzer = Mock()
        mock_analyzer.identify_headers.return_value = expected_results["headers"]
        mock_analyzer.identify_paragraphs.return_value = expected_results["paragraphs"]
        mock_analyzer.identify_blockquotes.return_value = expected_results["blockquotes"]
        mock_analyzer.identify_code_blocks.return_value = expected_results["code_blocks"]
        mock_analyzer.identify_lists.return_value = [
            [
                {"text": "Task 1: Implement login",
                    "task_item": True, "checked": False},
                {"text": "Task 2: Add dashboard",
                    "task_item": True, "checked": True}
            ],
            [
                {"text": "List item", "task_item": False},
                {"text": "Nested", "task_item": False},
                {"text": "Ordered item", "task_item": False}
            ]
        ]
        mock_analyzer.identify_tables.return_value = expected_results["tables"]
        mock_analyzer.identify_links.return_value = expected_results["links"]
        mock_analyzer.identify_footnotes.return_value = expected_results["footnotes"]
        mock_analyzer.identify_inline_code.return_value = expected_results["inline_code"]
        mock_analyzer.identify_emphasis.return_value = expected_results["emphasis"]
        mock_analyzer.identify_task_items.return_value = expected_results["task_items"]
        mock_analyzer.identify_html_blocks.return_value = expected_results["html_blocks"]
        mock_analyzer.identify_html_inline.return_value = expected_results["html_inline"]
        mock_analyzer.get_tokens_sequential.return_value = expected_results["tokens_sequential"]
        mock_analyzer.count_words.return_value = expected_results["word_count"]["word_count"]
        mock_analyzer.count_characters.return_value = expected_results["char_count"][0]
        mock_analyzer.analyse.return_value = expected_results["analysis"]
        with patch("jet.code.markdown_analyzer.MarkdownAnalyzer", return_value=mock_analyzer):
            result = analyze_markdown(temp_md_path)
            assert result == expected_results

    def test_given_nonexistent_file_path_then_raises_os_error(self):
        """Given a nonexistent file path, it should raise an OSError."""
        # Arrange
        nonexistent_path = Path("nonexistent.md")
        expected_error = OSError

        # Act & Assert
        with pytest.raises(expected_error):
            analyze_markdown(nonexistent_path)

    def test_given_empty_string_then_analyzes_and_returns_empty_results(self):
        """Given an empty string, it should analyze and return empty results."""
        # Arrange
        empty_content = ""
        expected_results: MarkdownAnalysisResult = {
            "headers": [],
            "paragraphs": [],
            "blockquotes": [],
            "code_blocks": [],
            "lists": [],
            "tables": [],
            "links": [],
            "footnotes": [],
            "inline_code": [],
            "emphasis": [],
            "task_items": [],
            "html_blocks": [],
            "html_inline": [],
            "tokens_sequential": [],
            "word_count": {"word_count": 0},
            "char_count": [0],
            "analysis": {"summary": "Empty markdown"}
        }

        mock_analyzer = Mock()
        mock_analyzer.identify_headers.return_value = expected_results["headers"]
        mock_analyzer.identify_paragraphs.return_value = expected_results["paragraphs"]
        mock_analyzer.identify_blockquotes.return_value = expected_results["blockquotes"]
        mock_analyzer.identify_code_blocks.return_value = expected_results["code_blocks"]
        mock_analyzer.identify_lists.return_value = expected_results["lists"]
        mock_analyzer.identify_tables.return_value = expected_results["tables"]
        mock_analyzer.identify_links.return_value = expected_results["links"]
        mock_analyzer.identify_footnotes.return_value = expected_results["footnotes"]
        mock_analyzer.identify_inline_code.return_value = expected_results["inline_code"]
        mock_analyzer.identify_emphasis.return_value = expected_results["emphasis"]
        mock_analyzer.identify_task_items.return_value = expected_results["task_items"]
        mock_analyzer.identify_html_blocks.return_value = expected_results["html_blocks"]
        mock_analyzer.identify_html_inline.return_value = expected_results["html_inline"]
        mock_analyzer.get_tokens_sequential.return_value = expected_results["tokens_sequential"]
        mock_analyzer.count_words.return_value = expected_results["word_count"]["word_count"]
        mock_analyzer.count_characters.return_value = expected_results["char_count"][0]
        mock_analyzer.analyse.return_value = expected_results["analysis"]

        with patch("jet.code.markdown_analyzer.MarkdownAnalyzer", return_value=mock_analyzer):
            # Act
            result = analyze_markdown(empty_content)

            # Assert
            assert result == expected_results
