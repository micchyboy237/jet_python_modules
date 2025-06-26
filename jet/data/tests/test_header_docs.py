from typing import List, Optional, Union, Literal, TypedDict, Dict, Any
from pydantic import BaseModel
import pytest
from jet.data.header_docs import HeaderDocs, HeaderNode, TextNode, MarkdownToken, ListItem, ListMeta, CodeMeta, TableMeta, Node


class TestHeaderDocs:
    def test_empty_token_list(self):
        # Given: An empty list of tokens
        tokens: List[MarkdownToken] = []

        # When: Converting the empty list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The result should be a HeaderDocs with an empty root
        assert result.root == []
        assert result.as_texts() == []
        assert result.as_nodes() == []
        assert result.as_tree() == {"root": []}

    def test_single_header(self):
        # Given: A single header token
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "Title",
                "level": 1,
                "meta": {},
                "line": 1
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The result should have one header node with correct attributes
        assert len(result.root) == 1
        assert isinstance(result.root[0], HeaderNode)
        assert result.root[0].title == "Title"
        assert result.root[0].level == 1
        assert result.root[0].children == []
        assert result.root[0].line == 1
        assert result.root[0].parent_id is None
        assert result.root[0].id.startswith("auto_")
        assert result.as_texts() == ["# Title"]
        assert len(result.as_nodes()) == 1
        assert result.as_nodes()[0] == result.root[0]
        assert result.as_tree() == {
            "root": [{
                "id": result.root[0].id,
                "parent_id": None,
                "line": 1,
                "type": "header",
                "title": "Title",
                "level": 1,
                "children": []
            }]
        }

    def test_header_with_text_nodes(self):
        # Given: A header followed by paragraph and code tokens
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "Overview",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "paragraph",
                "content": "This is a paragraph.",
                "level": None,
                "meta": {},
                "line": 2
            },
            {
                "type": "code",
                "content": "print('Hello')",
                "level": None,
                "meta": {"language": "python"},
                "line": 3
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The header should have two child text nodes with correct attributes
        assert len(result.root) == 1
        assert isinstance(result.root[0], HeaderNode)
        assert result.root[0].title == "Overview"
        assert len(result.root[0].children) == 2
        assert isinstance(result.root[0].children[0], TextNode)
        assert result.root[0].children[0].type == "paragraph"
        assert result.root[0].children[0].content == "This is a paragraph."
        assert result.root[0].children[0].parent_id == result.root[0].id
        assert isinstance(result.root[0].children[1], TextNode)
        assert result.root[0].children[1].type == "code"
        assert result.root[0].children[1].content == "``` python\nprint('Hello')\n```"
        assert result.root[0].children[1].parent_id == result.root[0].id
        assert result.as_texts() == [
            "# Overview", "This is a paragraph.", "``` python\nprint('Hello')\n```"]
        assert len(result.as_nodes()) == 3
        assert result.as_nodes() == [
            result.root[0], result.root[0].children[0], result.root[0].children[1]]
        assert result.as_tree() == {
            "root": [{
                "id": result.root[0].id,
                "parent_id": None,
                "line": 1,
                "type": "header",
                "title": "Overview",
                "level": 1,
                "children": [
                    {
                        "id": result.root[0].children[0].id,
                        "parent_id": result.root[0].id,
                        "line": 2,
                        "type": "paragraph",
                        "content": "This is a paragraph.",
                        "meta": {}
                    },
                    {
                        "id": result.root[0].children[1].id,
                        "parent_id": result.root[0].id,
                        "line": 3,
                        "type": "code",
                        "content": "``` python\nprint('Hello')\n```",
                        "meta": {"language": "python"}
                    }
                ]
            }]
        }

    def test_nested_headers(self):
        # Given: Tokens with nested headers and text nodes
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "Main",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "Subsection",
                "level": 2,
                "meta": {},
                "line": 2
            },
            {
                "type": "paragraph",
                "content": "Subsection text",
                "level": None,
                "meta": {},
                "line": 3
            },
            {
                "type": "header",
                "content": "Another Main",
                "level": 1,
                "meta": {},
                "line": 4
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The tree should have correct header nesting and parent-child relationships
        assert len(result.root) == 2
        assert isinstance(result.root[0], HeaderNode)
        assert result.root[0].title == "Main"
        assert len(result.root[0].children) == 1
        assert isinstance(result.root[0].children[0], HeaderNode)
        assert result.root[0].children[0].title == "Subsection"
        assert result.root[0].children[0].parent_id == result.root[0].id
        assert len(result.root[0].children[0].children) == 1
        assert isinstance(result.root[0].children[0].children[0], TextNode)
        assert result.root[0].children[0].children[0].content == "Subsection text"
        assert result.root[0].children[0].children[0].parent_id == result.root[0].children[0].id
        assert isinstance(result.root[1], HeaderNode)
        assert result.root[1].title == "Another Main"
        assert result.root[1].parent_id is None
        assert result.as_texts() == [
            "# Main", "## Subsection", "Subsection text", "# Another Main"]
        assert len(result.as_nodes()) == 4
        assert result.as_nodes() == [result.root[0], result.root[0].children[0],
                                     result.root[0].children[0].children[0], result.root[1]]

    def test_text_nodes_without_headers(self):
        # Given: Only text nodes without headers
        tokens: List[MarkdownToken] = [
            {
                "type": "paragraph",
                "content": "Intro",
                "level": None,
                "meta": {},
                "line": 1
            },
            {
                "type": "unordered_list",
                "content": "",
                "level": None,
                "meta": {
                    "items": [
                        {"text": "Item 1", "task_item": False}
                    ]
                },
                "line": 2
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The root should contain only text nodes with correct content
        assert len(result.root) == 2
        assert all(isinstance(node, TextNode) for node in result.root)
        assert result.root[0].content == "Intro"
        assert result.root[0].parent_id is None
        assert result.root[1].content == "* Item 1"
        assert result.root[1].parent_id is None
        assert result.as_texts() == ["Intro", "* Item 1"]
        assert len(result.as_nodes()) == 2
        assert result.as_nodes() == [result.root[0], result.root[1]]
        assert result.as_tree() == {
            "root": [
                {
                    "id": result.root[0].id,
                    "parent_id": None,
                    "line": 1,
                    "type": "paragraph",
                    "content": "Intro",
                    "meta": {}
                },
                {
                    "id": result.root[1].id,
                    "parent_id": None,
                    "line": 2,
                    "type": "unordered_list",
                    "content": "* Item 1",
                    "meta": {"items": [{"text": "Item 1", "task_item": False}]}
                }
            ]
        }

    def test_complex_nesting_with_lists_and_tables(self):
        # Given: Tokens with headers, lists, and tables
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "Project",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "unordered_list",
                "content": "",
                "level": None,
                "meta": {
                    "items": [
                        {"text": "Task 1", "task_item": True, "checked": False}
                    ]
                },
                "line": 2
            },
            {
                "type": "header",
                "content": "Details",
                "level": 2,
                "meta": {},
                "line": 3
            },
            {
                "type": "table",
                "content": "",
                "level": None,
                "meta": {
                    "header": ["Name", "Value"],
                    "rows": [["A", "1"]]
                },
                "line": 4
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The tree should correctly nest lists and tables
        assert len(result.root) == 1
        assert isinstance(result.root[0], HeaderNode)
        assert len(result.root[0].children) == 2
        assert isinstance(result.root[0].children[0], TextNode)
        assert result.root[0].children[0].content == "* [ ] Task 1"
        assert result.root[0].children[0].parent_id == result.root[0].id
        assert isinstance(result.root[0].children[1], HeaderNode)
        assert result.root[0].children[1].title == "Details"
        assert result.root[0].children[1].parent_id == result.root[0].id
        assert result.root[0].children[1].children[0].content == "| Name | Value |\n| ---- | ----- |\n| A    | 1     |"
        assert result.as_texts() == [
            "# Project",
            "* [ ] Task 1",
            "## Details",
            "| Name | Value |\n| ---- | ----- |\n| A    | 1     |"
        ]
        assert len(result.as_nodes()) == 4
        assert result.as_nodes() == [result.root[0], result.root[0].children[0],
                                     result.root[0].children[1], result.root[0].children[1].children[0]]

    def test_multiple_header_levels(self):
        # Given: Tokens with headers from level 1 to 4
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "Main",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "Section",
                "level": 2,
                "meta": {},
                "line": 2
            },
            {
                "type": "header",
                "content": "Subsection",
                "level": 3,
                "meta": {},
                "line": 3
            },
            {
                "type": "header",
                "content": "Detail",
                "level": 4,
                "meta": {},
                "line": 4
            },
            {
                "type": "paragraph",
                "content": "Detail text",
                "level": None,
                "meta": {},
                "line": 5
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The tree should have headers nested correctly
        assert len(result.root) == 1
        assert result.root[0].title == "Main"
        assert result.root[0].children[0].title == "Section"
        assert result.root[0].children[0].children[0].title == "Subsection"
        assert result.root[0].children[0].children[0].children[0].title == "Detail"
        assert result.root[0].children[0].children[0].children[0].children[0].content == "Detail text"
        assert result.as_texts() == [
            "# Main", "## Section", "### Subsection", "#### Detail", "Detail text"]
        assert len(result.as_nodes()) == 5

    def test_consecutive_same_level_headers(self):
        # Given: Multiple headers of the same level
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "Section 1",
                "level": 2,
                "meta": {},
                "line": 1
            },
            {
                "type": "paragraph",
                "content": "Text 1",
                "level": None,
                "meta": {},
                "line": 2
            },
            {
                "type": "header",
                "content": "Section 2",
                "level": 2,
                "meta": {},
                "line": 3
            },
            {
                "type": "paragraph",
                "content": "Text 2",
                "level": None,
                "meta": {},
                "line": 4
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The headers should be siblings with correct content
        assert len(result.root) == 2
        assert result.root[0].title == "Section 1"
        assert result.root[0].children[0].content == "Text 1"
        assert result.root[1].title == "Section 2"
        assert result.root[1].children[0].content == "Text 2"
        assert result.as_texts() == ["## Section 1",
                                     "Text 1", "## Section 2", "Text 2"]
        assert len(result.as_nodes()) == 4

    def test_empty_content_header(self):
        # Given: A header with empty content
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "paragraph",
                "content": "Text",
                "level": None,
                "meta": {},
                "line": 2
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The empty header should have correct derived content
        assert len(result.root) == 1
        assert result.root[0].title == ""
        assert result.root[0].children[0].content == "Text"
        assert result.as_texts() == ["", "Text"]
        assert len(result.as_nodes()) == 2

    def test_complex_meta_data(self):
        # Given: Tokens with complex meta data
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "Resources",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "code",
                "content": "x = 1",
                "level": None,
                "meta": {"language": "python", "code_type": "indented"},
                "line": 2
            },
            {
                "type": "unordered_list",
                "content": "",
                "level": None,
                "meta": {
                    "items": [
                        {"text": "Item 1", "task_item": True, "checked": True},
                        {"text": "Item 2", "task_item": False}
                    ]
                },
                "line": 3
            },
            {
                "type": "table",
                "content": "",
                "level": None,
                "meta": {
                    "header": ["ID", "Name"],
                    "rows": [["1", "Alice"], ["2", "Bob"]]
                },
                "line": 4
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The tree should preserve complex meta data
        assert len(result.root) == 1
        assert len(result.root[0].children) == 3
        assert result.root[0].children[0].content == "``` python\nx = 1\n```"
        assert result.root[0].children[1].content == "* [x] Item 1\n* Item 2"
        assert result.root[0].children[2].content == "| ID | Name  |\n| -- | ----- |\n| 1  | Alice |\n| 2  | Bob   |"
        assert result.as_texts() == [
            "# Resources",
            "``` python\nx = 1\n```",
            "* [x] Item 1\n* Item 2",
            "| ID | Name  |\n| -- | ----- |\n| 1  | Alice |\n| 2  | Bob   |"
        ]
        assert len(result.as_nodes()) == 4

    def test_single_text_node_with_complex_meta(self):
        # Given: A single text node with complex list meta
        tokens: List[MarkdownToken] = [
            {
                "type": "ordered_list",
                "content": "",
                "level": None,
                "meta": {
                    "items": [
                        {"text": "Step 1", "task_item": False},
                        {"text": "Step 2", "task_item": False}
                    ]
                },
                "line": 1
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The root should contain the text node with correct content
        assert len(result.root) == 1
        assert isinstance(result.root[0], TextNode)
        assert result.root[0].content == "1. Step 1\n2. Step 2"
        assert result.as_texts() == ["1. Step 1\n2. Step 2"]
        assert len(result.as_nodes()) == 1

    def test_header_level_gaps(self):
        # Given: Headers with gaps in levels
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "Main",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "Detail",
                "level": 3,
                "meta": {},
                "line": 2
            },
            {
                "type": "paragraph",
                "content": "Detail text",
                "level": None,
                "meta": {},
                "line": 3
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The tree should handle level gaps correctly
        assert len(result.root) == 1
        assert result.root[0].title == "Main"
        assert result.root[0].children[0].title == "Detail"
        assert result.root[0].children[0].level == 3
        assert result.root[0].children[0].children[0].content == "Detail text"
        assert result.as_texts() == ["# Main", "### Detail", "Detail text"]
        assert len(result.as_nodes()) == 3

    def test_nested_headers_without_siblings(self):
        # Given: A deep hierarchy of headers with no siblings
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "Main",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "Section",
                "level": 2,
                "meta": {},
                "line": 2
            },
            {
                "type": "header",
                "content": "Subsection",
                "level": 3,
                "meta": {},
                "line": 3
            },
            {
                "type": "header",
                "content": "Detail",
                "level": 4,
                "meta": {},
                "line": 4
            },
            {
                "type": "paragraph",
                "content": "Detail text",
                "level": None,
                "meta": {},
                "line": 5
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The tree should have a single nested path
        assert len(result.root) == 1
        assert result.root[0].title == "Main"
        assert len(result.root[0].children) == 1
        assert result.root[0].children[0].title == "Section"
        assert len(result.root[0].children[0].children) == 1
        assert result.root[0].children[0].children[0].title == "Subsection"
        assert len(result.root[0].children[0].children[0].children) == 1
        assert result.root[0].children[0].children[0].children[0].title == "Detail"
        assert result.as_texts() == [
            "# Main", "## Section", "### Subsection", "#### Detail", "Detail text"]
        assert len(result.as_nodes()) == 5

    def test_nested_headers_with_siblings(self):
        # Given: A hierarchy with sibling headers
        tokens: List[MarkdownToken] = [
            {
                "type": "header",
                "content": "Main",
                "level": 1,
                "meta": {},
                "line": 1
            },
            {
                "type": "header",
                "content": "Section A",
                "level": 2,
                "meta": {},
                "line": 2
            },
            {
                "type": "header",
                "content": "Subsection A1",
                "level": 3,
                "meta": {},
                "line": 3
            },
            {
                "type": "paragraph",
                "content": "Text A1",
                "level": None,
                "meta": {},
                "line": 4
            },
            {
                "type": "header",
                "content": "Section B",
                "level": 2,
                "meta": {},
                "line": 5
            },
            {
                "type": "header",
                "content": "Subsection B1",
                "level": 3,
                "meta": {},
                "line": 6
            },
            {
                "type": "paragraph",
                "content": "Text B1",
                "level": None,
                "meta": {},
                "line": 7
            }
        ]

        # When: Converting the token list to a HeaderDocs
        result = HeaderDocs.from_tokens(tokens)

        # Then: The tree should have sibling headers with correct nesting
        assert len(result.root) == 1
        assert result.root[0].title == "Main"
        assert len(result.root[0].children) == 2
        assert result.root[0].children[0].title == "Section A"
        assert result.root[0].children[1].title == "Section B"
        assert result.root[0].children[0].children[0].title == "Subsection A1"
        assert result.root[0].children[1].children[0].title == "Subsection B1"
        assert result.as_texts() == [
            "# Main",
            "## Section A",
            "### Subsection A1",
            "Text A1",
            "## Section B",
            "### Subsection B1",
            "Text B1"
        ]
        assert len(result.as_nodes()) == 7

    def test_derive_text_header(self):
        # Given: A header token
        token: MarkdownToken = {
            "type": "header",
            "content": "Title",
            "level": 2,
            "meta": {},
            "line": 1
        }

        # When: Deriving text
        result = HeaderDocs.derive_text(token)

        # Then: The text should have correct header prefix
        assert result == "## Title"

    def test_derive_text_empty_header(self):
        # Given: An empty header token
        token: MarkdownToken = {
            "type": "header",
            "content": "",
            "level": 1,
            "meta": {},
            "line": 1
        }

        # When: Deriving text
        result = HeaderDocs.derive_text(token)

        # Then: The text should be empty with prefix
        assert result == ""

    def test_derive_text_unordered_list(self):
        # Given: An unordered list token
        token: MarkdownToken = {
            "type": "unordered_list",
            "content": "",
            "level": None,
            "meta": {
                "items": [
                    {"text": "Item 1", "task_item": True, "checked": True},
                    {"text": "Item 2", "task_item": False}
                ]
            },
            "line": 1
        }

        # When: Deriving text
        result = HeaderDocs.derive_text(token)

        # Then: The text should have correct list format
        assert result == "* [x] Item 1\n* Item 2"

    def test_derive_text_ordered_list(self):
        # Given: An ordered list token
        token: MarkdownToken = {
            "type": "ordered_list",
            "content": "",
            "level": None,
            "meta": {
                "items": [
                    {"text": "Step 1", "task_item": False},
                    {"text": "Step 2", "task_item": False}
                ]
            },
            "line": 1
        }

        # When: Deriving text
        result = HeaderDocs.derive_text(token)

        # Then: The text should have correct ordered list format
        assert result == "1. Step 1\n2. Step 2"

    def test_derive_text_table(self):
        # Given: A table token
        token: MarkdownToken = {
            "type": "table",
            "content": "",
            "level": None,
            "meta": {
                "header": ["ID", "Name"],
                "rows": [["1", "Alice"], ["2", "Bob"]]
            },
            "line": 1
        }

        # When: Deriving text
        result = HeaderDocs.derive_text(token)

        # Then: The text should have correct table format
        assert result == "| ID | Name  |\n| -- | ----- |\n| 1  | Alice |\n| 2  | Bob   |"

    def test_derive_text_code(self):
        # Given: A code token
        token: MarkdownToken = {
            "type": "code",
            "content": "x = 1",
            "level": None,
            "meta": {"language": "python"},
            "line": 1
        }

        # When: Deriving text
        result = HeaderDocs.derive_text(token)

        # Then: The text should have correct code block format
        assert result == "``` python\nx = 1\n```"

    def test_derive_text_paragraph(self):
        # Given: A paragraph token
        token: MarkdownToken = {
            "type": "paragraph",
            "content": "Hello world",
            "level": None,
            "meta": {},
            "line": 1
        }

        # When: Deriving text
        result = HeaderDocs.derive_text(token)

        # Then: The text should match the content
        assert result == "Hello world"

    def test_derive_text_empty_meta(self):
        # Given: A list token with empty meta
        token: MarkdownToken = {
            "type": "unordered_list",
            "content": "",
            "level": None,
            "meta": {},
            "line": 1
        }

        # When: Deriving text
        result = HeaderDocs.derive_text(token)

        # Then: The text should be empty
        assert result == ""
