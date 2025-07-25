import pytest
from typing import Dict, List, Tuple
from jet.code.splitter_markdown_utils import Header, HeaderLink, get_md_header_contents, extract_markdown_links
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
from jet.vectors.document_types import HeaderDocument


class TestGetMdHeaderContents:
    def test_empty_input(self):
        # Given: An empty Markdown text
        md_text = ""

        # When: The function is called
        result = get_md_header_contents(md_text)

        # Then: The result is an empty list
        expected: List[HeaderDocument] = []
        assert result == expected

    def test_single_header_no_content(self):
        # Given: Markdown with a single header
        md_text = "# Header 1"

        # When: get_md_header_contents is called
        result = get_md_header_contents(md_text)

        # Then: A single HeaderDocument is returned with correct metadata
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="# Header 1",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 1"],
                    "id": result[0].id
                },
                relationships={}
            )
        ]
        assert result[0].text == expected[0].text
        assert result[0].id == expected[0].id
        assert result[0].metadata == expected[0].metadata
        assert result[0].relationships == expected[0].relationships

    def test_single_header_with_content(self):
        # Given: A Markdown text with one header and content
        md_text = "# Header 1\nThis is some content."

        # When: The function is called
        result = get_md_header_contents(md_text)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="# Header 1\nThis is some content.",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "This is some content.",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 1", "This is some content."],
                    "id": result[0].id
                },
                relationships={}
            )
        ]
        assert result[0].text == expected[0].text
        assert result[0].id == expected[0].id
        assert result[0].metadata == expected[0].metadata
        assert result[0].relationships == expected[0].relationships

    def test_nested_headers(self):
        # Given: A Markdown text with nested headers
        md_text = "# Header 1\nContent 1 [link](http://example.com)\n## Header 2\nContent 2"

        # When: The function is called
        result = get_md_header_contents(md_text, ignore_links=True)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 2
        expected = [
            HeaderDocument(
                text="# Header 1\nContent 1 link",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content 1 link",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [
                        {
                            "text": "link",
                            "url": "http://example.com",
                            "caption": None,
                            "start_idx": 12,
                            "end_idx": 35,
                            "line": "Content 1 [link](http://example.com)",
                            "line_idx": 1,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 1", "Content 1 link"],
                    "id": result[0].id
                },
                relationships={
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[1].id)]
                }
            ),
            HeaderDocument(
                text="# Header 2\nContent 2",
                id=result[1].id,
                metadata={
                    "header": "# Header 2",
                    "parent_header": "# Header 1",
                    "header_level": 2,
                    "content": "Content 2",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 2", "Content 2"],
                    "id": result[1].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=result[0].id)
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.text == e.text
            assert r.id == e.id
            assert r.metadata == e.metadata
            assert r.relationships == e.relationships

    def test_nested_headers_with_content(self):
        # Given: Markdown with nested headers and content
        md_text = """# Header 1
    Content 1 [link](http://example.com)
    ## Header 2
    Content 2
    ## Header 3
    Content 3 [link2](http://test.com)"""

        # When: get_md_header_contents is called
        result = get_md_header_contents(md_text, ignore_links=True)

        # Then: HeaderDocuments with correct relationships and metadata
        assert len(result) == 3
        expected = [
            HeaderDocument(
                text="# Header 1\nContent 1 link",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content 1 link",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [
                        {
                            "text": "link",
                            "url": "http://example.com",
                            "caption": None,
                            "start_idx": 12,
                            "end_idx": 35,
                            "line": "Content 1 [link](http://example.com)",
                            "line_idx": 1,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 1", "Content 1 link"],
                    "id": result[0].id
                },
                relationships={
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[1].id),
                        RelatedNodeInfo(node_id=result[2].id)
                    ]
                }
            ),
            HeaderDocument(
                text="# Header 2\nContent 2",
                id=result[1].id,
                metadata={
                    "header": "# Header 2",
                    "parent_header": "# Header 1",
                    "header_level": 2,
                    "content": "Content 2",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 2", "Content 2"],
                    "id": result[1].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=result[0].id)
                }
            ),
            HeaderDocument(
                text="# Header 3\nContent 3 link2",
                id=result[2].id,
                metadata={
                    "header": "# Header 3",
                    "parent_header": "# Header 1",
                    "header_level": 2,
                    "content": "Content 3 link2",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [
                        {
                            "text": "link2",
                            "url": "http://test.com",
                            "caption": None,
                            "start_idx": 45,
                            "end_idx": 66,
                            "line": "Content 3 [link2](http://test.com)",
                            "line_idx": 5,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 3", "Content 3 link2"],
                    "id": result[2].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=result[0].id)
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.text == e.text
            assert r.id == e.id
            assert r.metadata == e.metadata
            assert r.relationships == e.relationships

    def test_content_before_header(self):
        # Given: A Markdown text with content before any header
        md_text = "Intro content [link](http://example.com)\n# Header 1\nContent 1"

        # When: The function is called
        result = get_md_header_contents(md_text, ignore_links=True)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 2
        expected = [
            HeaderDocument(
                text="Intro content link",
                id=result[0].id,
                metadata={
                    "header": "",
                    "parent_header": "",
                    "header_level": 0,
                    "content": "Intro content link",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [
                        {
                            "text": "link",
                            "url": "http://example.com",
                            "caption": None,
                            "start_idx": 13,
                            "end_idx": 36,
                            "line": "Intro content [link](http://example.com)",
                            "line_idx": 0,
                            "is_heading": True
                        }
                    ],
                    "texts": ["Intro content link"],
                    "id": result[0].id
                },
                relationships={
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[1].id)]
                }
            ),
            HeaderDocument(
                text="# Header 1\nContent 1",
                id=result[1].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content 1",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 1", "Content 1"],
                    "id": result[1].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=result[0].id)
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.text == e.text
            assert r.id == e.id
            assert r.metadata == e.metadata
            assert r.relationships == e.relationships

    def test_with_links(self):
        # Given: A Markdown text with headers and links
        md_text = "# Header 1\nContent with [link](http://example.com)"

        # When: The function is called with ignore_links=True
        result = get_md_header_contents(md_text, ignore_links=True)

        # Then: The result includes the link information
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="# Header 1\nContent with link",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content with link",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [
                        {
                            "text": "link",
                            "url": "http://example.com",
                            "caption": None,
                            "start_idx": 13,
                            "end_idx": 36,
                            "line": "Content with [link](http://example.com)",
                            "line_idx": 1,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 1", "Content with link"],
                    "id": result[0].id
                },
                relationships={}
            )
        ]
        assert result[0].text == expected[0].text
        assert result[0].id == expected[0].id
        assert result[0].metadata == expected[0].metadata
        assert result[0].relationships == expected[0].relationships

    def test_with_links_and_base_url(self):
        # Given: Markdown with headers and links
        md_text = """# Header 1
    [Link](http://example.com)
    ## Header 2
    [Relative Link](/page)"""
        base_url = "http://base.com"

        # When: get_md_header_contents is called with ignore_links=True
        result = get_md_header_contents(
            md_text, ignore_links=True, base_url=base_url)

        # Then: HeaderDocuments with links in metadata
        assert len(result) == 2
        expected = [
            HeaderDocument(
                text="# Header 1\nLink",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Link",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": base_url,
                    "links": [
                        {
                            "text": "Link",
                            "url": "http://example.com",
                            "caption": None,
                            "start_idx": 11,
                            "end_idx": 34,
                            "line": "[Link](http://example.com)",
                            "line_idx": 1,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 1", "Link"],
                    "id": result[0].id
                },
                relationships={
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[1].id)]
                }
            ),
            HeaderDocument(
                text="# Header 2\nRelative Link",
                id=result[1].id,
                metadata={
                    "header": "# Header 2",
                    "parent_header": "# Header 1",
                    "header_level": 2,
                    "content": "Relative Link",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": base_url,
                    "links": [
                        {
                            "text": "Relative Link",
                            "url": "http://base.com/page",
                            "caption": None,
                            "start_idx": 28,
                            "end_idx": 46,
                            "line": "[Relative Link](/page)",
                            "line_idx": 3,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 2", "Relative Link"],
                    "id": result[1].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=result[0].id)
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.text == e.text
            assert r.id == e.id
            assert r.metadata == e.metadata
            assert r.relationships == e.relationships

    def test_html_input(self):
        # Given: An HTML input that should be converted to Markdown
        html_text = "<title>Page Title</title><h1>Header 1</h1><p>Content 1</p>"

        # When: The function is called with HTML input
        result = get_md_header_contents(html_text)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 2
        expected = [
            HeaderDocument(
                text="Page Title",
                id=result[0].id,
                metadata={
                    "header": "Page Title",
                    "parent_header": "",
                    "header_level": 0,
                    "content": "",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["Page Title"],
                    "id": result[0].id
                },
                relationships={
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[1].id)]
                }
            ),
            HeaderDocument(
                text="# Header 1\nContent 1",
                id=result[1].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "Page Title",
                    "header_level": 1,
                    "content": "Content 1",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 1", "Content 1"],
                    "id": result[1].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=result[0].id)
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.text == e.text
            assert r.id == e.id
            assert r.metadata == e.metadata
            assert r.relationships == e.relationships

    def test_html_with_title_and_content(self):
        # Given: An HTML input with title, content, and headers
        html_text = "<title>Page Title</title><p>Intro content</p><h1>Header 1</h1><p>Content 1</p>"

        # When: The function is called with HTML input
        result = get_md_header_contents(html_text)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 2
        expected = [
            HeaderDocument(
                text="Page Title\nIntro content",
                id=result[0].id,
                metadata={
                    "header": "Page Title",
                    "parent_header": "",
                    "header_level": 0,
                    "content": "Intro content",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["Page Title", "Intro content"],
                    "id": result[0].id
                },
                relationships={
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[1].id)]
                }
            ),
            HeaderDocument(
                text="# Header 1\nContent 1",
                id=result[1].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "Page Title",
                    "header_level": 1,
                    "content": "Content 1",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 1", "Content 1"],
                    "id": result[1].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=result[0].id)
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.text == e.text
            assert r.id == e.id
            assert r.metadata == e.metadata
            assert r.relationships == e.relationships

    def test_custom_headers(self):
        # Given: A Markdown text with custom header pattern
        md_text = "=== Header 1 ===\nContent 1"
        headers_to_split_on = [(r"^={3}\s(.+)\s={3}$", "header")]

        # When: The function is called with custom headers
        result = get_md_header_contents(
            md_text, headers_to_split_on=headers_to_split_on)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="=== Header 1 ===\nContent 1",
                id=result[0].id,
                metadata={
                    "header": "=== Header 1 ===",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content 1",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["=== Header 1 ===", "Content 1"],
                    "id": result[0].id
                },
                relationships={}
            )
        ]
        assert result[0].text == expected[0].text
        assert result[0].id == expected[0].id
        assert result[0].metadata == expected[0].metadata
        assert result[0].relationships == expected[0].relationships

    def test_base_url_for_links(self):
        # Given: A Markdown text with relative links and a base URL
        md_text = "# Header 1\nContent with [link](/path)"
        base_url = "http://example.com"

        # When: The function is called with base_url and ignore_links=False
        result = get_md_header_contents(
            md_text, ignore_links=False, base_url=base_url)

        # Then: The result resolves relative links correctly
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="# Header 1\nContent with [link](/path)",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content with [link](/path)",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": base_url,
                    "links": [
                        {
                            "text": "link",
                            "url": "http://example.com/path",
                            "caption": None,
                            "start_idx": -1,
                            "end_idx": -1,
                            "line": "Content with [link](/path)",
                            "line_idx": 2,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 1", "Content with [link](/path)"],
                    "id": result[0].id
                },
                relationships={}
            )
        ]
        assert result[0].text == expected[0].text
        assert result[0].id == expected[0].id
        assert result[0].metadata == expected[0].metadata
        assert result[0].relationships == expected[0].relationships

    def test_nested_headers_skipping_level(self):
        # Given: A Markdown text with headers skipping a level
        md_text = "# Header 1\nContent 1\n### Header 3\nContent 3"

        # When: The function is called
        result = get_md_header_contents(md_text)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 2
        expected = [
            HeaderDocument(
                text="# Header 1\nContent 1",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content 1",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 1", "Content 1"],
                    "id": result[0].id
                },
                relationships={
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[1].id)]
                }
            ),
            HeaderDocument(
                text="### Header 3\nContent 3",
                id=result[1].id,
                metadata={
                    "header": "### Header 3",
                    "parent_header": "# Header 1",
                    "header_level": 3,
                    "content": "Content 3",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["### Header 3", "Content 3"],
                    "id": result[1].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=result[0].id)
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.text == e.text
            assert r.id == e.id
            assert r.metadata == e.metadata
            assert r.relationships == e.relationships

    def test_consecutive_headers(self):
        # Given: A Markdown text with consecutive headers
        md_text = "# Header 1\n## Header 2\n### Header 3\nContent"

        # When: The function is called
        result = get_md_header_contents(md_text)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 3
        expected = [
            HeaderDocument(
                text="# Header 1",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 1"],
                    "id": result[0].id
                },
                relationships={
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[1].id)]
                }
            ),
            HeaderDocument(
                text="## Header 2",
                id=result[1].id,
                metadata={
                    "header": "## Header 2",
                    "parent_header": "# Header 1",
                    "header_level": 2,
                    "content": "",
                    "doc_index": 1,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["## Header 2"],
                    "id": result[1].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(node_id=result[0].id),
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[2].id)]
                }
            ),
            HeaderDocument(
                text="### Header 3\nContent",
                id=result[2].id,
                metadata={
                    "header": "### Header 3",
                    "parent_header": "## Header 2",
                    "header_level": 3,
                    "content": "Content",
                    "doc_index": 2,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["### Header 3", "Content"],
                    "id": result[2].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=result[1].id)
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.text == e.text
            assert r.id == e.id
            assert r.metadata == e.metadata
            assert r.relationships == e.relationships

    def test_empty_header(self):
        # Given: A Markdown text with an empty header
        md_text = "##\nContent"

        # When: The function is called
        result = get_md_header_contents(md_text)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="##\nContent",
                id=result[0].id,
                metadata={
                    "header": "##",
                    "parent_header": "",
                    "header_level": 2,
                    "content": "Content",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["##", "Content"],
                    "id": result[0].id
                },
                relationships={}
            )
        ]
        assert result[0].text == expected[0].text
        assert result[0].id == expected[0].id
        assert result[0].metadata == expected[0].metadata
        assert result[0].relationships == expected[0].relationships

    def test_headers_no_content(self):
        # Given: A Markdown text with headers but no content
        md_text = "# Header 1\n# Header 2"

        # When: The function is called
        result = get_md_header_contents(md_text)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 2
        expected = [
            HeaderDocument(
                text="# Header 1",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 1"],
                    "id": result[0].id
                },
                relationships={}
            ),
            HeaderDocument(
                text="# Header 2",
                id=result[1].id,
                metadata={
                    "header": "# Header 2",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 2"],
                    "id": result[1].id
                },
                relationships={}
            )
        ]
        for r, e in zip(result, expected):
            assert r.text == e.text
            assert r.id == e.id
            assert r.metadata == e.metadata
            assert r.relationships == e.relationships

    def test_malformed_markdown(self):
        # Given: A Markdown text with malformed headers
        md_text = "#Header\nContent\n###Header\nContent 2"

        # When: The function is called
        result = get_md_header_contents(md_text)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 2
        expected = [
            HeaderDocument(
                text="#Header\nContent",
                id=result[0].id,
                metadata={
                    "header": "#Header",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["#Header", "Content"],
                    "id": result[0].id
                },
                relationships={
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[1].id)]
                }
            ),
            HeaderDocument(
                text="###Header\nContent 2",
                id=result[1].id,
                metadata={
                    "header": "###Header",
                    "parent_header": "#Header",
                    "header_level": 3,
                    "content": "Content 2",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["###Header", "Content 2"],
                    "id": result[1].id
                },
                relationships={
                    NodeRelationship.PARENT: RelatedNodeInfo(
                        node_id=result[0].id)
                }
            )
        ]
        for r, e in zip(result, expected):
            assert r.text == e.text
            assert r.id == e.id
            assert r.metadata == e.metadata
            assert r.relationships == e.relationships

    def test_custom_empty_header(self):
        # Given: A Markdown text with an empty custom header
        md_text = "===  ===\nContent"
        headers_to_split_on = [(r"^={3}\s*(.*)\s*={3}$", "header")]

        # When: The function is called with custom headers
        result = get_md_header_contents(
            md_text, headers_to_split_on=headers_to_split_on)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="===  ===\nContent",
                id=result[0].id,
                metadata={
                    "header": "===  ===",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["===  ===", "Content"],
                    "id": result[0].id
                },
                relationships={}
            )
        ]
        assert result[0].text == expected[0].text
        assert result[0].id == expected[0].id
        assert result[0].metadata == expected[0].metadata
        assert result[0].relationships == expected[0].relationships

    def test_links_in_header(self):
        # Given: A Markdown text with a link in the header
        md_text = "# Header [link](http://example.com)\nContent"

        # When: The function is called with ignore_links=True
        result = get_md_header_contents(md_text, ignore_links=True)

        # Then: The result includes the link information
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="# Header link\nContent",
                id=result[0].id,
                metadata={
                    "header": "# Header link",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [
                        {
                            "text": "link",
                            "url": "http://example.com",
                            "caption": None,
                            "start_idx": 8,
                            "end_idx": 31,
                            "line": "# Header [link](http://example.com)",
                            "line_idx": 0,
                            "is_heading": True
                        }
                    ],
                    "texts": ["# Header link", "Content"],
                    "id": result[0].id
                },
                relationships={}
            )
        ]
        assert result[0].text == expected[0].text
        assert result[0].id == expected[0].id
        assert result[0].metadata == expected[0].metadata
        assert result[0].relationships == expected[0].relationships

    def test_multiple_newlines_between_content(self):
        # Given: A Markdown text with multiple newlines between content lines
        md_text = "# Header 1\nContent 1 [link](http://example.com)\n\n\nContent 2"

        # When: The function is called
        result = get_md_header_contents(md_text, ignore_links=True)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="# Header 1\nContent 1 link\n\nContent 2",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content 1 link\n\nContent 2",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [
                        {
                            "text": "link",
                            "url": "http://example.com",
                            "caption": None,
                            "start_idx": 12,
                            "end_idx": 35,
                            "line": "Content 1 [link](http://example.com)",
                            "line_idx": 1,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 1", "Content 1 link", "", "", "Content 2"],
                    "id": result[0].id
                },
                relationships={}
            )
        ]
        assert result[0].text == expected[0].text
        assert result[0].id == expected[0].id
        assert result[0].metadata == expected[0].metadata
        assert result[0].relationships == expected[0].relationships

    def test_header_with_image_link(self) -> None:
        # Given: A markdown text with a header containing an image link
        input_text = "###### ![Game Rant logo](https://screenrant.com/db/tv-show/the-beginning-after-the-end/)\n\nSample content"
        expected: List[Dict] = [
            {
                "header": "###### Game Rant logo",
                "content": "\nSample content",
                "text": "###### Game Rant logo\n\nSample content"
            }
        ]

        # When: The function is called with ignore_links=True
        result_docs: List[HeaderDocument] = get_md_header_contents(
            input_text, ignore_links=True)
        result = [{
            "header": doc["metadata"]["header"],
            "content": doc["metadata"]["content"],
            "text": doc["text"]
        } for doc in result_docs]

        # Then: The header is correctly cleaned, and content is preserved
        assert result == expected


class TestExtractMarkdownLinks:
    def test_ignore_links_true(self):
        # Given: Text with markdown and plain URLs
        input_text = "See [link1](http://example.com) and http://plain.com"
        expected_text = "See link1 and "
        expected_links: List[dict] = []

        # When: Extract links with ignore_links=True
        result_links, result_text = extract_markdown_links(
            input_text, ignore_links=True)

        # Then: Links are removed, markdown link replaced with label, plain URL removed
        assert result_text == expected_text
        assert result_links == expected_links

    def test_ignore_links_false(self):
        # Given: Text with markdown and plain URLs
        input_text = "See [link1](http://example.com) and http://plain.com"
        expected_text = "See [link1](http://example.com) and http://plain.com"
        expected_links = [
            {
                "text": "link1",
                "url": "http://example.com",
                "caption": None,
                "start_idx": 4,
                "end_idx": 27,
                "line": "[link1](http://example.com)",
                "line_idx": 0,
                "is_heading": True
            },
            {
                "text": "",
                "url": "http://plain.com",
                "caption": None,
                "start_idx": 31,
                "end_idx": 47,
                "line": "http://plain.com",
                "line_idx": 0,
                "is_heading": True
            }
        ]

        # When: Extract links with ignore_links=False
        result_links, result_text = extract_markdown_links(
            input_text, ignore_links=False)

        # Then: Links are extracted with cleaned URLs, text retains original links
        assert result_text == expected_text
        assert len(result_links) == len(expected_links)
        for result, expected in zip(result_links, expected_links):
            assert result["text"] == expected["text"]
            assert result["url"] == expected["url"]
            assert result["caption"] == expected["caption"]
            assert result["line_idx"] == expected["line_idx"]
            assert result["is_heading"] == expected["is_heading"]

    def test_ignore_links_true_with_caption(self):
        # Given: Text with a markdown link with caption
        input_text = 'Content with [link](http://example.com "Caption")'
        expected_text = "Content with link"
        expected_links: List[dict] = []

        # When: Extract links with ignore_links=True
        result_links, result_text = extract_markdown_links(
            input_text, ignore_links=True)

        # Then: Markdown link is replaced with label, no links returned
        assert result_text == expected_text
        assert result_links == expected_links

    def test_ignore_links_false_with_base_url(self):
        # Given: Text with a relative markdown link and base URL
        input_text = "See [link1](/path/page)"
        base_url = "http://example.com"
        expected_text = "See link1"
        expected_links = [
            {
                "text": "link1",
                "url": "http://example.com/path/page",
                "caption": None,
                "start_idx": 4,
                "end_idx": 22,
                "line": "[link1](http://example.com/path/page)",
                "line_idx": 0,
                "is_heading": True
            }
        ]

        # When: Extract links with ignore_links=False and base_url
        result_links, result_text = extract_markdown_links(
            input_text, base_url=base_url, ignore_links=False)

        # Then: Relative URL is resolved, link extracted, text replaced with label
        assert result_text == expected_text
        assert len(result_links) == len(expected_links)
        for result, expected in zip(result_links, expected_links):
            assert result["text"] == expected["text"]
            assert result["url"] == expected["url"]
            assert result["caption"] == expected["caption"]
            assert result["line_idx"] == expected["line_idx"]
            assert result["is_heading"] == expected["is_heading"]

    def test_empty_input(self):
        # Given: Empty input text
        input_text = ""
        expected_text = ""
        expected_links: List[dict] = []

        # When: Extract links with ignore_links=True
        result_links, result_text = extract_markdown_links(
            input_text, ignore_links=True)

        # Then: Empty text and links returned
        assert result_text == expected_text
        assert result_links == expected_links

    def test_no_links(self):
        # Given: Text with no links
        input_text = "Just plain text"
        expected_text = "Just plain text"
        expected_links: List[dict] = []

        # When: Extract links with ignore_links=True
        result_links, result_text = extract_markdown_links(
            input_text, ignore_links=True)

        # Then: Text unchanged, no links returned
        assert result_text == expected_text
        assert result_links == expected_links
