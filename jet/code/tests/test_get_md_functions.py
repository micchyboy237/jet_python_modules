import pytest
from typing import List, Tuple
from jet.code.splitter_markdown_utils import Header, HeaderLink, get_md_header_contents
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
        md_text = "# Header 1\nContent 1\n## Header 2\nContent 2"

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
Content 1
## Header 2
Content 2
## Header 3
Content 3"""

        # When: get_md_header_contents is called
        result = get_md_header_contents(md_text)

        # Then: HeaderDocuments with correct relationships and metadata
        assert len(result) == 3
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
                text="# Header 3\nContent 3",
                id=result[2].id,
                metadata={
                    "header": "# Header 3",
                    "parent_header": "# Header 1",
                    "header_level": 2,
                    "content": "Content 3",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 3", "Content 3"],
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
        md_text = "Intro content\n# Header 1\nContent 1"

        # When: The function is called
        result = get_md_header_contents(md_text)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 2
        expected = [
            HeaderDocument(
                text="Intro content",
                id=result[0].id,
                metadata={
                    "header": "",
                    "parent_header": "",
                    "header_level": 0,
                    "content": "Intro content",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["Intro content"],
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

        # When: The function is called with ignore_links=False
        result = get_md_header_contents(md_text, ignore_links=False)

        # Then: The result includes the link information
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="# Header 1\nContent with [link](http://example.com)",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content with [link](http://example.com)",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [
                        {
                            "text": "link",
                            "url": "http://example.com",
                            "caption": None,
                            "start_idx": -1,
                            "end_idx": -1,
                            "line": "Content with [link](http://example.com)",
                            "line_idx": 2,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 1", "Content with [link](http://example.com)"],
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

        # When: get_md_header_contents is called with ignore_links=False
        result = get_md_header_contents(
            md_text, ignore_links=False, base_url=base_url)

        # Then: HeaderDocuments with links in metadata
        assert len(result) == 2
        expected = [
            HeaderDocument(
                text="# Header 1\n[Link](http://example.com)",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "[Link](http://example.com)",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": base_url,
                    "links": [
                        {
                            "text": "Link",
                            "url": "http://example.com",
                            "caption": None,
                            "start_idx": -1,
                            "end_idx": -1,
                            "line": "[Link](http://example.com)",
                            "line_idx": 2,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 1", "[Link](http://example.com)"],
                    "id": result[0].id
                },
                relationships={
                    NodeRelationship.CHILD: [
                        RelatedNodeInfo(node_id=result[1].id)]
                }
            ),
            HeaderDocument(
                text="# Header 2\n[Relative Link](/page)",
                id=result[1].id,
                metadata={
                    "header": "# Header 2",
                    "parent_header": "# Header 1",
                    "header_level": 2,
                    "content": "[Relative Link](/page)",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": base_url,
                    "links": [
                        {
                            "text": "Relative Link",
                            "url": "http://base.com/page",
                            "caption": None,
                            "start_idx": -1,
                            "end_idx": -1,
                            "line": "[Relative Link](/page)",
                            "line_idx": 4,
                            "is_heading": False
                        }
                    ],
                    "texts": ["# Header 2", "[Relative Link](/page)"],
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
                text="# Header\nContent",
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
                text="### Header\nContent 2",
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

        # When: The function is called with ignore_links=False
        result = get_md_header_contents(md_text, ignore_links=False)

        # Then: The result includes the link information
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="# Header [link](http://example.com)\nContent",
                id=result[0].id,
                metadata={
                    "header": "# Header [link](http://example.com)",
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
                            "start_idx": -1,
                            "end_idx": -1,
                            "line": "# Header [link](http://example.com)",
                            "line_idx": 1,
                            "is_heading": True
                        }
                    ],
                    "texts": ["# Header [link](http://example.com)", "Content"],
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
        md_text = "# Header 1\nContent 1\n\n\nContent 2"

        # When: The function is called
        result = get_md_header_contents(md_text)

        # Then: The result matches the expected HeaderDocument structure
        assert len(result) == 1
        expected = [
            HeaderDocument(
                text="# Header 1\nContent 1\n\nContent 2",
                id=result[0].id,
                metadata={
                    "header": "# Header 1",
                    "parent_header": "",
                    "header_level": 1,
                    "content": "Content 1\n\nContent 2",
                    "doc_index": 0,
                    "chunk_index": None,
                    "tokens": None,
                    "source_url": None,
                    "links": [],
                    "texts": ["# Header 1", "Content 1", "", "", "Content 2"],
                    "id": result[0].id
                },
                relationships={}
            )
        ]
        assert result[0].text == expected[0].text
        assert result[0].id == expected[0].id
        assert result[0].metadata == expected[0].metadata
        assert result[0].relationships == expected[0].relationships
