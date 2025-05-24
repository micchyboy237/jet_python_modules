import pytest
from langchain_core.documents import Document
from jet.code.helpers.markdown_header_text_splitter import MarkdownHeaderTextSplitter


@pytest.fixture
def splitter():
    headers_to_split_on = [
        ("# ", "h1"),
        ("## ", "h2"),
        ("### ", "h3"),
    ]
    return MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)


@pytest.fixture
def strip_splitter():
    headers_to_split_on = [
        ("# ", "h1"),
        ("## ", "h2"),
        ("### ", "h3"),
    ]
    return MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=True)


def test_headers_no_content(splitter):
    md_text = """
# Level 1
# Level 2
# Level 3
"""
    expected = []
    result = splitter.split_text(md_text)
    assert result == expected, f"Expected no documents when only headers are present, got {len(result)} documents: {result}"


def test_header_with_content(splitter):
    md_text = """
# Level 1
Content
# Level 2
# Level 3
"""
    expected = [
        Document(
            page_content="# Level 1\nContent",
            metadata={"h1": "Level 1"},
            id=None,
            type="Document"
        )
    ]
    result = splitter.split_text(md_text)
    assert len(result) == len(
        expected), f"Expected {len(expected)} document(s), got {len(result)}: {result}"
    for res, exp in zip(result, expected):
        assert res.page_content == exp.page_content, f"Page content mismatch: got '{res.page_content}', expected '{exp.page_content}'"
        assert res.metadata == exp.metadata, f"Metadata mismatch: got {res.metadata}, expected {exp.metadata}"
        assert res.id == exp.id, f"ID mismatch: got {res.id}, expected {exp.id}"
        assert res.type == exp.type, f"Type mismatch: got {res.type}, expected {exp.type}"


def test_multiple_headers_with_content(splitter):
    md_text = """
# Level 1
Content
# Level 2
Content
# Level 3
"""
    expected = [
        Document(
            page_content="# Level 1\nContent",
            metadata={"h1": "Level 1"},
            id=None,
            type="Document"
        ),
        Document(
            page_content="# Level 2\nContent",
            metadata={"h1": "Level 2"},
            id=None,
            type="Document"
        )
    ]
    result = splitter.split_text(md_text)
    assert len(result) == len(
        expected), f"Expected {len(expected)} document(s), got {len(result)}: {result}"
    for res, exp in zip(result, expected):
        assert res.page_content == exp.page_content, f"Page content mismatch: got '{res.page_content}', expected '{exp.page_content}'"
        assert res.metadata == exp.metadata, f"Metadata mismatch: got {res.metadata}, expected {exp.metadata}"
        assert res.id == exp.id, f"ID mismatch: got {res.id}, expected {exp.id}"
        assert res.type == exp.type, f"Type mismatch: got {res.type}, expected {exp.type}"


def test_headers_with_content_and_strip_headers(strip_splitter):
    md_text = """
# Level 1
Content
# Level 2
Content
# Level 3
"""
    expected = [
        Document(
            page_content="Content",
            metadata={"h1": "Level 1"},
            id=None,
            type="Document"
        ),
        Document(
            page_content="Content",
            metadata={"h1": "Level 2"},
            id=None,
            type="Document"
        )
    ]
    result = strip_splitter.split_text(md_text)
    assert len(result) == len(
        expected), f"Expected {len(expected)} document(s), got {len(result)}: {result}"
    for res, exp in zip(result, expected):
        assert res.page_content == exp.page_content, f"Page content mismatch: got '{res.page_content}', expected '{exp.page_content}'"
        assert res.metadata == exp.metadata, f"Metadata mismatch: got {res.metadata}, expected {exp.metadata}"
        assert res.id == exp.id, f"ID mismatch: got {res.id}, expected {exp.id}"
        assert res.type == exp.type, f"Type mismatch: got {res.type}, expected {exp.type}"


def test_empty_input(splitter):
    md_text = ""
    expected = []
    result = splitter.split_text(md_text)
    assert result == expected, f"Expected no documents for empty input, got {len(result)} documents: {result}"


def test_content_without_headers(splitter):
    md_text = "Just content\nMore content"
    expected = [
        Document(
            page_content="Just content\nMore content",
            metadata={},
            id=None,
            type="Document"
        )
    ]
    result = splitter.split_text(md_text)
    assert len(result) == len(
        expected), f"Expected {len(expected)} document(s), got {len(result)}: {result}"
    for res, exp in zip(result, expected):
        assert res.page_content == exp.page_content, f"Page content mismatch: got '{res.page_content}', expected '{exp.page_content}'"
        assert res.metadata == exp.metadata, f"Metadata mismatch: got {res.metadata}, expected {exp.metadata}"
        assert res.id == exp.id, f"ID mismatch: got {res.id}, expected {exp.id}"
        assert res.type == exp.type, f"Type mismatch: got {res.type}, expected {exp.type}"
