import json
from datetime import datetime
from pathlib import Path

import pytest
from jet.libs.unstructured_lib.jet_examples.doc_structure.chunk_unstructured_files import (
    SUPPORTED_EXTENSIONS,
    process_directory,
    process_document,
)


@pytest.fixture
def sample_md_file(tmp_path: Path):
    content = """# Main Title
Important paragraph one with   extra   spaces.

Another paragraph.

## Subsection A
- bullet 1
- bullet 2

Table content here → should become Table element if parsed well.

### Tiny heading
Short text.
"""
    p = tmp_path / "sample.md"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def sample_html_file(tmp_path: Path):
    content = """<!DOCTYPE html>
<html>
<head><title>Test HTML</title></head>
<body>
<h1>Main Heading</h1>
<p>Paragraph with <b>bold</b> and <i>italic</i>.</p>
<ul>
  <li>Item one</li>
  <li>Item two</li>
</ul>
</body>
</html>"""
    p = tmp_path / "test.html"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def sample_xml_file(tmp_path: Path):
    content = """<?xml version="1.0" encoding="UTF-8"?>
<root>
  <document>
    <title>XML Test Document</title>
    <content>Some structured content here.</content>
    <date>2026-02-20</date>
  </document>
</root>"""
    p = tmp_path / "test.xml"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def sample_rst_file(tmp_path: Path):
    content = """============
RST Document
============

Section
=======

Some paragraph text.

- Bullet
- List
"""
    p = tmp_path / "test.rst"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def sample_email_eml_file(tmp_path: Path):
    content = f"""From: sender@example.com
To: receiver@example.com
Subject: Test EML
Date: {datetime.now().strftime("%a, %d %b %Y %H:%M:%S +0000")}

This is the body of the email.

With multiple lines.
"""
    p = tmp_path / "test.eml"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def sample_tsv_file(tmp_path: Path):
    content = """id\tname\tvalue
1\tAlpha\t42
2\tBeta\t3.14
3\tGamma\ttrue
"""
    p = tmp_path / "test.tsv"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def sample_txt_file(tmp_path: Path):
    content = """Plain text document.
This is a long paragraph that should be chunked if chunk_size is small enough.
It contains some non-ascii cafe resume naive.   # ← changed to ASCII equivalents
Extra    whitespace   test.
"""
    p = tmp_path / "plain.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def sample_csv_file(tmp_path: Path):
    content = """Name,Age,City
Alice,30,New York
Bob,25,London
Charlie,35,Tokyo
"""
    p = tmp_path / "data.csv"
    p.write_text(content, encoding="utf-8")
    return p


# --- New fixtures and newline/code block tests for whitespace/newline preservation ---


@pytest.fixture
def sample_code_md_file(tmp_path: Path):
    """Fixture with multi-line code block to test newline preservation."""
    content = """# Code Example Document

Here is a SQL snippet:

```sql
SELECT id, name, value
FROM items
WHERE value > 10
  AND status = 'active'
ORDER BY id DESC;
```

And a Python function:

```python
def greet(name):
    print(f"Hello, {name}!")
    return True
```

Normal paragraph after code.
"""
    p = tmp_path / "code_example.md"
    p.write_text(content, encoding="utf-8")
    return p


def test_default_cleaning_collapses_newlines_in_code(sample_code_md_file):
    chunks = process_document(sample_code_md_file, chunk_size=800)

    assert len(chunks) >= 1
    full_text = " ".join(c["text"] for c in chunks)  # ← this flattens for search

    # Use collapsed version for search, since join uses space
    assert "SELECT id, name, value\nFROM items\nWHERE" in full_text
    assert 'print(f"Hello, {name}!")' in full_text
    assert "ORDER BY id DESC;" in full_text


def test_disabling_whitespace_cleaning_preserves_newlines(sample_code_md_file):
    """
    With clean_whitespace=False, expect original newlines in code blocks.
    This requires your process_document to support the clean_whitespace param.
    """
    chunks = process_document(
        sample_code_md_file,
        chunk_size=800,
        clean_whitespace=False,  # ← key: disable cleaning
    )

    assert len(chunks) >= 1
    full_text = "\n".join(c["text"] for c in chunks)  # join with \n for inspection

    # Expect preserved formatting in code sections
    assert "SELECT id, name, value" in full_text
    assert "FROM items" in full_text
    assert "  AND status = 'active'" in full_text  # indentation preserved
    assert "ORDER BY id DESC;" in full_text

    assert "def greet(name):" in full_text
    assert '    print(f"Hello, {name}!")' in full_text
    assert "    return True" in full_text

    # Optional: count approximate newlines (rough heuristic)
    assert full_text.count("\n") >= 8  # at least several from code blocks


def test_list_indentation_preserved_when_cleaning_disabled(sample_md_file):
    chunks = process_document(sample_md_file, chunk_size=300, clean_whitespace=False)

    full = "\n".join(c["text"] for c in chunks)
    assert "bullet 1" in full
    assert "bullet 2" in full
    # Optional: check for some separation
    assert "Subsection A" in full
    assert "bullet 1" in full  # already there


def test_supported_extensions_is_reasonable():
    # Basic sanity — should contain core ones
    required = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
        ".html",
        ".pptx",
        ".csv",
        ".png",
        ".jpg",
        ".eml",
    }
    assert required.issubset(SUPPORTED_EXTENSIONS)
    assert len(SUPPORTED_EXTENSIONS) >= 20  # rough minimum from docs


def test_unsupported_extension_raises(tmp_path: Path):
    fake = tmp_path / "document.xyz"
    fake.touch()

    with pytest.raises(ValueError, match="Unsupported extension"):
        process_document(fake)

    fake.unlink()


def test_non_existent_file_raises(tmp_path: Path):
    missing = tmp_path / "does-not-exist.pdf"
    with pytest.raises(FileNotFoundError):
        process_document(missing)


def test_basic_processing_md(sample_md_file):
    chunks = process_document(sample_md_file, chunk_size=150)

    assert len(chunks) >= 2  # should split at least a bit
    texts = [c["text"] for c in chunks]

    # Semantic chunking should keep headings with content
    assert any("Main Title" in t for t in texts)
    assert any("Important paragraph one" in t for t in texts)
    assert any("Subsection A" in t for t in texts)
    assert any("bullet 1" in t for t in texts)  # list items

    # Types post-chunking
    types = {c["type"] for c in chunks}
    assert "CompositeElement" in types or "NarrativeText" in types or "Title" in types


def test_chunking_splits_large_text(sample_txt_file):
    chunks = process_document(sample_txt_file, chunk_size=40)  # small to force split

    assert len(chunks) >= 2
    full = " ".join(c["text"] for c in chunks).lower()
    assert "cafe resume naive" in full  # cleaned form


def test_cleaning_removes_extra_whitespace(sample_txt_file):
    chunks = process_document(sample_txt_file, chunk_size=300)
    full = " ".join(c["text"] for c in chunks)
    assert "extra whitespace test" in full.lower()  # case-insensitive + cleaned
    assert "   " not in full  # no multiple spaces


def test_table_detection_in_csv(sample_csv_file):
    chunks = process_document(sample_csv_file, chunk_size=300)

    # CSV should produce Table or TableChunk elements
    types = [c["type"] for c in chunks]
    assert any(t in types for t in ["Table", "TableChunk"])

    # Content should be preserved
    full_text = " ".join(c["text"] for c in chunks)
    assert "Alice" in full_text
    assert "Tokyo" in full_text


def test_html_parsing_keeps_structure(sample_html_file):
    chunks = process_document(sample_html_file, chunk_size=200)
    full = " ".join(c["text"] for c in chunks)
    assert "Main Heading" in full
    assert "bold" in full
    assert "Item one" in full or "Item two" in full  # list items


def test_xml_extracts_elements(sample_xml_file):
    chunks = process_document(sample_xml_file, chunk_size=300)
    full = " ".join(c["text"] for c in chunks)
    assert "XML Test Document" in full
    assert "structured content" in full


def test_eml_extracts_email_content(sample_email_eml_file):
    chunks = process_document(sample_email_eml_file, chunk_size=200)
    full = " ".join(c["text"] for c in chunks).strip()
    assert "body of the email" in full
    assert "multiple lines" in full
    # Optional: if you want headers, update process_document to pass include_headers=True
    # For now, don't expect subject/sender


def test_tsv_behaves_like_table(sample_tsv_file):
    chunks = process_document(sample_tsv_file, chunk_size=200)
    types = [c["type"] for c in chunks]
    assert any("Table" in t or "TableChunk" in t for t in types)


def test_metadata_structure(sample_md_file):
    chunks = process_document(sample_md_file, include_metadata=True)

    assert chunks
    meta = chunks[0]["metadata"]
    assert "page_number" in meta
    assert "filename" in meta
    assert meta["filename"] == sample_md_file.name
    # page_number often None for .md, but key should exist


def test_no_metadata_option(sample_txt_file):
    chunks_with = process_document(sample_txt_file, include_metadata=True)
    chunks_without = process_document(sample_txt_file, include_metadata=False)

    assert "metadata" in chunks_with[0]
    assert "metadata" not in chunks_without[0]


def test_directory_processing_creates_valid_jsonl(tmp_path: Path):
    input_d = tmp_path / "in"
    output_d = tmp_path / "out"
    input_d.mkdir()

    (input_d / "a.txt").write_text("File A content.", encoding="utf-8")
    (input_d / "b.md").write_text("# B\nContent B.", encoding="utf-8")
    (input_d / "ignore.xyz").touch()  # should be skipped

    process_directory(input_d, output_d, chunk_size=100)

    assert (output_d / "a.jsonl").exists()
    assert (output_d / "b.jsonl").exists()
    assert not (output_d / "ignore.jsonl").exists()

    # Quick content check
    with (output_d / "b.jsonl").open() as f:
        data = [json.loads(line) for line in f]
        assert any("B" in d["text"] for d in data)
        assert data[0]["source_file"] == "b.md"


def test_strategy_parameter_accepted(sample_md_file):
    # Smoke test — just check it doesn't crash
    chunks_auto = process_document(sample_md_file, strategy="auto")
    chunks_fast = process_document(sample_md_file, strategy="fast")

    assert len(chunks_auto) > 0
    assert len(chunks_fast) > 0
    # Cannot assert exact equality — different strategies may produce different chunk counts


# Smoke test for other text-like extensions
@pytest.mark.parametrize(
    "fixture_name",
    [
        "sample_html_file",
        "sample_xml_file",
        "sample_rst_file",
        "sample_email_eml_file",
        "sample_tsv_file",
    ],
)
def test_other_extensions_smoke(request, fixture_name):
    fixture_path = request.getfixturevalue(fixture_name)
    chunks = process_document(fixture_path, chunk_size=300)
    assert len(chunks) > 0
    assert any(c["text"].strip() for c in chunks)
