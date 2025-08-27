# tests/test_rst_parser.py
import pytest
from jet.code.doc_parsers.rst_parser import parse_rst_to_blocks


def top_types(doc):
    """Helper: return list of top-level child types for easier assertions."""
    return [c["type"] for c in doc["children"]]


def find_first(doc, t):
    for c in doc["children"]:
        if c["type"] == t or c["type"].endswith(t):
            return c
    return None


def test_section_and_paragraph():
    rst = """
Section Title
-------------

A paragraph under the section.
"""
    doc = parse_rst_to_blocks(rst)
    types = top_types(doc)
    assert "section" in types or "topic" in types
    # find section then title then paragraph child
    section = find_first(doc, "section")
    assert section is not None
    title = next(
        (ch for ch in section["children"] if ch["type"] == "title"), None)
    assert title and "Section Title" in (title["text"] or "")
    para = next((ch for ch in section["children"]
                if ch["type"] == "paragraph"), None)
    assert para and "A paragraph under the section." in (para["text"] or "")


def test_bullet_and_enumerated_lists():
    rst = """
- first bullet
- second bullet

1. first enumerated
2. second enumerated
"""
    doc = parse_rst_to_blocks(rst)
    # expect a bullet_list then enumerated_list (order preserved)
    types = top_types(doc)
    assert "bullet_list" in types
    assert "enumerated_list" in types or "list" in types
    # verify list item contents exist
    bullets = find_first(doc, "bullet_list")
    assert bullets is not None
    items = [ch for ch in bullets["children"] if ch["type"] == "list_item"]
    assert any("first bullet" in (it["text"] or "") or "first bullet" in "".join(
        [c.get("text", "") or "" for c in it["children"]]) for it in items)


def test_definition_list_and_field_list():
    rst = """
Term
  Definition for term.

:Author: John Doe
:Version: 1.0
"""
    doc = parse_rst_to_blocks(rst)
    # definition list should appear
    assert any(c["type"] == "definition_list" for c in doc["children"])
    # field_list appears as docinfo block
    assert any(c["type"] == "field_list" for c in doc["children"])


def test_literal_block_and_directive_admonition():
    rst = """
Paragraph.

::

    print("literal block")

.. note::

   This is a note admonition.
"""
    doc = parse_rst_to_blocks(rst)
    # literal block present
    assert any(c["type"] == "literal_block" or any(
        ch["type"] == "literal_block" for ch in c["children"]) for c in doc["children"])
    # admonition (note) is a directive; docutils produces admonition/note node
    assert any("note" in c["type"] or "admonition" in c["type"]
               for c in doc["children"] + [ch for c in doc["children"] for ch in c["children"]])


def test_block_quote_and_line_block_and_transition():
    rst = """
> A block quote example.

| line 1
| line 2

----------

A paragraph after transition.
"""
    doc = parse_rst_to_blocks(rst)
    # block_quote
    assert any(
        "block_quote" in c["type"] or "blockquote" in c["type"] for c in doc["children"])
    # line_block
    assert any(c["type"] == "line_block" or any(
        ch["type"] == "line_block" for ch in c["children"]) for c in doc["children"])
    # transition -> docutils uses 'transition' node
    assert any(c["type"] == "transition" for c in doc["children"])


def test_footnote_citation_and_target():
    rst = """
.. _target-label:

A paragraph with a reference_.

.. [#] Footnote text.

.. _reference: http://example.com

.. [1] Citation text.
"""
    doc = parse_rst_to_blocks(rst)
    # target
    assert any(c["type"] == "target" or "target" in c["type"]
               for c in doc["children"])
    # footnote or citation
    assert any(c["type"] == "footnote" or c["type"]
               == "citation" for c in doc["children"])


def test_table_grid_and_simple_table():
    rst = """
+--------+--------+
| Head A | Head B |
+========+========+
| a1     | b1     |
+--------+--------+

Simple table:

====  ====
A     B
====  ====
a1    b1
====  ====
"""
    doc = parse_rst_to_blocks(rst)
    # table nodes should appear
    assert any(c["type"] == "table" or any(
        ch["type"] == "table" for ch in c["children"]) for c in doc["children"])


def test_comment_ignored_by_doctree():
    rst = """
Paragraph.

.. this is a comment and should be ignored by doctree

Next paragraph.
"""
    doc = parse_rst_to_blocks(rst)
    # comments are not included as visible nodes in doctree; ensure paragraphs present and no explicit 'comment' top-level
    assert any(c["type"] == "paragraph" for c in doc["children"])
    assert not any(c["type"] == "comment" for c in doc["children"])


def test_code_block_directive_literal_block_and_parsed_literal():
    rst = """
.. code-block:: python

   def f():
       return 1

.. parsed-literal::

   print("parsed literal")
"""
    doc = parse_rst_to_blocks(rst)
    # code-block ends up as literal_block (or as directive node whose child is literal_block)
    assert any("literal_block" in (c["type"] or "") or any(
        "literal_block" in ch["type"] for ch in c["children"]) for c in doc["children"])
    # parsed-literal becomes parsed-literal or similar node
    assert any("parsed-literal" in c["type"] or any("parsed-literal" in ch["type"]
               for ch in c["children"]) for c in doc["children"])


# If you want to run tests locally: pytest -q
