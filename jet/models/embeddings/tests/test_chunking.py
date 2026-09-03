# tests/test_markdown_hierarchy_chunking.py

import re

import pytest
from jet.models.embeddings.chunking import (
    chunk_docs_by_hierarchy,
    chunk_headers_by_hierarchy,
)


def simple_tokenizer(x):
    """
    Deterministic tokenizer for tests.

    Avoids relying on nltk punkt/tokenizer data.
    """
    if isinstance(x, str):
        return x.split()
    return [s.split() for s in x]


def simple_sentence_splitter(text: str):
    """
    Simple sentence splitter for tests.

    Keeps punctuation attached to the sentence.
    """
    matches = re.finditer(r"[^.!?]+[.!?]|[^.!?]+$", text)
    return [m.group(0).strip() for m in matches if m.group(0).strip()]


def body_slice(source: str, chunk: dict) -> str:
    return source[chunk["metadata"]["start_idx"] : chunk["metadata"]["end_idx"]].strip()


def full_chunk_slice(source: str, chunk: dict) -> str:
    """Slice that includes both header and body content."""
    return source[chunk["metadata"]["start_idx"] : chunk["metadata"]["end_idx"]].strip()


def chunk_one(text: str, chunk_size: int = 50):
    return chunk_headers_by_hierarchy(
        text,
        chunk_size=chunk_size,
        tokenizer=simple_tokenizer,
        split_fn=simple_sentence_splitter,
    )


def test_empty_input_returns_empty_list():
    assert chunk_one("") == []
    assert chunk_one("   \n\n   ") == []


def test_basic_header_chunking():
    text = "# Introduction\nWelcome to the guide."
    chunks = chunk_one(text, chunk_size=20)

    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk["header"] == "# Introduction"
    assert chunk["content"] == "Welcome to the guide."

    # Metadata now covers header + body
    assert full_chunk_slice(text, chunk) == "# Introduction\nWelcome to the guide."


def test_nested_header_parent_relationship():
    text = """# Introduction
Welcome.

## Setup
Install dependencies first.
Then configure your environment.
"""
    chunks = chunk_one(text, chunk_size=50)
    intro, setup = chunks

    assert full_chunk_slice(text, intro) == "# Introduction\nWelcome."
    assert (
        full_chunk_slice(text, setup)
        == "## Setup\nInstall dependencies first.\nThen configure your environment."
    )


def test_multiple_header_levels_resolve_nearest_parent():
    text = """# H1
Top.

## H2
Middle.

### H3
Deep.
"""

    chunks = chunk_one(text, chunk_size=50)

    assert len(chunks) == 3

    h1, h2, h3 = chunks

    assert h1["header"] == "# H1"
    assert h1["parent_header"] is None

    assert h2["header"] == "## H2"
    assert h2["parent_header"] == "# H1"
    assert h2["parent_id"] == h1["header_doc_id"]

    assert h3["header"] == "### H3"
    assert h3["parent_header"] == "## H2"
    assert h3["parent_id"] == h2["header_doc_id"]


def test_sibling_headers_share_same_parent():
    text = """# Root
Root body.

## A
A body.

## B
B body.
"""

    chunks = chunk_one(text, chunk_size=50)

    assert len(chunks) == 3

    root, a, b = chunks

    assert a["parent_header"] == "# Root"
    assert b["parent_header"] == "# Root"
    assert a["parent_id"] == root["header_doc_id"]
    assert b["parent_id"] == root["header_doc_id"]


def test_header_without_body_is_skipped():
    text = """# Empty Header

## Child
Child body.
"""

    chunks = chunk_one(text, chunk_size=50)

    assert len(chunks) == 1

    chunk = chunks[0]
    assert chunk["header"] == "## Child"
    assert chunk["content"] == "Child body."

    # Even though "# Empty Header" produces no chunk,
    # it should still be tracked as the parent.
    assert chunk["parent_header"] == "# Empty Header"
    assert chunk["parent_level"] == 1
    assert chunk["parent_id"] is not None


def test_multiple_empty_headers_are_skipped_but_hierarchy_is_preserved():
    text = """# Root

## Empty Child

### Real Section
Actual body.
"""

    chunks = chunk_one(text, chunk_size=50)

    assert len(chunks) == 1

    chunk = chunks[0]
    assert chunk["header"] == "### Real Section"
    assert chunk["content"] == "Actual body."

    assert chunk["parent_header"] == "## Empty Child"
    assert chunk["parent_level"] == 2
    assert chunk["parent_id"] is not None


def test_header_tokens_count_toward_chunk_budget():
    text = """# Big Header
one two.
three four.
"""

    # Header tokens:
    # "# Big Header" -> ["#", "Big", "Header"] = 3
    #
    # Sentence tokens:
    # "one two." -> 2
    # "three four." -> 2
    #
    # Header + first sentence = 5
    # Header + both sentences = 7
    #
    # With chunk_size=5, each sentence should become its own chunk.
    chunks = chunk_one(text, chunk_size=5)

    assert len(chunks) == 2

    assert chunks[0]["content"] == "one two."
    assert chunks[0]["num_tokens"] == 5

    assert chunks[1]["content"] == "three four."
    assert chunks[1]["num_tokens"] == 5


def test_oversized_sentence_is_emitted_as_single_chunk():
    text = """# H
one two three four five six.
"""

    # Header has 2 tokens: "#", "H"
    # Sentence has 6 tokens.
    # Total = 8 > chunk_size.
    chunks = chunk_one(text, chunk_size=3)

    assert len(chunks) == 1

    chunk = chunks[0]
    assert chunk["content"] == "one two three four five six."
    assert chunk["num_tokens"] == 8


def test_chunk_index_resets_per_header():
    text = """# A
one two.
three four.

# B
five six.
seven eight.
"""

    chunks = chunk_one(text, chunk_size=4)

    a_chunks = [c for c in chunks if c["header"] == "# A"]
    b_chunks = [c for c in chunks if c["header"] == "# B"]

    assert [c["chunk_index"] for c in a_chunks] == [0, 1]
    assert [c["chunk_index"] for c in b_chunks] == [0, 1]


def test_repeated_sentences_have_correct_distinct_indices():
    text = """# H
Same.
Same.
"""
    chunks = chunk_one(text, chunk_size=3)
    first, second = chunks

    # Both chunks include the header (overlapping ranges are expected)
    assert full_chunk_slice(text, first) == "# H\nSame."

    # Second chunk also includes header + its own body sentence
    # The slice [header_start : second_body_end] naturally includes
    # the first sentence too since it falls within the range.
    # What matters is that content is correct and chunks are distinct.
    second_slice = full_chunk_slice(text, second)
    assert second_slice.startswith("# H")
    assert second["content"] == "Same."

    # Chunks must be distinct
    assert first["chunk_index"] == 0
    assert second["chunk_index"] == 1
    assert first["id"] != second["id"]

    # Body end positions must differ (second chunk's body ends later)
    assert first["metadata"]["end_idx"] < second["metadata"]["end_idx"]


def test_content_before_first_header_is_chunked():
    text = """Intro before any header.

# H
Body under header.
"""

    chunks = chunk_one(text, chunk_size=50)

    assert len(chunks) == 2

    preamble = chunks[0]
    header_chunk = chunks[1]

    assert preamble["header"] == ""
    assert preamble["parent_header"] is None
    assert preamble["content"] == "Intro before any header."

    assert header_chunk["header"] == "# H"
    assert header_chunk["content"] == "Body under header."


def test_chunk_docs_preserves_explicit_document_ids():
    docs = [
        "# Doc A\nContent A.",
        "# Doc B\nContent B.",
    ]

    chunks = chunk_docs_by_hierarchy(
        docs,
        chunk_size=50,
        tokenizer=simple_tokenizer,
        split_fn=simple_sentence_splitter,
        ids=["a1", "b2"],
    )

    assert len(chunks) == 2

    assert chunks[0]["doc_id"] == "a1"
    assert chunks[0]["doc_index"] == 0
    assert chunks[0]["header"] == "# Doc A"

    assert chunks[1]["doc_id"] == "b2"
    assert chunks[1]["doc_index"] == 1
    assert chunks[1]["header"] == "# Doc B"


def test_chunk_docs_generates_document_ids_when_missing():
    docs = [
        "# Doc A\nContent A.",
        "# Doc B\nContent B.",
    ]

    chunks = chunk_docs_by_hierarchy(
        docs,
        chunk_size=50,
        tokenizer=simple_tokenizer,
        split_fn=simple_sentence_splitter,
    )

    assert len(chunks) == 2

    assert chunks[0]["doc_id"]
    assert chunks[1]["doc_id"]
    assert chunks[0]["doc_id"] != chunks[1]["doc_id"]


def test_chunk_docs_rejects_mismatched_ids():
    docs = [
        "# Doc A\nContent A.",
        "# Doc B\nContent B.",
    ]

    with pytest.raises(ValueError, match="Number of provided IDs"):
        chunk_docs_by_hierarchy(
            docs,
            chunk_size=50,
            tokenizer=simple_tokenizer,
            split_fn=simple_sentence_splitter,
            ids=["only-one-id"],
        )


def test_all_chunks_under_same_header_include_header_in_range():
    """Every chunk from a single header must have the header in its metadata slice."""
    text = """# Shared Header
Sentence one is here.
Sentence two is here.
Sentence three is here.
Sentence four is here.
"""
    chunks = chunk_one(text, chunk_size=6)
    assert len(chunks) >= 2

    for chunk in chunks:
        sliced = full_chunk_slice(text, chunk)
        assert chunk["header"] in sliced
        assert chunk["content"] in sliced
        # Content must appear after header in the slice
        assert sliced.index(chunk["header"]) < sliced.index(chunk["content"])
