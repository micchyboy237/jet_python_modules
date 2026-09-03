# tests/test_markdown_hierarchy_chunking.py

import re

import pytest
from jet.models.embeddings.chunking import (
    chunk_docs_by_hierarchy,
    chunk_headers_by_hierarchy,
)


def simple_tokenizer(x):
    """Deterministic whitespace tokenizer for tests."""
    if isinstance(x, str):
        return x.split()
    return [s.split() for s in x]


def simple_sentence_splitter(text: str):
    """Simple sentence splitter keeping punctuation attached."""
    matches = re.finditer(r"[^.!?]+[.!?]|[^.!?]+$", text)
    return [m.group(0).strip() for m in matches if m.group(0).strip()]


def full_chunk_slice(source: str, chunk: dict) -> str:
    """Header-inclusive slice for full semantic-unit reconstruction."""
    return source[chunk["metadata"]["start_idx"] : chunk["metadata"]["end_idx"]].strip()


def body_slice(source: str, chunk: dict) -> str:
    """Non-overlapping body-only slice for highlighting/dedup."""
    return source[
        chunk["metadata"]["body_start_idx"] : chunk["metadata"]["body_end_idx"]
    ].strip()


def chunk_one(text: str, chunk_size: int = 50, overlap_tokens: int = 0):
    return chunk_headers_by_hierarchy(
        text,
        chunk_size=chunk_size,
        tokenizer=simple_tokenizer,
        split_fn=simple_sentence_splitter,
        overlap_tokens=overlap_tokens,
    )


# ─── Core Behavior ──────────────────────────────────────────────


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
    assert full_chunk_slice(text, chunk) == "# Introduction\nWelcome to the guide."
    assert body_slice(text, chunk) == "Welcome to the guide."


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

    assert h1["parent_header"] is None
    assert h2["parent_header"] == "# H1"
    assert h2["parent_id"] == h1["header_doc_id"]
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
    assert chunk["parent_header"] == "## Empty Child"
    assert chunk["parent_level"] == 2
    assert chunk["parent_id"] is not None


def test_header_tokens_count_toward_chunk_budget():
    text = """# Big Header
one two.
three four.
"""
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
    assert preamble["section_index"] == -1

    assert header_chunk["header"] == "# H"
    assert header_chunk["section_index"] == 0


# ─── Index Correctness ──────────────────────────────────────────


def test_repeated_sentences_have_correct_distinct_indices():
    text = """# H
Same.
Same.
"""
    chunks = chunk_one(text, chunk_size=3)
    first, second = chunks

    # Full slices include header (overlapping is expected)
    assert full_chunk_slice(text, first) == "# H\nSame."
    second_full = full_chunk_slice(text, second)
    assert second_full.startswith("# H")
    assert second["content"] == "Same."

    # Body slices are NON-overlapping
    assert body_slice(text, first) == "Same."
    assert body_slice(text, second) == "Same."
    assert first["metadata"]["body_end_idx"] <= second["metadata"]["body_start_idx"]

    # Chunks are distinct
    assert first["chunk_index"] == 0
    assert second["chunk_index"] == 1
    assert first["id"] != second["id"]


def test_all_chunks_under_same_header_include_header_in_range():
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
        assert sliced.index(chunk["header"]) < sliced.index(chunk["content"])


def test_body_indices_are_non_overlapping_across_chunks():
    text = """# H
Alpha bravo charlie.
Delta echo foxtrot.
Golf hotel india.
"""
    chunks = chunk_one(text, chunk_size=5)
    assert len(chunks) >= 2

    for i in range(len(chunks) - 1):
        assert (
            chunks[i]["metadata"]["body_end_idx"]
            <= chunks[i + 1]["metadata"]["body_start_idx"]
        )


def test_indices_correct_with_leading_blank_lines():
    text = """

# H
Body.
"""
    chunks = chunk_one(text, chunk_size=50)
    assert len(chunks) == 1

    chunk = chunks[0]
    expected_body_start = text.index("Body.")
    expected_body_end = expected_body_start + len("Body.")

    assert chunk["metadata"]["body_start_idx"] == expected_body_start
    assert chunk["metadata"]["body_end_idx"] == expected_body_end
    assert body_slice(text, chunk) == "Body."


# ─── Sentence Alignment Safety ──────────────────────────────────


def test_misaligned_sentence_raises_value_error():
    """split_fn returning text not in source must raise, not silently corrupt."""

    def bad_splitter(text: str):
        return ["THIS DOES NOT EXIST IN SOURCE."]

    text = "# H\nReal content here."

    with pytest.raises(ValueError, match="Sentence alignment failed"):
        chunk_headers_by_hierarchy(
            text,
            chunk_size=50,
            tokenizer=simple_tokenizer,
            split_fn=bad_splitter,
        )


# ─── Overlap Support ────────────────────────────────────────────


def test_overlap_prepends_words_from_previous_chunk():
    """Word-level overlap should prepend tail words of previous chunk."""
    text = """# H
one two three.
four five six.
seven eight nine.
"""
    # Header "# H" = 2 tokens. Each sentence = 3 tokens.
    # chunk_size=6: header(2) + sentence(3) = 5 ≤ 6 fits.
    # Two sentences: 5 + 3 = 8 > 6 → flush after first sentence.
    # With overlap_tokens=2, second chunk gets last 2 words of first prepended.
    chunks = chunk_one(text, chunk_size=6, overlap_tokens=2)

    assert len(chunks) >= 2
    assert chunks[0]["content"] == "one two three."
    # Second chunk should contain overlap words "two three" plus its own sentence
    second_content = chunks[1]["content"]
    assert "two" in second_content
    assert "three" in second_content
    assert "four" in second_content


def test_overlap_does_not_count_toward_budget():
    text = """# H
alpha beta.
gamma delta.
"""
    no_overlap = chunk_one(text, chunk_size=4, overlap_tokens=0)
    with_overlap = chunk_one(text, chunk_size=4, overlap_tokens=2)

    # Same number of chunks; overlap is free
    assert len(with_overlap) == len(no_overlap)


def test_zero_overlap_is_default():
    text = """# H
one two.
three four.
"""
    default = chunk_one(text, chunk_size=4)
    explicit_zero = chunk_one(text, chunk_size=4, overlap_tokens=0)

    assert len(default) == len(explicit_zero)
    for d, e in zip(default, explicit_zero):
        assert d["content"] == e["content"]


def test_overlap_larger_than_chunk_content():
    """When overlap_tokens exceeds available words, take all available."""
    text = """# H
one two.
three four.
"""
    chunks = chunk_one(text, chunk_size=4, overlap_tokens=100)

    assert len(chunks) >= 2
    # All words from first chunk should appear as overlap in second
    second_content = chunks[1]["content"]
    assert "one" in second_content
    assert "two" in second_content


# ─── Edge Cases ─────────────────────────────────────────────────


def test_trailing_hash_header_is_parsed_as_header():
    """Regex (.+) captures trailing ## as part of header text. Valid markdown."""
    text = "## Title ##\nBody content."
    chunks = chunk_one(text, chunk_size=50)

    assert len(chunks) == 1
    assert chunks[0]["header"] == "## Title ##"
    assert chunks[0]["level"] == 2
    assert chunks[0]["content"] == "Body content."


def test_six_level_deep_nesting():
    text = """# L1
One.

## L2
Two.

### L3
Three.

#### L4
Four.

##### L5
Five.

###### L6
Six.
"""
    chunks = chunk_one(text, chunk_size=50)
    assert len(chunks) == 6

    levels = [c["level"] for c in chunks]
    assert levels == [1, 2, 3, 4, 5, 6]

    for i in range(1, len(chunks)):
        assert chunks[i]["parent_level"] == chunks[i - 1]["level"]
        assert chunks[i]["parent_id"] == chunks[i - 1]["header_doc_id"]


def test_header_immediately_followed_by_same_level_header():
    text = """# First
# Second
Body under second.
"""
    chunks = chunk_one(text, chunk_size=50)
    assert len(chunks) == 1

    assert chunks[0]["header"] == "# Second"
    assert chunks[0]["content"] == "Body under second."
    assert chunks[0]["parent_header"] is None


def test_unicode_content():
    text = """# 日本語テスト
これはテストです。

## Émojis 🎉
Café résumé naïve.
"""
    chunks = chunk_one(text, chunk_size=50)
    assert len(chunks) == 2

    assert chunks[0]["header"] == "# 日本語テスト"
    assert "これはテストです" in chunks[0]["content"]

    assert chunks[1]["header"] == "## Émojis 🎉"
    assert "Café résumé naïve" in chunks[1]["content"]


def test_crlf_line_endings():
    text = "# Header\r\nBody line one.\r\nBody line two.\r\n"
    chunks = chunk_one(text, chunk_size=50)

    assert len(chunks) == 1
    assert chunks[0]["header"] == "# Header"
    assert "Body line one" in chunks[0]["content"]


def test_extremely_long_single_sentence_emitted_as_oversized_chunk():
    """A single sentence exceeding chunk_size is emitted as one oversized chunk.
    This is documented behavior: we cannot split mid-sentence."""
    words = " ".join([f"word{i}" for i in range(500)])
    text = f"# Long\n{words}"
    chunks = chunk_one(text, chunk_size=100)

    # No periods → single sentence → cannot split → one oversized chunk
    assert len(chunks) == 1
    assert chunks[0]["num_tokens"] == 502  # 500 words + "# Long" (2 tokens)
    assert "word0" in chunks[0]["content"]
    assert "word499" in chunks[0]["content"]


def test_long_multi_sentence_paragraph_splits_correctly():
    """Multiple sentences on one line DO split when exceeding budget."""
    sentences = ". ".join([f"Sentence number {i} here" for i in range(50)]) + "."
    text = f"# Long\n{sentences}"
    chunks = chunk_one(text, chunk_size=20)

    assert len(chunks) >= 2
    all_content = " ".join(c["content"] for c in chunks)
    assert "Sentence number 0" in all_content
    assert "Sentence number 49" in all_content


# ─── Multi-Document ─────────────────────────────────────────────


def test_chunk_docs_preserves_explicit_document_ids():
    docs = ["# Doc A\nContent A.", "# Doc B\nContent B."]
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
    assert chunks[1]["doc_id"] == "b2"
    assert chunks[1]["doc_index"] == 1


def test_chunk_docs_generates_document_ids_when_missing():
    docs = ["# Doc A\nContent A.", "# Doc B\nContent B."]
    chunks = chunk_docs_by_hierarchy(
        docs,
        chunk_size=50,
        tokenizer=simple_tokenizer,
        split_fn=simple_sentence_splitter,
    )

    assert chunks[0]["doc_id"] != chunks[1]["doc_id"]


def test_chunk_docs_rejects_mismatched_ids():
    docs = ["# Doc A\nContent A.", "# Doc B\nContent B."]
    with pytest.raises(ValueError, match="Number of provided IDs"):
        chunk_docs_by_hierarchy(
            docs,
            chunk_size=50,
            tokenizer=simple_tokenizer,
            split_fn=simple_sentence_splitter,
            ids=["only-one-id"],
        )


def test_section_index_starts_at_negative_one_for_preamble():
    """Preamble content gets section_index=-1, first header gets 0."""
    text = "Preamble text.\n# First Header\nBody."
    chunks = chunk_one(text, chunk_size=50)

    assert chunks[0]["section_index"] == -1
    assert chunks[1]["section_index"] == 0
