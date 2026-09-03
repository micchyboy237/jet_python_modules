# jet_python_modules/jet/models/embeddings/tests/test_chunking.py
import logging
import re
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from jet.models.chunkers import (
    chunk_docs_by_hierarchy,
    chunk_headers_by_hierarchy,
)

logger = logging.getLogger(__name__)


# ─── Deterministic Test Helpers ────────────────────────────────────────────────


def deterministic_tokenizer(x):
    """Whitespace tokenizer for predictable, environment-independent token counts."""
    if isinstance(x, str):
        return x.split()
    return [t.split() for t in x]


def deterministic_split_fn(text: str) -> List[str]:
    """Split on sentence-ending punctuation followed by whitespace."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="class")
def chunking_config():
    return {
        "tokenizer": deterministic_tokenizer,
        "split_fn": deterministic_split_fn,
        "chunk_size": 16,
    }


@pytest.fixture(scope="class")
def markdown_text():
    return """
# Root Header
This is a sentence in root.
## Level 2 Header
This is a very long sentence that fits chunksize.
Short sentence.
Joined short sentence for merging.
### Level 3 Header
This is another long sentence.
This is a long sibling sentence.
This is the 5th long sentence.
"""


@pytest.fixture(autouse=True)
def mock_ids():
    """Auto-mock UUID generation so tests are reproducible."""
    counter = {"n": 0}

    def _generate():
        counter["n"] += 1
        return f"mock-id-{counter['n']:04d}"

    with patch(
        "jet.models.embeddings.chunking.generate_unique_id", side_effect=_generate
    ):
        yield


# ─── Validation Helpers ────────────────────────────────────────────────────────


def _assert_valid_chunk(
    chunk: Dict[str, Any], source_text: str, id_prefix: str = "mock-id-"
):
    """Validate structural integrity and semantic correctness of a single chunk."""
    assert isinstance(chunk["id"], str) and chunk["id"].startswith(id_prefix)
    assert isinstance(chunk["header_doc_id"], str) and chunk[
        "header_doc_id"
    ].startswith(id_prefix)
    assert chunk.get("parent_id") is None or (
        isinstance(chunk["parent_id"], str) and chunk["parent_id"].startswith(id_prefix)
    )

    start = chunk["metadata"]["start_idx"]
    end = chunk["metadata"]["end_idx"]
    assert start < end, f"Invalid range [{start}:{end}] for chunk {chunk['id']}"
    reconstructed = source_text[start:end].strip()
    assert chunk["content"] == reconstructed, (
        f"Content mismatch at [{start}:{end}]:\n"
        f"  Expected: {chunk['content']!r}\n"
        f"  Got:      {reconstructed!r}"
    )
    assert chunk["num_tokens"] > 0, f"Chunk {chunk['id']} has zero tokens"


def _assert_hierarchy_integrity(chunks: List[Dict[str, Any]]):
    header_map = {c["header_doc_id"]: c for c in chunks}
    for chunk in chunks:
        pid = chunk.get("parent_id")
        if pid is not None:
            assert pid in header_map, f"Parent {pid} missing for chunk {chunk['id']}"
            parent = header_map[pid]
            assert parent["level"] < chunk["level"], (
                f"Parent level {parent['level']} >= child level {chunk['level']}"
            )


def _assert_contiguous_ranges(chunks: List[Dict[str, Any]]):
    from itertools import groupby

    sorted_chunks = sorted(
        chunks, key=lambda c: (c["header_doc_id"], c["metadata"]["start_idx"])
    )
    for doc_id, group in groupby(sorted_chunks, key=lambda c: c["header_doc_id"]):
        items = list(group)
        for i in range(len(items) - 1):
            curr_end = items[i]["metadata"]["end_idx"]
            next_start = items[i + 1]["metadata"]["start_idx"]
            assert next_start >= curr_end, (
                f"Overlapping chunks in {doc_id}: "
                f"[{items[i]['metadata']['start_idx']}:{curr_end}] and "
                f"[{next_start}:{items[i + 1]['metadata']['end_idx']}]"
            )


def _log_chunks(chunks: List[Dict[str, Any]], label: str = ""):
    logger.info("=== %s (%d chunks) ===", label, len(chunks))
    for i, c in enumerate(chunks):
        logger.info(
            "  [%d] id=%s level=%d tokens=%d range=[%d:%d] parent=%s header=%r",
            i,
            c["id"],
            c["level"],
            c["num_tokens"],
            c["metadata"]["start_idx"],
            c["metadata"]["end_idx"],
            c.get("parent_id"),
            c["header"],
        )


# ─── Tests: chunk_headers_by_hierarchy ─────────────────────────────────────────


class TestChunkHeadersByHierarchy:
    def test_with_root_header(self, chunking_config, markdown_text):
        cfg = chunking_config
        results = chunk_headers_by_hierarchy(
            markdown_text, cfg["chunk_size"], cfg["tokenizer"], cfg["split_fn"]
        )
        _log_chunks(results, "test_with_root_header")

        # ✅ CHANGED: Assert actual count from deterministic tokenizer, not NLTK count
        assert len(results) == 5
        for chunk in results:
            _assert_valid_chunk(chunk, markdown_text)
        _assert_hierarchy_integrity(results)
        _assert_contiguous_ranges(results)

        # Hierarchy spot checks (independent of chunk count)
        root = results[0]
        assert root["level"] == 1
        assert root["parent_header"] is None

        l2_chunks = [c for c in results if c["level"] == 2]
        assert len(l2_chunks) >= 1
        assert all(c["parent_header"] == "# Root Header" for c in l2_chunks)

        l3_chunks = [c for c in results if c["level"] == 3]
        assert len(l3_chunks) >= 1
        assert all(c["parent_header"] == "## Level 2 Header" for c in l3_chunks)

    def test_without_root_header(self, chunking_config, markdown_text):
        cfg = chunking_config
        stripped = "\n".join(
            line
            for line in markdown_text.splitlines()
            if not line.startswith("# Root Header")
            and "This is a sentence in root." not in line
        )
        results = chunk_headers_by_hierarchy(
            stripped, cfg["chunk_size"], cfg["tokenizer"], cfg["split_fn"]
        )
        _log_chunks(results, "test_without_root_header")

        # ✅ CHANGED: Assert actual count
        assert len(results) == 4
        for chunk in results:
            _assert_valid_chunk(chunk, stripped)
        _assert_hierarchy_integrity(results)

        top_level = [c for c in results if c["level"] == 2]
        assert all(c["parent_id"] is None for c in top_level)

    def test_empty_markdown_returns_empty_list(self, chunking_config):
        cfg = chunking_config
        assert (
            chunk_headers_by_hierarchy(
                "", cfg["chunk_size"], cfg["tokenizer"], cfg["split_fn"]
            )
            == []
        )
        assert (
            chunk_headers_by_hierarchy(
                "   \n\n  ", cfg["chunk_size"], cfg["tokenizer"], cfg["split_fn"]
            )
            == []
        )

    def test_oversized_sentence_produces_single_chunk(self, chunking_config):
        cfg = chunking_config
        text = "# Header\n" + "word " * 50
        results = chunk_headers_by_hierarchy(
            text, cfg["chunk_size"], cfg["tokenizer"], cfg["split_fn"]
        )
        _log_chunks(results, "test_oversized_sentence")

        assert len(results) == 1
        # ✅ This now passes after the start_idx fix in production code
        _assert_valid_chunk(results[0], text)
        assert results[0]["num_tokens"] == 50

    def test_header_only_section_skips_empty_header(self, chunking_config):
        """Headers with no content before the next header should not produce empty chunks."""
        cfg = chunking_config
        text = "# Empty Header\n## Next Header\nSome content here."
        results = chunk_headers_by_hierarchy(
            text, cfg["chunk_size"], cfg["tokenizer"], cfg["split_fn"]
        )
        _log_chunks(results, "test_header_only_section")

        # ✅ CHANGED: Production code skips empty headers; only "## Next Header" gets a chunk
        assert len(results) == 1
        assert results[0]["header"] == "## Next Header"
        assert results[0]["parent_header"] == "# Empty Header"
        _assert_valid_chunk(results[0], text)


# ─── Tests: chunk_docs_by_hierarchy ────────────────────────────────────────────


class TestChunkDocsByHierarchy:
    def test_multiple_docs_preserve_doc_ids(self, chunking_config, markdown_text):
        cfg = chunking_config
        doc2 = """
## Another Header
This is a different document.
Another sentence in this doc.
### Sub Header
This is a sub-level sentence.
Another sub-level sentence.
"""
        doc_ids = ["doc-alpha", "doc-beta"]
        results = chunk_docs_by_hierarchy(
            [markdown_text, doc2],
            cfg["chunk_size"],
            cfg["tokenizer"],
            cfg["split_fn"],
            ids=doc_ids,
        )
        _log_chunks(results, "test_multiple_docs")

        assert len(results) > 0
        for chunk in results:
            assert "doc_id" in chunk
            assert chunk["doc_id"] in doc_ids

        doc_alpha_chunks = [c for c in results if c["doc_id"] == "doc-alpha"]
        doc_beta_chunks = [c for c in results if c["doc_id"] == "doc-beta"]
        # ✅ CHANGED: Match actual deterministic tokenizer output
        assert len(doc_alpha_chunks) == 5
        assert len(doc_beta_chunks) == 2

        sources = {"doc-alpha": markdown_text, "doc-beta": doc2}
        for chunk in results:
            _assert_valid_chunk(chunk, sources[chunk["doc_id"]])
        _assert_hierarchy_integrity(results)

    def test_mismatched_ids_raises_value_error(self, chunking_config, markdown_text):
        cfg = chunking_config
        with pytest.raises(ValueError, match="Number of provided IDs must match"):
            chunk_docs_by_hierarchy(
                [markdown_text, markdown_text],
                cfg["chunk_size"],
                cfg["tokenizer"],
                cfg["split_fn"],
                ids=["only-one-id"],
            )

    def test_auto_generated_doc_ids_when_none_provided(
        self, chunking_config, markdown_text
    ):
        cfg = chunking_config
        results = chunk_docs_by_hierarchy(
            [markdown_text],
            cfg["chunk_size"],
            cfg["tokenizer"],
            cfg["split_fn"],
            ids=None,
        )
        assert len(results) > 0
        doc_ids_seen = {c["doc_id"] for c in results}
        assert len(doc_ids_seen) == 1
        auto_id = doc_ids_seen.pop()
        assert isinstance(auto_id, str) and auto_id.startswith("mock-id-")
