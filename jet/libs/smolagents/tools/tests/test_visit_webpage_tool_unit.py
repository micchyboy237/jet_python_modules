"""Unit tests for VisitWebpageTool – pure functions & lightweight helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from jet.libs.smolagents.tools.visit_webpage_tool import (
    DebugSaver,
    SearchResult,
    build_excerpts,
    build_result_header,
    create_search_documents,
    extract_markdown_section_texts,
    format_final_result,
    resolve_search_query,
)

# ─── Pure function tests ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "input_query, expected",
    [
        (None, "main content and key information from the webpage"),
        ("", "main content and key information from the webpage"),
        ("   ", "main content and key information from the webpage"),
        ("find python version", "find python version"),
        ("  latest features  ", "latest features"),
    ],
)
def test_resolve_search_query(input_query: str | None, expected: str):
    # Given a query or None/empty
    # When resolve_search_query is called
    # Then returns stripped input or fallback
    result = resolve_search_query(input_query)
    expected_result = expected
    assert result == expected_result


def test_build_excerpts():
    # Given mock search results
    fake_results = [
        SearchResult(item={"content": "First paragraph"}, score=0.92),
        SearchResult(item={"content": "Second text\nwith newline"}, score=0.87),
        SearchResult(item={"content": "Low relevance"}, score=0.41),
    ]

    # When formatting top 2
    excerpts = build_excerpts(fake_results, max_count=2)

    # Then get correctly formatted numbered excerpts
    expected_excerpts = [
        "[1] Relevance 0.920\nFirst paragraph\n",
        "[2] Relevance 0.870\nSecond text\nwith newline\n",
    ]
    assert excerpts == expected_excerpts


def test_build_result_header():
    header = build_result_header(
        url="https://example.com/guide",
        search_query="best python practices 2026",
    )
    assert "https://example.com/guide" in header
    assert "best python practices 2026" in header
    assert "hybrid BM25 + embedding retrieval" in header


def test_format_final_result():
    header = "Most relevant excerpts\n\n"
    excerpts = ["[1] abc...", "[2] def..."]
    result = format_final_result(header, excerpts)
    expected_result = "Most relevant excerpts\n\n[1] abc...\n[2] def..."
    assert result == expected_result


def test_create_search_documents():
    chunks = [
        {"id": "c0", "content": "text one", "extra": 123},
        {"id": "c1", "content": "text two"},
    ]
    docs = create_search_documents(chunks)
    expected_docs = [
        {"id": "c0", "content": "text one"},
        {"id": "c1", "content": "text two"},
    ]
    assert docs == expected_docs


# ─── DebugSaver ─────────────────────────────────────────────────────────────────


def test_debug_saver_no_dir_does_nothing(tmp_path: Path):
    # Given saver with no directory
    saver = DebugSaver(base_dir=None)

    # When saving files
    saver.save("dummy.txt", "content")
    saver.save_json("data.json", {"x": 42})

    # Then nothing is written to disk
    assert not (tmp_path / "dummy.txt").exists()


def test_debug_saver_writes_files(tmp_path: Path):
    # Given saver pointing to temp directory
    saver = DebugSaver(base_dir=tmp_path)

    # When saving text and json
    saver.save("note.md", "# Test\ncontent")
    saver.save_json("stats.json", {"count": 7, "items": [1, 2]}, indent=2)

    # Then files exist with correct content
    assert (tmp_path / "note.md").read_text(encoding="utf-8") == "# Test\ncontent"
    assert json.loads((tmp_path / "stats.json").read_text()) == {
        "count": 7,
        "items": [1, 2],
    }


# ─── extract_markdown_section_texts ─────────────────────────────────────────────


def test_extract_markdown_section_texts(monkeypatch: MonkeyPatch):
    # Given mocked header extraction
    fake_blocks = [
        {"header": "## Overview", "content": "Intro text\nSecond line"},
        {"header": "### API", "content": "Endpoint list"},
    ]
    monkeypatch.setattr(
        "jet.code.splitter_markdown_utils.get_md_header_contents",
        lambda html, ignore_links: fake_blocks,
    )

    # When extracting markdown texts
    md_texts = extract_markdown_section_texts("<html></html>", ignore_links=True)

    # Then we get properly formatted blocks
    expected_texts = [
        "## Overview\n\nIntro text\nSecond line",
        "### API\n\nEndpoint list",
    ]
    assert md_texts == expected_texts
