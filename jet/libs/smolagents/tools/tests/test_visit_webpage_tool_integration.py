"""Integration & functional tests for VisitWebpageTool."""

from __future__ import annotations

from pathlib import Path

import responses
from _pytest.monkeypatch import MonkeyPatch
from jet.libs.smolagents.tools.visit_webpage_tool import (
    PageFetchResult,
    VisitWebpageTool,
)
from rich.console import Console

# ─── Fetch ──────────────────────────────────────────────────────────────────────


@responses.activate
def test_fetch_url_success():
    # Given a valid URL that returns HTML
    url = "https://example.com/test-page"
    responses.get(url, body="<html><body>Hello World</body></html>", status=200)

    # When fetching the page
    tool = VisitWebpageTool(verbose=False)
    result = tool._fetch_url(url, log=Console().print)  # type: ignore

    # Then we get successful PageFetchResult with content
    assert isinstance(result, PageFetchResult)
    assert result.success is True
    assert "<html>" in result.html
    assert result.error_message is None


@responses.activate
def test_fetch_url_http_error():
    # Given a URL that returns 404
    url = "https://example.com/missing"
    responses.get(url, status=404)

    # When trying to fetch
    tool = VisitWebpageTool(verbose=False)
    result = tool._fetch_url(url, log=Console().print)  # type: ignore

    # Then we get failed result with error message
    assert result.success is False
    assert result.html == ""
    assert result.error_message is not None and (
        "404" in result.error_message or "Client" in result.error_message
    )


# ─── Full raw path ──────────────────────────────────────────────────────────────


def test_process_full_raw_calls_truncate(monkeypatch: MonkeyPatch):
    # Given markdown sections and mocked truncate
    sections = ["# Title\ntext", "## Section\nmore content"]
    fake_output = "TRUNCATED FULL CONTENT"

    called = {}

    def fake_truncate(texts, model, max_tokens):
        called.update(texts=texts, model=model, max_tokens=max_tokens)
        return fake_output

    monkeypatch.setattr("jet.wordnet.text_chunker.truncate_texts", fake_truncate)

    # When processing full raw
    tool = VisitWebpageTool(max_output_length=3000, verbose=False)
    result = tool._process_full_raw(sections)

    # Then truncate is called correctly and result returned
    assert result == fake_output
    assert called["texts"] == sections
    assert called["max_tokens"] == 3000


@responses.activate
def test_forward_smoke_full_raw(monkeypatch: MonkeyPatch, tmp_path: Path):
    # Given a simple page and mocked truncate
    url = "https://example.com/smoke"
    responses.get(url, body="# Hello\nSome content here", status=200)

    # Force string return even if real truncate_texts returns list
    monkeypatch.setattr(
        "jet.wordnet.text_chunker.truncate_texts",
        lambda texts, model=None, max_tokens=None: (
            "\n\n".join(texts[:1])[: max_tokens or 8000] + " … [truncated]"
        ),
    )

    # When calling forward in full_raw mode with logging
    tool = VisitWebpageTool(
        verbose=False,
        logs_dir=tmp_path / "tool-logs",
    )
    result = tool.forward(url=url, full_raw=True)

    # Added safety: ensure result is a string and contains [truncated]
    assert isinstance(result, str)
    assert "… [truncated]" in result or "[TRUNC]" in result
    log_dir = tmp_path / "tool-logs"
    assert (log_dir / "page.html").exists()
    assert (log_dir / "full_results.md").exists()
    assert (log_dir / "truncated_text.md").exists()


# TODO: add smart-path integration test once you have a good mocking strategy
#       for chunk_texts_with_data + HybridSearcher (heavy dependencies)
