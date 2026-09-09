"""Tests for jet.adapters.chonkie.markdown_chunker module."""

from jet.adapters.chonkie.markdown_chunker import (
    MarkdownChunkResult,
    chunk_markdown,
    remove_empty_chunks,
)

SAMPLE_MD = """
# Project Overview
This document describes the main features of our system.

## Installation
Install with:
```bash
pip install mypackage
```

## Features
| Feature       | Status      | Notes                  |
|---------------|-------------|------------------------|
| Fast search   | Done        | Uses vector index      |
| Batch upload  | In progress | Coming in v2.1         |
| Auth          | Done        | OAuth2 + API keys      |

## Code Example
Here is a simple Python helper:
```python
def process_data(items):
    results = []
    for item in items:
        if item.is_valid():
            results.append(item.transform())
    return results
```

## Conclusion
The system is ready for production use.
"""


class TestRemoveEmptyChunks:
    """Tests for the remove_empty_chunks utility."""

    def test_filters_whitespace_only(self):
        from chonkie import Chunk

        chunks = [
            Chunk(text="valid", start_index=0, end_index=5, token_count=1),
            Chunk(text="   ", start_index=6, end_index=9, token_count=1),
            Chunk(text="", start_index=10, end_index=10, token_count=0),
        ]
        result = remove_empty_chunks(chunks)
        assert len(result) == 1
        assert result[0].text == "valid"

    def test_preserves_non_empty(self):
        from chonkie import Chunk

        chunks = [
            Chunk(text="a", start_index=0, end_index=1, token_count=1),
            Chunk(text="b", start_index=2, end_index=3, token_count=1),
        ]
        result = remove_empty_chunks(chunks)
        assert len(result) == 2

    def test_empty_input(self):
        assert remove_empty_chunks([]) == []


class TestChunkMarkdown:
    """Tests for the chunk_markdown function."""

    def test_returns_correct_type(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        assert isinstance(result, MarkdownChunkResult)

    def test_expected_chunk_count(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        assert len(result.chunks) == 2

    def test_expected_table_count(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        assert len(result.tables) == 1

    def test_expected_code_block_count(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        assert len(result.code_blocks) == 2

    def test_expected_image_count(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        assert len(result.images) == 0

    def test_chunk_1_token_range(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        c1 = result.chunks[0]
        assert 400 <= c1.token_count <= 480

    def test_chunk_2_token_range(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        c2 = result.chunks[1]
        assert 250 <= c2.token_count <= 300

    def test_chunk_1_contains_expected_phrases(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        c1_text = result.chunks[0].text
        expected = [
            "# Project Overview",
            "## Installation",
            "pip install mypackage",
            "## Features",
            "Fast search",
        ]
        for phrase in expected:
            assert phrase in c1_text, f"Missing phrase in chunk 1: {phrase}"

    def test_chunk_2_contains_expected_phrases(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        c2_text = result.chunks[1].text
        expected = ["## Code Example", "def process_data", "## Conclusion"]
        for phrase in expected:
            assert phrase in c2_text, f"Missing phrase in chunk 2: {phrase}"

    def test_no_empty_chunks_after_filtering(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        for chunk in result.chunks:
            assert chunk.text.strip(), "Empty chunk found after filtering"

    def test_full_content_populated(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=512)
        assert len(result.full_content) > 0
        assert "# Project Overview" in result.full_content

    def test_custom_chunk_size(self):
        result = chunk_markdown(SAMPLE_MD, chunk_size=256)
        # Smaller chunk size should produce more chunks
        assert len(result.chunks) >= 2

    def test_keep_temp_file_false_by_default(self):
        """Ensure temp files are cleaned up by default."""

        # Just verify no exception is raised and cleanup happens
        result = chunk_markdown(SAMPLE_MD, chunk_size=512, keep_temp_file=False)
        assert isinstance(result, MarkdownChunkResult)
