"""Core Markdown chunking logic using Chonkie's MarkdownChef and RecursiveChunker."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from chonkie import Chunk, MarkdownChef, RecursiveChunker


@dataclass
class MarkdownChunkResult:
    """Container for chunked markdown content and extracted metadata."""

    chunks: List[Chunk]
    tables: List = field(default_factory=list)
    code_blocks: List = field(default_factory=list)
    images: List = field(default_factory=list)
    full_content: str = ""


def remove_empty_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Filter out chunks whose stripped text is empty.

    Args:
        chunks: List of Chunk objects to filter.

    Returns:
        Filtered list containing only non-empty chunks.
    """
    return [c for c in chunks if c.text.strip()]


def chunk_markdown(
    source: str,
    chunk_size: int = 512,
    lang: str = "en",
    keep_temp_file: bool = False,
) -> MarkdownChunkResult:
    """Parse and chunk markdown from a file path or raw text string.

    Automatically detects whether `source` is a local file path or raw
    markdown content. File paths are passed directly to MarkdownChef
    without creating a temporary copy. Raw text is written to a temp
    file for Chef processing.

    Args:
        source: Local .md file path OR raw markdown string.
        chunk_size: Target maximum tokens per chunk.
        lang: Language code for recipe selection.
        keep_temp_file: If True, preserves temp .md file (only applies
            when source is raw text, not a file path).

    Returns:
        MarkdownChunkResult containing chunks, tables, code blocks, and images.
    """
    source_path = Path(source)
    is_file = source_path.is_file()

    if is_file:
        console.log(f"[green]📄 Detected local file: {source_path.resolve()}[/]")
        chef = MarkdownChef()
        md_doc = chef.process(source_path)
    else:
        console.log(f"[dim]📝 Processing raw markdown ({len(source):,} chars)[/]")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(source)
            tmp_path = Path(tmp.name)
        try:
            chef = MarkdownChef()
            md_doc = chef.process(tmp_path)
        finally:
            if not keep_temp_file and tmp_path.exists():
                os.unlink(tmp_path)

    chunker = RecursiveChunker.from_recipe("markdown", lang=lang, chunk_size=chunk_size)
    chunks = chunker.chunk(md_doc.content)
    chunks = remove_empty_chunks(chunks)

    return MarkdownChunkResult(
        chunks=chunks,
        tables=md_doc.tables,
        code_blocks=md_doc.code,
        images=md_doc.images,
        full_content=md_doc.content,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from jet.adapters.chonkie.main._main_markdown_chunker import main

    main()
