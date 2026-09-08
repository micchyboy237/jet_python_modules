"""
Reusable HTML-aware chunkers for scraped web content.
Bridges Unstructured element classification with Chonkie chunking strategies.
"""

import re
from dataclasses import dataclass
from typing import Optional, Union

from rich.console import Console

from chonkie.chunker.base import BaseChunker
from chonkie.chunker.code import CodeChunker
from chonkie.chunker.recursive import RecursiveChunker
from chonkie.chunker.semantic import SemanticChunker
from chonkie.chunker.table import TableChunker
from chonkie.tokenizer import TokenizerProtocol
from chonkie.types import Chunk, RecursiveLevel, RecursiveRules

# Rich console + logger setup
console = Console()
logger = console.log

# Pre-compiled patterns (avoid recompilation on every call)
_FENCED_CODE_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


# ---------------------------------------------------------------------------
# 1. HTML-Aware Chunker (Custom Chonkie Chunker)
# ---------------------------------------------------------------------------


class HTMLAwareChunker(BaseChunker):
    """
    A composite chunker that auto-routes content to the best strategy
    based on lightweight structural detection.

    Expects CLEANED Markdown/text (not raw HTML). Use ScrapedHTMLPipeline
    for end-to-end raw HTML processing.
    """

    def __init__(
        self,
        tokenizer: Union[str, TokenizerProtocol] = "gpt2",
        chunk_size: int = 512,
        min_chars_per_chunk: int = 50,
        semantic_model: str = "minishlab/potion-base-32M",
        semantic_threshold: float = 0.65,
        table_chunk_size: int = 5,
        code_language: str = "auto",
    ):
        super().__init__(tokenizer=tokenizer)
        self.chunk_size = chunk_size
        self.min_chars_per_chunk = min_chars_per_chunk

        # Pre-initialize sub-chunkers for reuse
        self._recursive_chunker = RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            rules=RecursiveRules(
                rules=[
                    RecursiveLevel(
                        delimiters=["\n# ", "\n## ", "\n### "], include_delim="next"
                    ),
                    RecursiveLevel(delimiters=["\n\n"], include_delim="next"),
                    RecursiveLevel(delimiters=[". ", "! ", "? "], include_delim="prev"),
                    RecursiveLevel(whitespace=True),
                ]
            ),
            min_characters_per_chunk=min_chars_per_chunk,
        )

        self._table_chunker = TableChunker(tokenizer="row", chunk_size=table_chunk_size)

        self._code_chunker = CodeChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            language=code_language,
        )

        # Lazy-loaded semantic chunker (heavy model)
        self._semantic_chunker: Optional[SemanticChunker] = None
        self._semantic_model = semantic_model
        self._semantic_threshold = semantic_threshold

        self._use_multiprocessing = False

    @property
    def semantic_chunker(self) -> SemanticChunker:
        if self._semantic_chunker is None:
            logger(f"[yellow]Lazy-loading SemanticChunker ({self._semantic_model})[/]")
            self._semantic_chunker = SemanticChunker(
                embedding_model=self._semantic_model,
                threshold=self._semantic_threshold,
                chunk_size=self.chunk_size,
                skip_window=2,
                min_sentences_per_chunk=2,
                min_characters_per_sentence=20,
            )
        return self._semantic_chunker

    # --- Detection heuristics (fast, no ML) ---

    @staticmethod
    def _has_markdown_table(text: str) -> bool:
        """Check for Markdown table separator pattern |---|"""
        return bool(_TABLE_SEPARATOR_RE.search(text))

    @staticmethod
    def _has_fenced_code(text: str) -> bool:
        """Check for fenced code blocks ```...```"""
        return "```" in text

    @staticmethod
    def _has_headings(text: str) -> bool:
        """Check for Markdown heading markers"""
        return bool(_HEADING_RE.search(text))

    @staticmethod
    def _extract_code_blocks(text: str) -> list[tuple[str, Optional[str], int, int]]:
        """Extract fenced code blocks with language hints and span positions."""
        results = []
        for m in _FENCED_CODE_RE.finditer(text):
            lang_hint = m.group(1)
            code_content = m.group(2)
            results.append((code_content, lang_hint, m.start(), m.end()))
        return results

    @staticmethod
    def _extract_tables(text: str) -> list[tuple[str, int, int]]:
        """Extract contiguous Markdown table blocks with span positions."""
        lines = text.split("\n")
        tables: list[tuple[str, int, int]] = []
        current_lines: list[str] = []
        block_start: int = 0
        char_pos = 0

        for line in lines:
            stripped = line.strip()
            is_table_line = stripped.startswith("|") and stripped.endswith("|")
            if is_table_line:
                if not current_lines:
                    block_start = char_pos
                current_lines.append(line)
            else:
                if current_lines:
                    tables.append(("\n".join(current_lines), block_start, char_pos - 1))
                    current_lines = []
            char_pos += len(line) + 1  # +1 for newline

        if current_lines:
            tables.append(("\n".join(current_lines), block_start, char_pos - 1))
        return tables

    @staticmethod
    def _strip_code_and_tables(text: str) -> str:
        """Remove code blocks AND table lines, leaving only prose."""
        # Remove fenced code blocks first
        text = _FENCED_CODE_RE.sub("", text)
        # Remove table lines
        lines = text.split("\n")
        prose_lines = [
            l
            for l in lines
            if not (l.strip().startswith("|") and l.strip().endswith("|"))
        ]
        return "\n".join(prose_lines).strip()

    # --- Core chunking ---

    def chunk(self, text: str) -> list[Chunk]:
        if not text or not text.strip():
            return []

        all_chunks: list[Chunk] = []
        code_count = 0
        table_count = 0
        prose_count = 0

        # 1. Extract and chunk code blocks (with accurate positions)
        if self._has_fenced_code(text):
            for code_text, lang_hint, start, end in self._extract_code_blocks(text):
                if not code_text.strip():
                    continue
                try:
                    if lang_hint:
                        code_chunks = self._code_chunker._chunk_code_block(
                            code_text, lang_hint
                        )
                    else:
                        code_chunks = self._code_chunker.chunk(code_text)
                    for c in code_chunks:
                        c.start_index = start + c.start_index
                        c.end_index = start + c.end_index
                    all_chunks.extend(code_chunks)
                    code_count += len(code_chunks)
                except Exception as e:
                    logger(f"[red]CodeChunker failed: {e}[/]")
                    token_count = self.tokenizer.count_tokens(code_text)
                    all_chunks.append(
                        Chunk(
                            text=code_text,
                            start_index=start,
                            end_index=end,
                            token_count=token_count,
                        )
                    )
                    code_count += 1

        # 2. Extract and chunk tables (with accurate positions)
        if self._has_markdown_table(text):
            for table_text, start, end in self._extract_tables(text):
                table_chunks = self._table_chunker.chunk(table_text)
                for c in table_chunks:
                    c.start_index = start + c.start_index
                    c.end_index = start + c.end_index
                all_chunks.extend(table_chunks)
                table_count += len(table_chunks)

        # 3. Chunk remaining prose
        prose = self._strip_code_and_tables(text)
        if prose and len(prose.strip()) >= self.min_chars_per_chunk:
            if self._has_headings(prose):
                logger("[green]Routing prose → RecursiveChunker (headings detected)[/]")
                prose_chunks = self._recursive_chunker.chunk(prose)
            else:
                logger("[yellow]No headings → falling back to SemanticChunker[/]")
                prose_chunks = self.semantic_chunker.chunk(prose)
            all_chunks.extend(prose_chunks)
            prose_count += len(prose_chunks)

        # Sort by position
        all_chunks.sort(key=lambda c: c.start_index)
        logger(
            f"[bold green]✔ HTMLAwareChunker:[/] {len(all_chunks)} chunks "
            f"(code={code_count}, table={table_count}, prose={prose_count})"
        )
        return all_chunks

    def __repr__(self) -> str:
        return (
            f"HTMLAwareChunker(chunk_size={self.chunk_size}, "
            f"min_chars={self.min_chars_per_chunk})"
        )


# ---------------------------------------------------------------------------
# 2. Scraped HTML Pipeline (Unstructured → Chonkie Bridge)
# ---------------------------------------------------------------------------


@dataclass
class ChunkWithMetadata:
    """Extended chunk with source element metadata from Unstructured."""

    chunk: Chunk
    element_category: str = ""
    page_number: Optional[int] = None
    source_url: Optional[str] = None


class ScrapedHTMLPipeline:
    """
    End-to-end pipeline: Raw HTML → Unstructured parsing → Smart chunking.
    """

    TABLE_CATEGORIES = {"Table"}
    CODE_CATEGORIES = {"CodeSnippet"}
    HEADING_CATEGORIES = {"Title", "Section-header", "Headline", "Subheadline"}
    SKIP_CATEGORIES = {"Header", "Footer", "PageBreak", "PageNumber", "Image"}

    def __init__(
        self,
        chunk_size: int = 512,
        min_chars_per_chunk: int = 50,
        semantic_model: str = "minishlab/potion-base-32M",
        table_rows_per_chunk: int = 5,
        skip_headers_footers: bool = True,
    ):
        self.chunk_size = chunk_size
        self.min_chars_per_chunk = min_chars_per_chunk
        self.skip_headers_footers = skip_headers_footers

        self._table_chunker = TableChunker(
            tokenizer="row", chunk_size=table_rows_per_chunk
        )
        self._code_chunker = CodeChunker(
            tokenizer="gpt2", chunk_size=chunk_size, language="auto"
        )
        self._recursive_chunker = RecursiveChunker(
            tokenizer="gpt2",
            chunk_size=chunk_size,
            rules=RecursiveRules(
                rules=[
                    RecursiveLevel(
                        delimiters=["\n# ", "\n## ", "\n### "], include_delim="next"
                    ),
                    RecursiveLevel(delimiters=["\n\n"], include_delim="next"),
                    RecursiveLevel(delimiters=[". ", "! ", "? "], include_delim="prev"),
                    RecursiveLevel(whitespace=True),
                ]
            ),
            min_characters_per_chunk=min_chars_per_chunk,
        )
        self._semantic_chunker: Optional[SemanticChunker] = None
        self._semantic_model = semantic_model

    @property
    def semantic_chunker(self) -> SemanticChunker:
        if self._semantic_chunker is None:
            logger(f"[yellow]Lazy-loading SemanticChunker ({self._semantic_model})[/]")
            self._semantic_chunker = SemanticChunker(
                embedding_model=self._semantic_model,
                threshold=0.65,
                chunk_size=self.chunk_size,
                skip_window=2,
                min_sentences_per_chunk=2,
            )
        return self._semantic_chunker

    def process(
        self,
        html: str,
        source_url: Optional[str] = None,
    ) -> list[ChunkWithMetadata]:
        """Parse raw HTML and return intelligently chunked results."""
        from unstructured.partition.html import partition_html

        logger(f"[cyan]Parsing HTML ({len(html):,} chars)…[/]")
        elements = partition_html(
            text=html,
            skip_headers_and_footers=self.skip_headers_footers,
            skip_nav=True,
        )

        if not elements:
            logger("[red]⚠ No elements extracted from HTML[/]")
            return []

        # Classify elements into buckets
        table_texts = []
        code_texts = []
        structured_texts = []
        has_headings = False

        for el in elements:
            cat = getattr(el, "category", "UncategorizedText")
            text = str(el).strip()

            if not text or len(text) < 10:
                continue
            if cat in self.SKIP_CATEGORIES:
                continue
            elif cat in self.TABLE_CATEGORIES:
                table_texts.append((text, el))
            elif cat in self.CODE_CATEGORIES:
                code_texts.append((text, el))
            elif cat in self.HEADING_CATEGORIES:
                has_headings = True
                structured_texts.append((text, el))
            else:
                structured_texts.append((text, el))

        results: list[ChunkWithMetadata] = []

        # Chunk tables
        for table_text, el in table_texts:
            chunks = self._table_chunker.chunk(table_text)
            for c in chunks:
                results.append(
                    ChunkWithMetadata(
                        chunk=c,
                        element_category="Table",
                        page_number=getattr(el.metadata, "page_number", None),
                        source_url=source_url,
                    )
                )

        # Chunk code
        for code_text, el in code_texts:
            try:
                chunks = self._code_chunker.chunk(code_text)
            except Exception as e:
                logger(f"[red]Code chunking failed: {e}[/]")
                chunks = [
                    Chunk(
                        text=code_text,
                        start_index=0,
                        end_index=len(code_text),
                        token_count=len(code_text),
                    )
                ]
            for c in chunks:
                results.append(
                    ChunkWithMetadata(
                        chunk=c,
                        element_category="CodeSnippet",
                        page_number=getattr(el.metadata, "page_number", None),
                        source_url=source_url,
                    )
                )

        # Chunk prose
        combined_prose = "\n\n".join(t for t, _ in structured_texts)
        if combined_prose.strip():
            if has_headings:
                logger("[green]Routing prose → RecursiveChunker[/]")
                chunks = self._recursive_chunker.chunk(combined_prose)
                cat_label = "StructuredText"
            else:
                logger("[yellow]Routing prose → SemanticChunker (no headings)[/]")
                chunks = self.semantic_chunker.chunk(combined_prose)
                cat_label = "UnstructuredText"

            page_nums = [
                getattr(el.metadata, "page_number", None) for _, el in structured_texts
            ]
            for c in chunks:
                results.append(
                    ChunkWithMetadata(
                        chunk=c,
                        element_category=cat_label,
                        page_number=page_nums[0] if page_nums else None,
                        source_url=source_url,
                    )
                )

        logger(
            f"[bold green]✔ Pipeline:[/] {len(results)} chunks "
            f"(table={len(table_texts)}, code={len(code_texts)}, prose={len(structured_texts)})"
        )
        return results

    async def aprocess(
        self,
        html: str,
        source_url: Optional[str] = None,
    ) -> list[ChunkWithMetadata]:
        """Async wrapper around process()."""
        import asyncio

        return await asyncio.to_thread(self.process, html, source_url)
