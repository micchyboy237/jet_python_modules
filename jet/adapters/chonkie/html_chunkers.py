"""
Reusable HTML-aware chunkers for scraped web content.
Bridges Unstructured element classification with Chonkie chunking strategies.
Includes hierarchy reconstruction for complex nested header structures.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Union

from bs4 import BeautifulSoup
from jet.adapters.chonkie.llamacpp_tokenizer import LlamaCppTokenizer
from rich.console import Console

from chonkie.chunker.base import BaseChunker
from chonkie.chunker.code import CodeChunker
from chonkie.chunker.recursive import RecursiveChunker
from chonkie.chunker.semantic import SemanticChunker
from chonkie.chunker.table import TableChunker
from chonkie.tokenizer import TokenizerProtocol
from chonkie.types import Chunk, RecursiveLevel, RecursiveRules

console = Console()

_FENCED_CODE_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


@dataclass
class HierarchicalSection:
    """A section with full breadcrumb path preserved."""

    level: int
    heading_text: str
    breadcrumb: list[str]
    content_parts: list[str] = field(default_factory=list)
    page_number: Optional[int] = None
    source_url: Optional[str] = None

    @property
    def full_content(self) -> str:
        """Reconstruct as Markdown with breadcrumb context prepended."""
        lines = []
        if self.breadcrumb:
            lines.append(f"[{' > '.join(self.breadcrumb)}]")
            lines.append("")
        if self.heading_text:
            prefix = "#" * max(1, self.level)
            lines.append(f"{prefix} {self.heading_text}")
            lines.append("")
        lines.extend(self.content_parts)
        return "\n".join(lines).strip()


_CATEGORY_TO_LEVEL = {
    "Title": 1,
    "Headline": 1,
    "Subtitle": 2,
    "Subheadline": 2,
    "Section-header": 3,
}

_HEADING_CATEGORIES = frozenset(_CATEGORY_TO_LEVEL.keys())

_SKIP_CATEGORIES = frozenset(
    {
        "Header",
        "Footer",
        "PageBreak",
        "PageNumber",
        "Image",
        "FigureCaption",
        "Page-header",
        "Page-footer",
    }
)


def _ensure_v2_compatible_html(html: str) -> str:
    """Ensure HTML has the structure required by Unstructured's v2 ontology parser.

    The v2 parser requires either ``<body class="Document">`` or
    ``<div class="Page">`` as its entry point. This function safely adds the
    required wrapper without creating nested ``<body>`` elements:

    - If the HTML already contains ``class="Document"`` or ``class="Page"``,
      it is returned unchanged.
    - If a ``<body>`` tag exists, ``class="Document"`` is injected into it
      (preserving any existing classes and attributes).
    - If no ``<body>`` tag exists, the entire content is wrapped in
      ``<body class="Document">``.

    Uses BeautifulSoup for safe DOM manipulation instead of regex/string
    concatenation to avoid issues with malformed HTML, attribute quoting,
    or nested body elements.

    Args:
        html: Raw HTML string to normalize.

    Returns:
        HTML string guaranteed to satisfy v2 parser entry-point requirements.
    """
    # Fast-path: already ontology-compliant
    if 'class="Document"' in html or "class='Document'" in html:
        return html
    if 'class="Page"' in html or "class='Page'" in html:
        return html

    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")

    if body is not None:
        # Inject class="Document" into existing <body>, preserving other classes
        existing_classes = body.get("class", [])
        if isinstance(existing_classes, str):
            existing_classes = existing_classes.split()
        if "Document" not in existing_classes:
            body["class"] = ["Document"] + list(existing_classes)
    else:
        # No <body> at all — wrap entire content safely
        new_body = soup.new_tag("body", **{"class": "Document"})
        # Move all top-level contents into the new body
        children = list(soup.children)
        for child in children:
            new_body.append(child.extract() if hasattr(child, "extract") else child)
        soup.clear()
        soup.append(new_body)

    return str(soup)


def build_hierarchy(
    elements: list,
    source_url: Optional[str] = None,
) -> list[HierarchicalSection]:
    """
    Reconstruct heading hierarchy from Unstructured's flat element list.
    Uses Unstructured's actual element categories (Title, Subtitle, Section-header)
    to determine nesting depth, then builds breadcrumbs for each section.
    """
    sections: list[HierarchicalSection] = []
    stack: dict[int, str] = {}
    current_section: Optional[HierarchicalSection] = None

    def _get_heading_level(el) -> int:
        cat = getattr(el, "category", "")
        if cat not in ("Title", "Subtitle", "Headline", "Section-header"):
            return 0
        meta = getattr(el, "metadata", None)
        depth = getattr(meta, "category_depth", None) if meta else None
        if depth is not None and isinstance(depth, int):
            return depth + 1  # category_depth is 0-indexed (h1=0), we need 1-indexed
        return 0

    def _build_breadcrumb(level: int) -> list[str]:
        return [
            stack[lvl] for lvl in sorted(stack.keys()) if lvl <= level and stack[lvl]
        ]

    for el in elements:
        cat = getattr(el, "category", "UncategorizedText")
        text = str(el).strip()
        page_num = getattr(getattr(el, "metadata", None), "page_number", None)

        if not text or len(text) < 3:
            continue
        if cat in _SKIP_CATEGORIES:
            continue

        level = _get_heading_level(el)

        if level > 0:
            for lvl in list(stack.keys()):
                if lvl >= level:
                    del stack[lvl]
            stack[level] = text
            breadcrumb = _build_breadcrumb(level)
            current_section = HierarchicalSection(
                level=level,
                heading_text=text,
                breadcrumb=breadcrumb,
                page_number=page_num,
                source_url=source_url,
            )
            sections.append(current_section)
            console.log(f"  [cyan]H{level}:[/] {' > '.join(breadcrumb)}")
        else:
            if current_section is None:
                current_section = HierarchicalSection(
                    level=0,
                    heading_text="",
                    breadcrumb=[],
                    page_number=page_num,
                    source_url=source_url,
                )
                sections.append(current_section)
            current_section.content_parts.append(text)

    console.log(f"[bold green]✔ Built {len(sections)} hierarchical sections[/]")
    return sections


class HTMLAwareChunker(BaseChunker):
    """
    Composite chunker that auto-routes content to the best strategy
    based on lightweight structural detection.
    Expects CLEANED Markdown/text (not raw HTML).
    """

    def __init__(
        self,
        tokenizer: Union[str, TokenizerProtocol] = None,
        chunk_size: int = 512,
        min_chars_per_chunk: int = 50,
        semantic_model: str = "minishlab/potion-base-32M",
        semantic_threshold: float = 0.65,
        table_chunk_size: int = 5,
        code_language: str = "auto",
    ):
        if tokenizer is None:
            tokenizer = LlamaCppTokenizer()
        super().__init__(tokenizer=tokenizer)
        self.chunk_size = chunk_size
        self.min_chars_per_chunk = min_chars_per_chunk
        self._recursive_chunker = RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            rules=RecursiveRules(
                levels=[
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
        self._semantic_chunker: Optional[SemanticChunker] = None
        self._semantic_model = semantic_model
        self._semantic_threshold = semantic_threshold
        self._use_multiprocessing = False

    @property
    def semantic_chunker(self) -> SemanticChunker:
        if self._semantic_chunker is None:
            console.log(
                f"[yellow]Lazy-loading SemanticChunker ({self._semantic_model})[/]"
            )
            self._semantic_chunker = SemanticChunker(
                embedding_model=self._semantic_model,
                threshold=self._semantic_threshold,
                chunk_size=self.chunk_size,
                skip_window=2,
                min_sentences_per_chunk=2,
                min_characters_per_sentence=20,
            )
        return self._semantic_chunker

    @staticmethod
    def _has_markdown_table(text: str) -> bool:
        return bool(_TABLE_SEPARATOR_RE.search(text))

    @staticmethod
    def _has_fenced_code(text: str) -> bool:
        return "```" in text

    @staticmethod
    def _has_headings(text: str) -> bool:
        return bool(_HEADING_RE.search(text))

    @staticmethod
    def _extract_code_blocks(text: str) -> list[tuple[str, Optional[str], int, int]]:
        """Extract fenced code blocks with language hints and span positions."""
        results = []
        for m in _FENCED_CODE_RE.finditer(text):
            results.append((m.group(2), m.group(1), m.start(), m.end()))
        return results

    @staticmethod
    def _extract_tables(text: str) -> list[tuple[str, int, int]]:
        """Extract contiguous Markdown table blocks with span positions."""
        lines = text.split("\n")
        tables: list[tuple[str, int, int]] = []
        current_lines: list[str] = []
        block_start = 0
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
            char_pos += len(line) + 1
        if current_lines:
            tables.append(("\n".join(current_lines), block_start, char_pos - 1))
        return tables

    @staticmethod
    def _strip_code_and_tables(text: str) -> str:
        """Remove code blocks AND table lines, leaving only prose."""
        text = _FENCED_CODE_RE.sub("", text)
        lines = text.split("\n")
        prose_lines = [
            l
            for l in lines
            if not (l.strip().startswith("|") and l.strip().endswith("|"))
        ]
        return "\n".join(prose_lines).strip()

    def chunk(self, text: str) -> list[Chunk]:
        if not text or not text.strip():
            return []

        all_chunks: list[Chunk] = []
        code_count = table_count = prose_count = 0

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
                    console.log(f"[red]CodeChunker failed: {e}[/]")
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

        if self._has_markdown_table(text):
            for table_text, start, end in self._extract_tables(text):
                table_chunks = self._table_chunker.chunk(table_text)
                for c in table_chunks:
                    c.start_index = start + c.start_index
                    c.end_index = start + c.end_index
                all_chunks.extend(table_chunks)
                table_count += len(table_chunks)

        prose = self._strip_code_and_tables(text)
        if prose and len(prose.strip()) >= self.min_chars_per_chunk:
            if self._has_headings(prose):
                console.log(
                    "[green]Routing prose → RecursiveChunker (headings detected)[/]"
                )
                prose_chunks = self._recursive_chunker.chunk(prose)
            else:
                console.log("[yellow]No headings → falling back to SemanticChunker[/]")
                prose_chunks = self.semantic_chunker.chunk(prose)
            all_chunks.extend(prose_chunks)
            prose_count += len(prose_chunks)

        all_chunks.sort(key=lambda c: c.start_index)
        console.log(
            f"[bold green]✔ HTMLAwareChunker:[/] {len(all_chunks)} chunks "
            f"(code={code_count}, table={table_count}, prose={prose_count})"
        )
        return all_chunks

    def __repr__(self) -> str:
        return (
            f"HTMLAwareChunker(chunk_size={self.chunk_size}, "
            f"min_chars={self.min_chars_per_chunk})"
        )


@dataclass
class ChunkWithMetadata:
    """Extended chunk with source element metadata from Unstructured."""

    chunk: Chunk
    element_category: str = ""
    breadcrumb: str = ""
    page_number: Optional[int] = None
    source_url: Optional[str] = None


class ScrapedHTMLPipeline:
    """
    End-to-end pipeline: Raw HTML → Unstructured parsing →
    Hierarchy reconstruction → Smart chunking.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        tokenizer: Union[str, TokenizerProtocol] = None,
        min_chars_per_chunk: int = 50,
        semantic_model: str = "minishlab/potion-base-32M",
        table_rows_per_chunk: int = 5,
        skip_headers_footers: bool = True,
    ):
        if tokenizer is None:
            tokenizer = LlamaCppTokenizer()
        self.chunk_size = chunk_size
        self.min_chars_per_chunk = min_chars_per_chunk
        self.skip_headers_footers = skip_headers_footers
        self._table_chunker = TableChunker(
            tokenizer="row", chunk_size=table_rows_per_chunk
        )
        self._code_chunker = CodeChunker(
            tokenizer=tokenizer, chunk_size=chunk_size, language="auto"
        )
        self._recursive_chunker = RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            rules=RecursiveRules(
                levels=[
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
            console.log(
                f"[yellow]Lazy-loading SemanticChunker ({self._semantic_model})[/]"
            )
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
        """Parse raw HTML, reconstruct hierarchy, and chunk intelligently."""
        from unstructured.partition.html import partition_html

        console.log(f"[cyan]Parsing HTML ({len(html):,} chars)…[/]")

        # Ensure HTML satisfies v2 ontology parser entry-point requirements
        html = _ensure_v2_compatible_html(html)

        elements = partition_html(
            text=html,
            html_parser_version="v2",
            skip_headers_and_footers=self.skip_headers_footers,
            skip_nav=True,
        )

        if not elements:
            console.log("[red]⚠ No elements extracted from HTML[/]")
            return []

        sections = build_hierarchy(elements, source_url=source_url)

        results: list[ChunkWithMetadata] = []
        for section in sections:
            section_text = section.full_content
            if not section_text.strip():
                continue

            breadcrumb_str = (
                " > ".join(section.breadcrumb) if section.breadcrumb else ""
            )

            if self._has_markdown_table(section_text):
                chunks = self._table_chunker.chunk(section_text)
                cat_label = "Table"
            elif self._has_fenced_code(section_text):
                try:
                    chunks = self._code_chunker.chunk(section_text)
                except Exception as e:
                    console.log(f"[red]Code chunking failed: {e}[/]")
                    chunks = [
                        Chunk(
                            text=section_text,
                            start_index=0,
                            end_index=len(section_text),
                            token_count=len(section_text),
                        )
                    ]
                cat_label = "CodeSnippet"
            elif section.level > 0:
                console.log(f"[green]Section '{breadcrumb_str}' → RecursiveChunker[/]")
                chunks = self._recursive_chunker.chunk(section_text)
                cat_label = "StructuredText"
            else:
                console.log("[yellow]No-heading section → SemanticChunker[/]")
                chunks = self.semantic_chunker.chunk(section_text)
                cat_label = "UnstructuredText"

            for c in chunks:
                results.append(
                    ChunkWithMetadata(
                        chunk=c,
                        element_category=cat_label,
                        breadcrumb=breadcrumb_str,
                        page_number=section.page_number,
                        source_url=section.source_url,
                    )
                )

        console.log(
            f"[bold green]✔ Pipeline:[/] {len(results)} total chunks from {len(sections)} sections"
        )
        return results

    @staticmethod
    def _has_markdown_table(text: str) -> bool:
        return bool(_TABLE_SEPARATOR_RE.search(text))

    @staticmethod
    def _has_fenced_code(text: str) -> bool:
        return "```" in text

    async def aprocess(
        self,
        html: str,
        source_url: Optional[str] = None,
    ) -> list[ChunkWithMetadata]:
        """Async wrapper around process()."""
        import asyncio

        return await asyncio.to_thread(self.process, html, source_url)
