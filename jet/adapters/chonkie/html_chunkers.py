"""
Reusable HTML-aware chunkers for scraped web content.
Uses jet/scrapers for robust header hierarchy detection and Chonkie for smart chunking.
Supports direct URL input via Playwright scraping utilities.

Updated: Replaced CodeChunker(language="auto") with LLMDetectedCodeChunker
to eliminate Magika/trial-parse bottleneck during code language detection.
"""

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from bs4 import BeautifulSoup
from jet.adapters.chonkie.llamacpp_tokenizer import LlamaCppTokenizer
from jet.adapters.chonkie.llm_code_chunker import LLMDetectedCodeChunker
from jet.scrapers.header_hierarchy import HtmlHeaderDoc, extract_header_hierarchy
from jet.scrapers.playwright_utils import scrape_urls, scrape_urls_sync
from rich.console import Console

from chonkie.chunker.base import BaseChunker
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
    html_fragment: str = ""
    global_start: int = 0
    global_end: int = 0

    @property
    def full_content(self) -> str:
        """Reconstruct as clean text. Breadcrumb is metadata-only, NOT embedded."""
        lines = []
        if self.heading_text:
            prefix = "#" * max(1, self.level)
            lines.append(f"{prefix} {self.heading_text}")
            lines.append("")
        lines.extend(self.content_parts)
        return "\n".join(lines).strip()


def _map_header_docs_to_sections(
    header_docs: list[HtmlHeaderDoc],
    source_url: Optional[str] = None,
    raw_html: str = "",
) -> list[HierarchicalSection]:
    """
    Adapts jet.scrapers HtmlHeaderDoc output to HierarchicalSection.
    Fixes content leakage by filtering child headings from parent content.
    Preserves raw HTML for accurate content-type detection.
    """
    sections: list[HierarchicalSection] = []
    all_headings = {
        doc.get("header", "").strip() for doc in header_docs if doc.get("header")
    }

    for i, doc in enumerate(header_docs):
        content = (doc.get("content") or "").strip()
        header = (doc.get("header") or "").strip()
        html_frag = (doc.get("html") or "").strip()

        if not content and not header:
            continue

        # Filter out child heading text that leaked into parent content
        filtered_lines = []
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped and stripped not in all_headings:
                filtered_lines.append(line)
            elif not stripped:
                filtered_lines.append(line)
        filtered_content = "\n".join(filtered_lines).strip()

        # Map HTML fragment position back to raw HTML for accurate span tracking
        global_start = 0
        global_end = 0
        if raw_html and html_frag:
            idx = raw_html.find(html_frag)
            if idx != -1:
                global_start = idx
                global_end = idx + len(html_frag)

        section = HierarchicalSection(
            level=doc.get("level", 0) or 0,
            heading_text=header,
            breadcrumb=doc.get("parent_headers") or [],
            content_parts=[filtered_content] if filtered_content else [],
            page_number=None,
            source_url=source_url,
            html_fragment=html_frag,
            global_start=global_start,
            global_end=global_end,
        )
        sections.append(section)

    console.log(
        f"[bold green]✔ Mapped {len(sections)} sections from header hierarchy[/]"
    )
    return sections


def _merge_short_sections(
    sections: list[HierarchicalSection],
    min_tokens: int = 50,
    max_tokens: int = 512,
    tokenizer=None,
) -> list[HierarchicalSection]:
    """
    Merge adjacent short sections that share the same parent breadcrumb
    to improve token utilization. Never merges across different parents.
    """
    if not sections or tokenizer is None:
        return sections

    merged: list[HierarchicalSection] = []
    buffer: Optional[HierarchicalSection] = None

    for section in sections:
        text = section.full_content
        tokens = tokenizer.count_tokens(text) if text else 0

        if buffer is None:
            if tokens < min_tokens:
                buffer = section
            else:
                merged.append(section)
            continue

        buffer_parent = " > ".join(buffer.breadcrumb) if buffer.breadcrumb else ""
        section_parent = " > ".join(section.breadcrumb) if section.breadcrumb else ""
        combined_text = buffer.full_content + "\n" + text
        combined_tokens = tokenizer.count_tokens(combined_text)
        same_parent = buffer_parent == section_parent
        fits = combined_tokens <= max_tokens

        if same_parent and fits and tokens < min_tokens:
            buffer.content_parts.extend(section.content_parts)
            buffer.html_fragment += "\n" + section.html_fragment
            buffer.global_end = max(buffer.global_end, section.global_end)
        else:
            merged.append(buffer)
            if tokens < min_tokens:
                buffer = section
            else:
                buffer = None
                merged.append(section)

    if buffer is not None:
        merged.append(buffer)

    if len(merged) < len(sections):
        console.log(
            f"[cyan]Merged {len(sections)} sections → {len(merged)} "
            f"(min_tokens={min_tokens}, max_tokens={max_tokens})[/]"
        )
    return merged


class HTMLAwareChunker(BaseChunker):
    """
    Composite chunker that auto-routes content to the best strategy
    based on lightweight structural detection.
    Expects CLEANED Markdown/text (not raw HTML).

    Uses LLMDetectedCodeChunker for code blocks when language="auto"
    to avoid the Magika/trial-parse bottleneck.
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
        llm_model: Optional[str] = None,
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
                    RecursiveLevel(delimiters=["\n"], include_delim="next"),
                    RecursiveLevel(delimiters=[". ", "! ", "? "], include_delim="prev"),
                    RecursiveLevel(whitespace=True),
                ]
            ),
            min_characters_per_chunk=min_chars_per_chunk,
        )
        self._table_chunker = TableChunker(tokenizer="row", chunk_size=table_chunk_size)

        # Use LLMDetectedCodeChunker for auto-detection to avoid Magika bottleneck
        if code_language == "auto":
            self._code_chunker = LLMDetectedCodeChunker(
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                language="auto",
                llm_model=llm_model,
            )
            console.log("[green]Using LLM-based code language detection[/]")
        else:
            from chonkie.chunker.code import CodeChunker

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

        # Handle fenced code blocks
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

        # Handle markdown tables
        if self._has_markdown_table(text):
            for table_text, start, end in self._extract_tables(text):
                table_chunks = self._table_chunker.chunk(table_text)
                for c in table_chunks:
                    c.start_index = start + c.start_index
                    c.end_index = start + c.end_index
                all_chunks.extend(table_chunks)
                table_count += len(table_chunks)

        # Handle prose (remaining text after stripping code/tables)
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
    """Extended chunk with source element metadata."""

    chunk: Chunk
    element_category: str = ""
    breadcrumb: str = ""
    page_number: Optional[int] = None
    source_url: Optional[str] = None


class ScrapedHTMLPipeline:
    """
    End-to-end pipeline: Raw HTML or URL(s) → jet/scrapers hierarchy extraction → Smart chunking.
    Uses extract_header_hierarchy for accurate DOM-based header detection.

    Uses LLMDetectedCodeChunker for code blocks to avoid Magika/trial-parse bottleneck.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        tokenizer: Union[str, TokenizerProtocol] = None,
        min_chars_per_chunk: int = 50,
        semantic_model: str = "minishlab/potion-base-32M",
        table_rows_per_chunk: int = 5,
        headless: bool = True,
        use_cache: bool = False,
        scroll_strategy: str = "until_stable",
        min_section_tokens: int = 50,
        llm_model: Optional[str] = None,
    ):
        if tokenizer is None:
            tokenizer = LlamaCppTokenizer()
        self.chunk_size = chunk_size
        self.min_chars_per_chunk = min_chars_per_chunk
        self.min_section_tokens = min_section_tokens
        self.headless = headless
        self.use_cache = use_cache
        self.scroll_strategy = scroll_strategy
        self._tokenizer = tokenizer

        self._table_chunker = TableChunker(
            tokenizer="row", chunk_size=table_rows_per_chunk
        )

        # Use LLMDetectedCodeChunker instead of CodeChunker(language="auto")
        self._code_chunker = LLMDetectedCodeChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            language="auto",
            llm_model=llm_model,
        )
        console.log("[green]Pipeline using LLM-based code language detection[/]")

        self._recursive_chunker = RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            rules=RecursiveRules(
                levels=[
                    RecursiveLevel(
                        delimiters=["\n# ", "\n## ", "\n### "], include_delim="next"
                    ),
                    RecursiveLevel(delimiters=["\n"], include_delim="next"),
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

    @staticmethod
    def _normalize_input(source: Union[str, List[str]]) -> tuple[List[str], bool]:
        """
        Normalize input to list of URLs or detect raw HTML.
        Returns: (list_of_sources, is_url)
        """
        if isinstance(source, str):
            if source.strip().startswith(("http://", "https://")):
                return [source], True
            return [source], False
        if not source:
            return [], False
        first_valid = next((s for s in source if s and s.strip()), "")
        is_url = first_valid.strip().startswith(("http://", "https://"))
        return list(source), is_url

    @staticmethod
    def _has_html_table(html_fragment: str) -> bool:
        """Detect tables via DOM inspection of raw HTML fragment."""
        if not html_fragment:
            return False
        try:
            soup = BeautifulSoup(html_fragment, "html.parser")
            return bool(soup.find("table"))
        except Exception:
            return False

    @staticmethod
    def _has_html_code(html_fragment: str) -> bool:
        """Detect code blocks via DOM inspection of raw HTML fragment."""
        if not html_fragment:
            return False
        try:
            soup = BeautifulSoup(html_fragment, "html.parser")
            return bool(soup.find("pre") or soup.find("code"))
        except Exception:
            return False

    @staticmethod
    def _extract_html_table_as_markdown(html_fragment: str) -> Optional[str]:
        """Convert HTML table to Markdown pipe format for TableChunker."""
        if not html_fragment:
            return None
        try:
            soup = BeautifulSoup(html_fragment, "html.parser")
            table = soup.find("table")
            if not table:
                return None
            rows = table.find_all("tr")
            if not rows:
                return None
            md_lines = []
            for i, row in enumerate(rows):
                cells = row.find_all(["th", "td"])
                cell_texts = [c.get_text(strip=True).replace("|", "\\|") for c in cells]
                md_lines.append("| " + " | ".join(cell_texts) + " |")
                if i == 0:
                    md_lines.append("| " + " | ".join(["---"] * len(cell_texts)) + " |")
            return "\n".join(md_lines)
        except Exception as e:
            console.log(f"[red]HTML→Markdown table conversion failed: {e}[/]")
            return None

    @staticmethod
    def _extract_html_code_text(html_fragment: str) -> Optional[str]:
        """Extract code text with fences from HTML pre/code elements."""
        if not html_fragment:
            return None
        try:
            soup = BeautifulSoup(html_fragment, "html.parser")
            pre = soup.find("pre")
            if pre:
                code = pre.find("code")
                lang = ""
                if code:
                    classes = code.get("class", [])
                    for cls in classes:
                        if cls.startswith("language-"):
                            lang = cls.replace("language-", "")
                            break
                    text = code.get_text()
                else:
                    text = pre.get_text()
                return f"```{lang}\n{text.strip()}\n```"
            code = soup.find("code")
            if code:
                return f"```\n{code.get_text().strip()}\n```"
            return None
        except Exception as e:
            console.log(f"[red]HTML code extraction failed: {e}[/]")
            return None

    def _process_html(
        self, html: str, source_url: Optional[str]
    ) -> list[ChunkWithMetadata]:
        """Core processing logic using jet/scrapers for hierarchy detection."""
        console.log(f"[cyan]Extracting header hierarchy ({len(html):,} chars)…[/]")
        try:
            header_docs = extract_header_hierarchy(
                source=html,
                excludes=["nav", "footer", "script", "style"],
                timeout_ms=10000,
            )
            sections = _map_header_docs_to_sections(
                header_docs, source_url=source_url, raw_html=html
            )
        except Exception as e:
            console.log(
                f"[red]⚠ Header hierarchy extraction failed: {e}. Falling back to flat chunking.[/]"
            )
            sections = [
                HierarchicalSection(
                    level=0,
                    heading_text="",
                    breadcrumb=[],
                    content_parts=[html],
                    source_url=source_url,
                    html_fragment=html,
                    global_start=0,
                    global_end=len(html),
                )
            ]

        if not sections:
            console.log("[red]⚠ No sections extracted from HTML[/]")
            return []

        sections = _merge_short_sections(
            sections,
            min_tokens=self.min_section_tokens,
            max_tokens=self.chunk_size,
            tokenizer=self._tokenizer,
        )

        results: list[ChunkWithMetadata] = []
        for section in sections:
            section_text = section.full_content
            if not section_text.strip():
                continue

            breadcrumb_str = (
                " > ".join(section.breadcrumb) if section.breadcrumb else ""
            )
            has_table = self._has_html_table(section.html_fragment)
            has_code = self._has_html_code(section.html_fragment)

            if has_table:
                md_table = self._extract_html_table_as_markdown(section.html_fragment)
                if md_table:
                    chunks = self._table_chunker.chunk(md_table)
                else:
                    chunks = self._recursive_chunker.chunk(section_text)
                cat_label = "Table"
            elif has_code:
                fenced = self._extract_html_code_text(section.html_fragment)
                if fenced:
                    try:
                        chunks = self._code_chunker.chunk(fenced)
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
                else:
                    chunks = self._recursive_chunker.chunk(section_text)
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
                if section.global_start > 0:
                    c.start_index = section.global_start + c.start_index
                    c.end_index = section.global_start + c.end_index
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

    def process(
        self,
        source: Union[str, List[str]],
        source_url: Optional[str] = None,
    ) -> list[ChunkWithMetadata]:
        """
        Parse raw HTML or URL(s), reconstruct hierarchy via jet/scrapers, and chunk intelligently.

        Args:
            source: Raw HTML string, URL string, or list of URLs/HTML strings.
            source_url: Optional override for source URL metadata (only used if source is raw HTML).
        """
        sources, is_url = self._normalize_input(source)
        all_results: list[ChunkWithMetadata] = []

        if is_url:
            console.log(f"[cyan]Fetching {len(sources)} URL(s) via Playwright…[/]")
            for result in scrape_urls_sync(
                urls=sources,
                headless=self.headless,
                use_cache=self.use_cache,
                scroll_strategy=self.scroll_strategy,
                show_progress=True,
            ):
                if result["status"] == "completed" and result["html"]:
                    chunks = self._process_html(result["html"], result["url"])
                    all_results.extend(chunks)
                else:
                    console.log(
                        f"[red]Failed to scrape {result['url']}: {result['status']}[/]"
                    )
        else:
            for i, html in enumerate(sources):
                url_meta = source_url if len(sources) == 1 else None
                chunks = self._process_html(html, url_meta)
                all_results.extend(chunks)

        return all_results

    async def aprocess(
        self,
        source: Union[str, List[str]],
        source_url: Optional[str] = None,
    ) -> list[ChunkWithMetadata]:
        """Async wrapper supporting URLs and raw HTML."""
        sources, is_url = self._normalize_input(source)
        all_results: list[ChunkWithMetadata] = []

        if is_url:
            console.log(f"[cyan]Async fetching {len(sources)} URL(s)…[/]")
            async for result in scrape_urls(
                urls=sources,
                headless=self.headless,
                use_cache=self.use_cache,
                scroll_strategy=self.scroll_strategy,
                show_progress=True,
            ):
                if result["status"] == "completed" and result["html"]:
                    chunks = await asyncio.to_thread(
                        self._process_html, result["html"], result["url"]
                    )
                    all_results.extend(chunks)
                else:
                    console.log(
                        f"[red]Failed to scrape {result['url']}: {result['status']}[/]"
                    )
        else:
            for html in sources:
                chunks = await asyncio.to_thread(self._process_html, html, source_url)
                all_results.extend(chunks)

        return all_results


def chunk(
    source: Union[str, List[str]],
    *,
    chunk_size: int = 512,
    tokenizer: Union[str, TokenizerProtocol] = None,
    min_chars_per_chunk: int = 50,
    semantic_model: str = "minishlab/potion-base-32M",
    table_rows_per_chunk: int = 5,
    headless: bool = True,
    use_cache: bool = False,
    scroll_strategy: str = "until_stable",
    min_section_tokens: int = 50,
    llm_model: Optional[str] = None,
) -> list[ChunkWithMetadata]:
    """
    Reusable standalone HTML-aware chunker.

    Args:
        source: HTML string, URL, local file path, or list of any combination.
            File paths are validated for existence before reading.
        chunk_size: Target token count per chunk.
        tokenizer: Tokenizer instance or name. Defaults to LlamaCppTokenizer.
        min_chars_per_chunk: Minimum character threshold for prose chunks.
        semantic_model: Model name for semantic chunking fallback.
        table_rows_per_chunk: Rows per chunk for table content.
        headless: Run browser in headless mode for URL scraping.
        use_cache: Enable Playwright response caching.
        scroll_strategy: Page scroll strategy for dynamic content.
        min_section_tokens: Threshold for merging short hierarchical sections.
        llm_model: Optional LLM model key for code language detection.

    Returns:
        List of ChunkWithMetadata objects with source provenance.
    """
    sources = source if isinstance(source, list) else [source]
    if not sources:
        console.log("[yellow]⚠ Empty source provided to chunk()[/]")
        return []

    pipeline = ScrapedHTMLPipeline(
        chunk_size=chunk_size,
        tokenizer=tokenizer,
        min_chars_per_chunk=min_chars_per_chunk,
        semantic_model=semantic_model,
        table_rows_per_chunk=table_rows_per_chunk,
        headless=headless,
        use_cache=use_cache,
        scroll_strategy=scroll_strategy,
        min_section_tokens=min_section_tokens,
        llm_model=llm_model,
    )

    all_results: list[ChunkWithMetadata] = []
    urls_to_scrape: list[str] = []
    html_inputs: list[tuple[str, Optional[str]]] = []

    for item in sources:
        if not item or not str(item).strip():
            console.log("[yellow]⚠ Skipping empty source item[/]")
            continue

        item_str = str(item).strip()

        if item_str.startswith(("http://", "https://")):
            urls_to_scrape.append(item_str)
            console.log(f"[cyan]🔗 Queued URL: {item_str}[/]")
            continue

        path = Path(item_str)
        if path.exists() and path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
                html_inputs.append((content, str(path.resolve())))
                console.log(
                    f"[green]📄 Read local file: {path} ({len(content):,} chars)[/]"
                )
            except Exception as e:
                console.log(f"[red]✖ Failed to read file '{path}': {e}[/]")
            continue

        html_inputs.append((item_str, None))
        console.log(f"[dim]📝 Treating input as raw HTML ({len(item_str):,} chars)[/]")

    if urls_to_scrape:
        console.log(f"[cyan]Fetching {len(urls_to_scrape)} URL(s)…[/]")
        for result in scrape_urls_sync(
            urls=urls_to_scrape,
            headless=headless,
            use_cache=use_cache,
            scroll_strategy=scroll_strategy,
            show_progress=True,
        ):
            if result["status"] == "completed" and result["html"]:
                chunks = pipeline._process_html(result["html"], result["url"])
                all_results.extend(chunks)
            else:
                console.log(
                    f"[red]✖ Failed to scrape {result['url']}: {result['status']}[/]"
                )

    for html_content, source_label in html_inputs:
        chunks = pipeline._process_html(html_content, source_label)
        all_results.extend(chunks)

    console.log(
        f"[bold green]✔ chunk() complete:[/] {len(all_results)} total chunks "
        f"from {len(sources)} source(s)"
    )
    return all_results


if __name__ == "__main__":
    from jet.adapters.chonkie.main._main_html_chunkers import main

    main()
