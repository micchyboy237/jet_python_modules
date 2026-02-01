# visit_webpage_tool.py

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import requests
from jet.adapters.llama_cpp.hybrid_search import (
    HybridConfig,
    HybridSearcher,
    SearchResult,
)
from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_CONTEXTS
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.libs.smolagents._logging import structured_tool_logger
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from jet.wordnet.text_chunker import chunk_texts_with_data, truncate_texts
from smolagents.tools import Tool

logger = logging.getLogger(__name__)


def search_result_serializer(obj):
    if isinstance(obj, SearchResult):
        return {"item": obj.item, "score": obj.score}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


@dataclass
class PageFetchResult:
    html: str
    success: bool = True
    error_message: str | None = None


class DebugSaver:
    """Handles all debug file writing – easy to mock/disable"""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or (
            Path(get_entry_file_dir())
            / "generated"
            / Path(get_entry_file_name()).stem
            / "visit_webpage_tool_logs"
        )

    def save(self, filename: str, content: str, encoding: str = "utf-8") -> None:
        if not self.base_dir:
            return
        path = self.base_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)

    def save_json(self, filename: str, obj, **json_kwargs):
        self.save(filename, json.dumps(obj, **json_kwargs, ensure_ascii=False))


def extract_markdown_section_texts(html: str, ignore_links: bool = True) -> list[str]:
    """Extract content grouped by headers into markdown-formatted text blocks.

    Each block starts with the header (ATX style) followed by its cleaned content.
    """
    header_blocks = get_md_header_contents(html, ignore_links=ignore_links)
    return [
        f"{block['header']}\n\n{block['content']}".strip() for block in header_blocks
    ]


# ───────────────────────────────────────────────
# Pure helper functions moved out of the class
# ───────────────────────────────────────────────


def resolve_search_query(query: str | None) -> str:
    """Determine final search query — use provided or fallback."""
    if query and (stripped := query.strip()):
        return stripped
    return "main content and key information from the webpage"


def build_excerpts(
    results: list[SearchResult],
    max_count: int,
) -> list[str]:
    """Format top results into numbered, scored excerpts."""
    return [
        f"[{i}] Relevance {r.score:.3f}\n{r.item['content'].strip()}\n"
        for i, r in enumerate(results[:max_count], 1)
    ]


def build_result_header(url: str, search_query: str) -> str:
    """Create descriptive header for smart-excerpt output."""
    return (
        f"Most relevant excerpts from {url} "
        f"(hybrid BM25 + embedding retrieval, query: {search_query!r})\n\n"
    )


def format_final_result(header: str, excerpts: list[str]) -> str:
    """Combine header and excerpts into final string."""
    return header + "\n".join(excerpts)


def create_search_documents(chunks: list[dict]) -> list[dict]:
    """Convert chunk format to documents expected by HybridSearcher."""
    return [{"id": c["id"], "content": c["content"]} for c in chunks]


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = """Visits a webpage and returns relevant content.

By default returns the most relevant excerpts (top 5–8 chunks) for the current task/question using hybrid (BM25 + embeddings) retrieval.
This usually gives much better context than raw full page content.

If you really need the complete raw page (code, full tables, legal text, etc.),
add "full_raw": true in the call — but prefer focused follow-up calls instead."""

    inputs = {
        "url": {"type": "string", "description": "The url of the webpage to visit."},
        "full_raw": {
            "type": "boolean",
            "description": "If true, return full (truncated) raw markdown instead of smart excerpts. Rarely needed.",
            "default": False,
            "nullable": True,
        },
        "query": {
            "type": "string",
            "description": "(optional) Specific question/topic to focus retrieval on. Usually auto-inferred from conversation.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        embed_model: LLAMACPP_EMBED_KEYS = "nomic-embed-text",
        max_output_length: int = 8192,
        default_k_final: int = 7,
        chunk_target_tokens: int = 500,
        chunk_overlap_tokens: int = 100,
        hybrid_config: HybridConfig | None = None,
        verbose: bool = True,
        logs_dir: str | Path | None = None,
    ):
        super().__init__()
        self.embed_model = embed_model

        _max_model_context = LLAMACPP_MODEL_CONTEXTS[self.embed_model]
        self.max_output_length = (
            max_output_length
            if _max_model_context > max_output_length
            else _max_model_context
        )
        self.default_k_final = default_k_final
        self.chunk_target_tokens = chunk_target_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        self.hybrid_config = hybrid_config or HybridConfig(
            k_candidates=50,
            k_final=default_k_final,
            rrf_constant=60.0,
            bm25_weight=1.12,
            vector_weight=1.0,
        )
        self.verbose = verbose
        # Setup DebugSaver for debug/dump file writing
        if logs_dir:
            logs_base = Path(logs_dir).resolve()
        else:
            logs_base = None
        self.debug_saver = DebugSaver(logs_base)

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def forward(
        self,
        url: str,
        full_raw: bool = False,
        query: str | None = None,
    ) -> str:
        with structured_tool_logger(
            self.debug_saver.base_dir,
            self.name,
            {"url": url, "full_raw": full_raw, "query": query},
            self.verbose,
        ) as (call_dir, log):
            # Update debug_saver for per-call directory
            self.debug_saver.base_dir = call_dir

            fetch_result = self._fetch_url(url, log)
            if not fetch_result.success:
                error_text = f"Failed to fetch page: {fetch_result.error_message}"
                self.debug_saver.save("full_results.md", error_text)
                return error_text

            self.debug_saver.save("page.html", fetch_result.html)

            md_texts = extract_markdown_section_texts(
                fetch_result.html, ignore_links=True
            )

            if full_raw:
                result = self._process_full_raw(md_texts)
            else:
                result = self._process_smart_excerpts(md_texts, query or None, log, url)

            self.debug_saver.save("full_results.md", result)

            if self.verbose:
                log(f"Returning {len(result)} characters")
                if call_dir:
                    log(f"Saved full markdown → {call_dir / 'full_results.md'}")

            return result

    def _fetch_url(self, url: str, log) -> PageFetchResult:
        try:
            if self.verbose:
                log(f"Fetching URL: {url}")
            resp = requests.get(
                url,
                timeout=18,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; VisitWebpageTool/1.0)"
                },
            )
            resp.raise_for_status()
            return PageFetchResult(html=resp.text)
        except Exception as e:
            msg = f"Fetch failed: {str(e)}"
            if self.verbose:
                log(msg)
            return PageFetchResult(html="", success=False, error_message=msg)

    def _process_full_raw(self, md_texts: list[str]) -> str:
        truncated = truncate_texts(
            md_texts, model=self.embed_model, max_tokens=self.max_output_length
        )
        self.debug_saver.save("truncated_text.md", truncated)
        return " ".join(truncated)

    def chunk_sections(self, md_texts: list[str]) -> list[dict]:
        """Split markdown sections into overlapping chunks."""
        return chunk_texts_with_data(
            texts=md_texts,
            chunk_size=self.chunk_target_tokens,
            chunk_overlap=self.chunk_overlap_tokens,
            strict_sentences=True,
            model=self.embed_model,
        )

    def _process_smart_excerpts(
        self, md_texts: list[str], query: str | None, log, url: str
    ) -> str:
        chunks = self.chunk_sections(md_texts)
        self.debug_saver.save_json("chunks.json", chunks, indent=2)

        docs = create_search_documents(chunks)

        searcher = HybridSearcher.from_documents(
            documents=docs,
            model=self.embed_model,
            **self.hybrid_config.__dict__,
        )

        search_query = resolve_search_query(query)

        if self.verbose:
            log(f"Using hybrid search with query: {search_query}")

        results = searcher.search(search_query)
        self.debug_saver.save_json(
            "search_results.json", results, default=search_result_serializer, indent=2
        )

        excerpts = build_excerpts(results, self.hybrid_config.k_final)

        header = build_result_header(url, search_query)
        result = format_final_result(header, excerpts)

        self.debug_saver.save("searched_text.md", result)
        return result
