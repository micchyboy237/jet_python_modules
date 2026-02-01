# visit_webpage_tool.py

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import requests
from jet.adapters.llama_cpp.hybrid_search import (
    RELATIVE_CATEGORY_CONFIG,
    HybridSearch,
    HybridSearchResult,
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
    if isinstance(obj, dict) and "text" in obj:  # HybridSearchResult
        return {
            "text": obj.get("text", ""),
            "hybrid_score": obj.get("hybrid_score"),
            "category": obj.get("category"),
            "category_level": obj.get("category_level"),
        }
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
        self.save(
            filename,
            json.dumps(
                obj, **json_kwargs, ensure_ascii=False, default=search_result_serializer
            ),
        )


def extract_markdown_section_texts(html: str, ignore_links: bool = True) -> list[str]:
    """Extract content grouped by headers into markdown-formatted text blocks."""
    header_blocks = get_md_header_contents(html, ignore_links=ignore_links)
    return [
        f"{block['header']}\n\n{block['content']}".strip() for block in header_blocks
    ]


# ───────────────────────────────────────────────
# Pure helper functions
# ───────────────────────────────────────────────


def resolve_search_query(query: str | None) -> str:
    if query and (stripped := query.strip()):
        return stripped
    return "main content and key information from the webpage"


def build_excerpts(
    results: list[HybridSearchResult],
    max_count: int,
) -> list[str]:
    """Format top results into numbered, scored excerpts with category info."""
    lines = []
    for i, r in enumerate(results[:max_count], 1):
        preview = (
            r["text"].strip()[:200] + "..."
            if len(r["text"]) > 200
            else r["text"].strip()
        )
        norm = f" norm={r['normalized_hybrid']:.3f}" if "normalized_hybrid" in r else ""
        cat = f" {r['category']} (lvl {r['category_level']})" if "category" in r else ""
        lines.append(f"[{i}] Relevance {r['hybrid_score']:.3f}{norm}{cat}\n{preview}\n")
    return lines


def build_result_header(url: str, search_query: str) -> str:
    return (
        f"Most relevant excerpts from {url} "
        f"(hybrid BM25 + embedding retrieval, query: {search_query!r})\n\n"
    )


def format_final_result(header: str, excerpts: list[str]) -> str:
    return header + "\n".join(excerpts)


def create_search_documents(chunks: list[dict]) -> list[dict]:
    """Convert chunk format to documents expected by HybridSearch."""
    return [
        {"id": c.get("id", str(i)), "content": c["content"]}
        for i, c in enumerate(chunks)
    ]


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = """Visits a webpage and returns relevant content.

By default returns the most relevant excerpts (top 5–8 chunks) using hybrid (BM25 + embeddings) retrieval.
This is usually much better than dumping the full page.

If you need the complete raw content (code, tables, legal text, etc.),
pass "full_raw": true — but prefer focused follow-up calls instead."""

    inputs = {
        "url": {"type": "string", "description": "The url of the webpage to visit."},
        "full_raw": {
            "type": "boolean",
            "description": "If true, return full (truncated) raw markdown instead of smart excerpts.",
            "default": False,
            "nullable": True,
        },
        "query": {
            "type": "string",
            "description": "(optional) Specific question/topic to focus retrieval on. Usually auto-inferred.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        embed_model: LLAMACPP_EMBED_KEYS = "nomic-embed-text",
        max_output_length: int = 8192,
        default_top_k: int = 7,
        chunk_target_tokens: int = 500,
        chunk_overlap_tokens: int = 100,
        verbose: bool = True,
        logs_dir: str | Path | None = None,
    ):
        super().__init__()
        self.embed_model = embed_model

        _max_model_context = LLAMACPP_MODEL_CONTEXTS.get(self.embed_model, 8192)
        self.max_output_length = min(max_output_length, _max_model_context)

        self.default_top_k = default_top_k
        self.chunk_target_tokens = chunk_target_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        self.verbose = verbose

        # Setup DebugSaver
        self.debug_saver = DebugSaver(base_dir=Path(logs_dir) if logs_dir else None)

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
                result = self._process_smart_excerpts(md_texts, query, log, url)

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
        self.debug_saver.save("truncated_text.md", "\n\n".join(truncated))
        return "\n\n".join(truncated)

    def chunk_sections(self, md_texts: list[str]) -> list[dict]:
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

        documents = [chunk["content"] for chunk in chunks]
        # Optional: preserve ids if present
        chunk_ids = [chunk.get("id", str(i)) for i, chunk in enumerate(chunks)]

        hybrid = HybridSearch.from_documents(
            documents=documents,
            ids=chunk_ids,
            model=self.embed_model,
            # Tune these as needed — current defaults are good
            dense_weight=1.5,
            sparse_weight=0.7,
            # Use relative categories for robustness
            category_config=RELATIVE_CATEGORY_CONFIG,
        )

        search_query = resolve_search_query(query)

        if self.verbose:
            log(f"Using hybrid search with query: {search_query}")

        results = hybrid.search(
            search_query,
            top_k=self.default_top_k,
            dense_top_k=self.default_top_k
            * 4,  # old k_candidates ≈ 50 → ×4–5 oversampling
            sparse_top_k=self.default_top_k * 4,
            normalize_scores=True,  # add normalized_hybrid field
            debug=self.verbose,
        )

        self.debug_saver.save_json("search_results.json", results, indent=2)

        excerpts = build_excerpts(results, self.default_top_k)

        header = build_result_header(url, search_query)
        result = format_final_result(header, excerpts)

        self.debug_saver.save("searched_text.md", result)
        return result
