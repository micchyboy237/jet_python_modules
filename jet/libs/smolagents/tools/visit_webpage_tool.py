# visit_webpage_tool.py

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
from jet.adapters.llama_cpp.tokens import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.code.splitter_markdown_utils import get_md_header_contents
from jet.libs.smolagents.utils.debug_saver import DebugSaver
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
        max_output_length: int = 3800,  # now treated as **token** limit
        top_k: int = 7,
        chunk_target_tokens: int = 500,
        chunk_overlap_tokens: int = 100,
        verbose: bool = True,
        logs_dir: str | Path | None = None,
    ):
        super().__init__()
        self.embed_model = embed_model

        _max_model_context = LLAMACPP_MODEL_CONTEXTS.get(
            self.embed_model, max_output_length
        )
        # We now interpret max_output_length as tokens, not characters
        self.max_output_tokens = min(max_output_length, _max_model_context)

        self.top_k = top_k
        self.chunk_target_tokens = chunk_target_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        self.verbose = verbose

        self.debug_saver = DebugSaver(
            tool_name=self.name,
            base_dir=Path(logs_dir)
            if logs_dir
            else (
                Path(get_entry_file_dir())
                / "generated"
                / Path(get_entry_file_name()).stem
                / "visit_webpage_tool_logs"
            ),
            serializer=search_result_serializer,
        )

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    # ────────────────────────────────────────────────
    #  Helper: final safety trim by token count
    # ────────────────────────────────────────────────
    def _trim_to_token_limit(self, text: str) -> str:
        """Trim text so that count_tokens(text) ≤ self.max_output_tokens"""
        # Remove use of log, and always use embed_model as LLAMACPP_EMBED_KEYS (not string)
        token_count = count_tokens(text, model=self.embed_model)

        if token_count <= self.max_output_tokens:
            return text

        # Remove log (no logging output needed), just trim
        chars = list(text)
        low, high = 0, len(chars)

        while low < high:
            mid = (low + high + 1) // 2
            candidate = "".join(chars[:mid])
            if (
                count_tokens(candidate, model=self.embed_model)
                <= self.max_output_tokens
            ):
                low = mid
            else:
                high = mid - 1

        trimmed = "".join(chars[:low])
        return trimmed

    def forward(
        self,
        url: str,
        full_raw: bool = False,
        query: str | None = None,
    ) -> str:
        # Determine the effective search query
        def resolve_search_query(q: str | None) -> str:
            return (
                q.strip()
                if q is not None and isinstance(q, str) and q.strip()
                else "summary"
            )

        search_query = resolve_search_query(query)
        input_text = f"{url} {search_query}"
        input_tokens = count_tokens(input_text, model=self.embed_model)

        request_data = {
            "url": url,
            "full_raw": full_raw,
            "query": query,
            "resolved_search_query": search_query,
            "input_tokens": input_tokens,
        }
        self.debug_saver.save_json("request.json", request_data)
        logger.info("Saved request.json")

        with self.debug_saver.new_call(request_data) as call_dir:
            fetch_result = self._fetch_url(url)
            if not fetch_result.success:
                error_text = f"Failed to fetch page: {fetch_result.error_message}"
                logger.error(error_text)
                self.debug_saver.save("full_results.md", error_text)
                logger.info("Saved error message at full_results.md")
                return error_text

            self.debug_saver.save("page.html", fetch_result.html)
            logger.info("Saved page.html")

            headings = extract_markdown_section_texts(
                fetch_result.html, ignore_links=True
            )
            self.debug_saver.save_json("headings.json", headings)
            logger.info("Saved headings.json")

            if full_raw:
                result = self._process_full_raw(headings)
            else:
                result = self._process_smart_excerpts(headings, query, url=url)
            self.debug_saver.save("full_results.md", result)
            logger.info("Saved result at full_results.md")

            # ─── Final safety net: enforce token limit ──────────────────
            result = self._trim_to_token_limit(result)
            self.debug_saver.save("trimmed_results.md", result)
            logger.info("Saved result at trimmed_results.md")

            # ── New: save the final returned string with output_tokens ─────────
            self.debug_saver.save_json(
                "response.json",
                {
                    "result": result,
                    "output_tokens": count_tokens(result, model=self.embed_model),
                    "char_length": len(result),
                },
                indent=2,
            )
            logger.info("Saved final response.json")

            return result

    def _fetch_url(self, url: str) -> PageFetchResult:
        try:
            if self.verbose:
                logger.info(f"Fetching URL: {url}")
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
                logger.error(msg)
            return PageFetchResult(html="", success=False, error_message=msg)

    def _process_full_raw(self, md_texts: list[str]) -> str:
        # First pass: truncate individual sections
        truncated = truncate_texts(
            md_texts,
            model=self.embed_model,
            max_tokens=self.chunk_target_tokens * 2,  # be more generous per section
        )

        joined = "\n\n".join(truncated)

        # We still apply final trim later in forward()
        self.debug_saver.save("truncated_text.md", joined)
        return joined

    def _process_smart_excerpts(
        self, md_texts: list[str], query: str | None, url: str
    ) -> str:
        chunks = chunk_texts_with_data(
            texts=md_texts,
            chunk_size=self.chunk_target_tokens,
            chunk_overlap=self.chunk_overlap_tokens,
            strict_sentences=True,
            model=self.embed_model,
        )
        self.debug_saver.save_json("chunks.json", chunks)
        logger.info("Saved chunks.json")

        documents = [chunk["content"] for chunk in chunks]
        chunk_ids = [chunk["id"] for chunk in chunks]

        hybrid = HybridSearch.from_documents(
            documents=documents,
            ids=chunk_ids,
            model=self.embed_model,
            dense_weight=1.5,
            sparse_weight=0.7,
            category_config=RELATIVE_CATEGORY_CONFIG,
        )

        search_query = resolve_search_query(query)

        if self.verbose:
            logger.info(f"Using hybrid search with query: {search_query}")

        results = hybrid.search(
            search_query,
            top_k=self.top_k,
            dense_top_k=self.top_k * 4,
            sparse_top_k=self.top_k * 4,
            normalize_scores=True,
            debug=self.verbose,
        )

        self.debug_saver.save_json("search_results.json", results, indent=2)

        # ─── Optional improvement: dynamically choose how many excerpts fit ───
        header = build_result_header(url, search_query)
        header_tokens = count_tokens(header, model=self.embed_model)

        remaining_tokens = self.max_output_tokens - header_tokens - 200  # margin
        if remaining_tokens < 300:
            remaining_tokens = 300  # minimum useful content

        excerpts = []
        current_tokens = 0

        for i, r in enumerate(results, 1):
            preview = (
                r["text"].strip()[:400] + "..."
                if len(r["text"]) > 400
                else r["text"].strip()
            )
            line = (
                f"[{i}] Relevance {r['hybrid_score']:.3f}"
                f"{' norm=' + f'{r['normalized_hybrid']:.3f}' if 'normalized_hybrid' in r else ''}"
                f"{' ' + r['category'] + ' (lvl ' + str(r['category_level']) + ')' if 'category' in r else ''}\n"
                f"{preview}\n"
            )
            line_tokens = count_tokens(line, model=self.embed_model)

            if current_tokens + line_tokens > remaining_tokens:
                break

            excerpts.append(line)
            current_tokens += line_tokens

        if not excerpts and results:
            # at least return the best one if nothing fits
            excerpts = [build_excerpts(results[:1], 1)[0]]

        result = header + "\n".join(excerpts)
        self.debug_saver.save("searched_text.md", result)
        return result
