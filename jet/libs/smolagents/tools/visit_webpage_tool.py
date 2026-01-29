# visit_webpage_tool.py
import json
import logging
import re
from pathlib import Path

import requests
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.hybrid_search import (
    HybridConfig,
    HybridSearcher,
)
from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_CONTEXTS
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.libs.smolagents._logging import structured_tool_logger
from jet.wordnet.text_chunker import chunk_texts_with_data, truncate_texts
from markdownify import markdownify
from smolagents.tools import Tool

logger = logging.getLogger(__name__)


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
        verbose: bool = False,
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

        self.embedder = LlamacppEmbedding(
            model=self.embed_model,
            use_cache=True,
            verbose=False,
        )

        self.hybrid_config = hybrid_config or HybridConfig(
            k_candidates=50,
            k_final=default_k_final,
            rrf_constant=60.0,
            bm25_weight=1.12,
            vector_weight=1.0,
        )
        self.verbose = verbose
        self.logs_dir = Path(logs_dir) if logs_dir else None

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def _simple_chunk(self, text: str) -> list[str]:
        """Very naive but fast chunker — improve later if needed"""
        if len(text) < 1800:
            return [text]

        chunks = []
        pos = 0
        target_chars = (
            self.chunk_target_tokens * 4
        )  # very rough heuristic (avg ~4 chars/token)

        while pos < len(text):
            end = pos + target_chars
            if end >= len(text):
                chunks.append(text[pos:].strip())
                break

            # Try to break at paragraph or sentence
            while end < len(text) and text[end] not in ".\n!?":
                end += 1
            if end < len(text):
                end += 1  # include boundary

            chunk = text[pos:end].strip()
            if chunk:
                chunks.append(chunk)

            # Overlap
            pos = end - (self.chunk_overlap_tokens * 4)

        return [c for c in chunks if len(c.strip()) > 60]

    def forward(
        self,
        url: str,
        full_raw: bool = False,
        query: str | None = None,
    ) -> str:
        request_data = {
            "url": url,
            "full_raw": full_raw,
            "query": query,
            "max_output_length": self.max_output_length,
        }

        with structured_tool_logger(
            self.logs_dir, self.name, request_data, self.verbose
        ) as (call_dir, log):
            if self.verbose:
                log(f"Fetching URL: {url}")
                if query:
                    log(f"Focused query: {query}")

            try:
                response = requests.get(
                    url,
                    timeout=18,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; VisitWebpageTool/1.0)"
                    },
                )
                response.raise_for_status()
            except Exception as e:
                if self.verbose:
                    log(f"Fetch failed: {str(e)}")
                if call_dir:
                    (call_dir / "full_results.md").write_text(
                        f"Fetch error: {str(e)}", encoding="utf-8"
                    )
                raise

            md = markdownify(response.text, heading_style="ATX").strip()
            md = re.sub(r"\n{3,}", "\n\n", md)

            if full_raw:
                result = truncate_texts(
                    md, model=self.embed_model, max_tokens=self.max_output_length
                )
            else:
                # chunks = self._simple_chunk(md)
                chunks = chunk_texts_with_data(
                    texts=[md],
                    chunk_size=self.chunk_target_tokens,
                    chunk_overlap=self.chunk_overlap_tokens,
                    strict_sentences=False,
                    model=self.embed_model,
                    # model=None,  # Using default word-based tokenization
                )

                try:
                    docs = [
                        {"id": c["id"], "content": c["content"]}
                        for i, c in enumerate(chunks)
                    ]
                    searcher = HybridSearcher.from_documents(
                        documents=docs,
                        model=self.embed_model,
                        **self.hybrid_config.__dict__,
                    )
                    search_query = (
                        query.strip()
                        if query and query.strip()
                        else "main content and key information from the webpage"
                    )
                    if self.verbose:
                        log(f"Using hybrid search with query: {search_query}")
                    results = searcher.search(search_query)
                    excerpts = [
                        f"[{i}] Relevance {r.score:.3f}\n{r.item['content'].strip()}\n"
                        for i, r in enumerate(results[: self.hybrid_config.k_final], 1)
                    ]
                    header = (
                        f"Most relevant excerpts from {url} "
                        f"(hybrid BM25 + embedding retrieval, query: {search_query!r})\n\n"
                    )
                    result = header + "\n".join(excerpts)

                    # if len(chunks) > 12:
                    #     result += (
                    #         "\n\n(Page had many sections. If you need information about a specific part "
                    #         "(table, section title, code block…), ask a more focused follow-up question "
                    #         "or call visit_webpage again with a precise query parameter. "
                    #         "Use full_raw=true only if you really need the complete raw markdown."
                    #     )
                    # result = self._truncate(result, self.max_output_length)
                except Exception as e:
                    if self.verbose:
                        log(f"Hybrid retrieval failed: {str(e)}")

                    raise
                    # result = (
                    #     self._truncate(md, min(self.max_output_length, 4800))
                    #     + "\n\n(Note: smart retrieval failed — showing truncated raw content)"
                    # )

            if call_dir:
                # Summary stats
                response_info = {
                    "result_length": len(result),
                    "full_raw": full_raw,
                    "chunk_count": len(chunks) if "chunks" in locals() else None,
                    "excerpt_count": len(excerpts) if "excerpts" in locals() else None,
                    "preview": result[:600] + "..." if len(result) > 600 else result,
                    "full_markdown_file": "full_results.md",
                }
                (call_dir / "response.json").write_text(
                    json.dumps(response_info, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                # Full content the agent receives
                (call_dir / "full_results.md").write_text(result, encoding="utf-8")

            if self.verbose:
                log(f"Returning {len(result)} characters")
                if call_dir:
                    log(f"Saved full markdown → {call_dir / 'full_results.md'}")

            return result
