"""Reusable crawl result processor with RAG context generation.

Processes streaming crawl results, computes relevance scores, and assembles
token-safe RAG context for LLM consumption. Memory-optimized to prevent
crawler safety-lock stalls.
"""

import asyncio
import gc
from typing import Any, List

from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.libs.crawl4ai_lib.config import LLM_MODEL, RAG_MAX_TOKENS
from jet.logger import logger


class CrawlResultProcessor:
    """Reusable class for processing crawl results with RAG support."""

    def __init__(self):
        self.results: List[dict[str, Any]] = []
        self._current_user_query: str = ""

    def set_current_query(self, query: str) -> None:
        """Set the current query so it can be used during processing."""
        self._current_user_query = query

    async def process_result(self, result: Any) -> None:
        """Async callback required by crawl_many.

        Memory-optimized: calculates score once, sorts periodically,
        avoids duplicate markdown storage, and triggers GC after each result.
        """
        try:
            url = getattr(result, "url", "Unknown")
            success = getattr(result, "success", False)
            user_query = self._current_user_query

            if success:
                title = (
                    result.metadata.get("title")
                    if getattr(result, "metadata", None)
                    else "N/A"
                )
                markdown_obj = getattr(result, "markdown", None)
                raw_len = len(getattr(markdown_obj, "raw_markdown", "") or "")
                fit_len = len(getattr(markdown_obj, "fit_markdown", "") or "")

                # Calculate score ONCE and reuse (was previously called 2×)
                score = (
                    calculate_relevance_score(result, user_query) if user_query else 0.0
                )

                print(f"✅ [SUCCESS] {url}")
                print(f" Title : {title}")
                print(f" Raw Markdown : {raw_len:,} chars")
                if fit_len:
                    print(f" Fit Markdown : {fit_len:,} chars (BM25 filtered)")
                print(f" Relevance Score : {score:.3f} / 1.0")
                print("-" * 90)
            else:
                error_msg = getattr(result, "error_message", "Unknown error")
                status = getattr(result, "status_code", None)
                print(f"❌ [FAILED] {url}")
                print(f" Error : {error_msg}")
                if status:
                    print(f" Status: {status}")
                print("-" * 90)

            data: dict[str, Any] = {
                "url": getattr(result, "url", None),
                "success": success,
                "timestamp": asyncio.get_event_loop().time(),
                "query": user_query,
            }

            if success:
                data["title"] = (
                    result.metadata.get("title")
                    if getattr(result, "metadata", None)
                    else None
                )
                data["status_code"] = getattr(result, "status_code", None)

                markdown_obj = getattr(result, "markdown", None)
                if hasattr(markdown_obj, "raw_markdown"):
                    data["raw_markdown"] = markdown_obj.raw_markdown
                    data["raw_markdown_length"] = len(markdown_obj.raw_markdown)

                fit_md = getattr(markdown_obj, "fit_markdown", None)
                if fit_md:
                    data["fit_markdown"] = fit_md
                    data["fit_markdown_length"] = len(fit_md)
                    # NOTE: No longer storing duplicate data["markdown"] = fit_md
                    # get_rag_context() resolves priority at read time
                else:
                    md_str = getattr(markdown_obj, "raw_markdown", None) or str(
                        markdown_obj or ""
                    )
                    data["markdown"] = md_str
                    data["markdown_length"] = len(md_str)

                # Reuse pre-calculated score (eliminates duplicate call)
                data["relevance_score"] = score

                extracted = getattr(result, "extracted_content", None)
                if extracted:
                    data["extracted_content"] = extracted
                    data["extracted_length"] = (
                        len(extracted) if isinstance(extracted, str) else 0
                    )
            else:
                data["error_message"] = getattr(
                    result, "error_message", "Unknown error"
                )
                data["status_code"] = getattr(result, "status_code", None)
                data["relevance_score"] = 0.0

            self.results.append(data)

            # Sort periodically instead of every insert (~80% less sorting)
            if len(self.results) % 5 == 0:
                self.results.sort(
                    key=lambda x: (
                        x.get("relevance_score", 0.0),
                        x.get("timestamp", 0),
                    ),
                    reverse=True,
                )

            # Explicit GC hint to release large markdown strings promptly
            del result
            gc.collect()

        except Exception as e:
            logger.warning(
                f"Error processing result for {getattr(result, 'url', 'Unknown')}: {e}"
            )
            self.results.append(
                {
                    "url": getattr(result, "url", None),
                    "success": False,
                    "error_message": str(e),
                    "relevance_score": 0.0,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

    def get_rag_context(
        self,
        max_tokens: int = RAG_MAX_TOKENS,
        model: str | None = None,
    ) -> str:
        """Generate token-safe RAG context from successful results.

        Assembles markdown sources sorted by relevance, truncating when
        the token budget is exhausted to prevent LLM context overflow.

        Args:
            max_tokens: Maximum tokens for assembled context.
            model: Model name for token counting. Uses LLM_MODEL default.
        """
        if not self.results:
            return ""

        if model is None:
            model = LLM_MODEL

        # Final sort before assembly (since we skip some sorts during streaming)
        self.results.sort(
            key=lambda x: (x.get("relevance_score", 0.0), x.get("timestamp", 0)),
            reverse=True,
        )

        parts: List[str] = []
        total_tokens = 0

        for item in self.results:
            if not item.get("success", False):
                continue

            url = item.get("url", "")
            title = item.get("title", "Untitled")

            # Resolve markdown priority without duplicate storage
            markdown = (
                item.get("fit_markdown")
                or item.get("markdown")
                or item.get("raw_markdown", "")
            ).strip()

            if not markdown or len(markdown) < 50:
                continue

            section = f"Source: {title}\nURL: {url}\n\n{markdown}\n\n---\n\n"

            # Token-aware truncation
            section_tokens = count_tokens(section, model=model)
            if total_tokens + section_tokens > max_tokens:
                remaining = max_tokens - total_tokens
                if remaining < 100:
                    break
                char_ratio = len(section) / max(section_tokens, 1)
                truncated = section[: int(remaining * char_ratio)]
                parts.append(
                    truncated + "\n\n[... truncated due to token limit ...]\n\n"
                )
                logger.info(
                    f"get_rag_context: truncated at {max_tokens} tokens "
                    f"({len(parts)} sources included)"
                )
                break

            parts.append(section)
            total_tokens += section_tokens

        logger.info(
            f"get_rag_context: assembled {total_tokens} tokens "
            f"from {len(parts)} sources"
        )
        return "".join(parts).strip()

    def get_results(self) -> List[dict[str, Any]]:
        return self.results.copy()

    def clear(self) -> None:
        self.results.clear()


def calculate_relevance_score(result: Any, user_query: str) -> float:
    """Calculate relevance score based on content survival and keyword density.

    Combines BM25-filtered content ratio (65%) with keyword match density (35%).
    """
    if not getattr(result, "success", False):
        return 0.0

    markdown_obj = getattr(result, "markdown", None)
    if not markdown_obj:
        return 0.0

    fit_md = getattr(markdown_obj, "fit_markdown", None)
    raw_md = getattr(markdown_obj, "raw_markdown", None) or str(markdown_obj)
    content = fit_md or raw_md

    if not content or len(content) < 50:
        return 0.0

    ratio = len(content) / max(len(raw_md), 1)
    survival_score = min(1.0, ratio * 1.8)

    query_terms = [term.lower() for term in user_query.split() if len(term) > 2]
    if not query_terms:
        return round(survival_score, 3)

    text_lower = content.lower()
    hits = sum(text_lower.count(term) for term in query_terms)
    density = hits / max(len(content.split()), 1)
    keyword_bonus = min(0.6, density * 8)

    final_score = (survival_score * 0.65) + (keyword_bonus * 0.35)
    return round(min(1.0, final_score), 3)
