"""ReAct tool implementations: SearXNG search, URL reading, synthesis.
Tool wrappers accept an optional _step_tracker list and _memory instance.
When provided by ReactEngine, each tool call:
1. Appends an AgentStep to step_tracker for accurate step counting
2. Updates AccumulationMemory with full observations and token counts
✅ IMPROVEMENTS:
- synthesize() uses PromptBudget for safe prompt assembly
- read_url() applies SmartChunker + hybrid_search for structured content
- All tools propagate source metadata into AgentStep for citation
- ✅ NEW: Memory integration for real-time context accumulation
- ✅ NEW: Token counting via count_tokens() for accurate budget tracking
- ✅ NEW: List-intent guardrail via _query_intent parameter
- ✅ NEW: Reuses jet.search.searxng for caching, retries, and filtering
- ✅ NEW: Token-aware snippet truncation replaces MAX_SNIPPET_CHARS
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import TYPE_CHECKING, Any

from jet.adapters.llama_cpp.budget_utils import PromptBudget
from jet.adapters.llama_cpp.config import EMBED_MODEL_LG, LLM_MODEL, RERANK_MODEL
from jet.adapters.llama_cpp.llm_utils import achat
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.search.searxng import search_searxng as jet_search_searxng
from openai import AsyncOpenAI

from .types import AgentStep, QueryIntent, SearchResult
from .url_extractor import UrlContextExtractor

if TYPE_CHECKING:
    from .memory import AccumulationMemory

logger = logging.getLogger(__name__)

# ✅ CHANGED: Token-based limit replaces MAX_SNIPPET_CHARS = 500
MAX_SNIPPET_TOKENS = 150
_url_extractor = UrlContextExtractor(model=LLM_MODEL)


def truncate_to_tokens(
    text: str, max_tokens: int, model: str, suffix: str = "..."
) -> str:
    """Truncate text to fit within max_tokens using binary search.

    Avoids repeated full-tokenization by narrowing the search space.
    Returns original text if already within budget.
    Safe for use across tools and engine modules.
    """
    if not text:
        return text

    full_count = count_tokens(text, model=model)
    if full_count <= max_tokens:
        return text

    # Binary search for the longest prefix that fits
    lo, hi = 0, len(text)
    best = ""
    iterations = 0
    max_iterations = 30  # log2(len(text)) is typically < 20

    while lo <= hi and iterations < max_iterations:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        tokens = count_tokens(candidate, model=model)

        if tokens <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
        iterations += 1

    if best and suffix:
        suffix_tokens = count_tokens(suffix, model=model)
        if suffix_tokens < max_tokens:
            combined_budget = max_tokens - suffix_tokens
            if count_tokens(best, model=model) > combined_budget:
                while best and count_tokens(best + suffix, model=model) > max_tokens:
                    best = best[:-1]
            return best + suffix

    return best


async def searxng_search(
    query: str,
    categories: str = "general",
    num_results: int = 5,
    time_range: str | None = None,
    engines: list[str] | None = None,
    include_sites: list[str] | None = None,
    exclude_sites: list[str] | None = None,
    _step_tracker: list[AgentStep] | None = None,
    _query_intent: QueryIntent = QueryIntent.UNKNOWN,
    _memory: "AccumulationMemory | None" = None,
) -> str:
    """Search via SearXNG using jet.search.searxng and return formatted results.

    ✅ REUSES: jet.search.searxng for Redis caching, automatic retries,
    deduplication, relevance filtering, and site filtering.
    ✅ ASYNC SAFE: Wraps synchronous search in asyncio.to_thread.
    ✅ TOKEN-AWARE: Snippets truncated by token count, not char count.
    """
    logger.info("🔎 SearXNG search: %r (categories=%s)", query[:60], categories)

    kwargs: dict[str, Any] = {
        "count": num_results,
        "categories": [categories] if isinstance(categories, str) else categories,
        "use_cache": True,
        "max_retries": 3,
    }
    if engines:
        kwargs["engines"] = engines
    if include_sites:
        kwargs["include_sites"] = include_sites
    if exclude_sites:
        kwargs["exclude_sites"] = exclude_sites

    effective_query = query
    if time_range:
        effective_query = f"{query} {time_range}"
        logger.debug("⏳ Appended time_range '%s' to query", time_range)

    try:
        raw_results = await asyncio.to_thread(
            jet_search_searxng,
            query=effective_query,
            **kwargs,
        )
    except Exception as e:
        logger.error("❌ SearXNG search failed: %s", e)
        observation = (
            f"Search failed: {e}. Try a different query or check SearXNG status."
        )
        if _step_tracker is not None:
            _step_tracker.append(
                AgentStep(
                    thought="",
                    action="searxng_search",
                    action_input={"query": query, "categories": categories},
                    observation=observation,
                )
            )
        return observation

    if not raw_results:
        logger.warning("⚠️ No results for query: %r", query[:60])
        observation = "No search results found. Try rephrasing the query."
        if _step_tracker is not None:
            _step_tracker.append(
                AgentStep(
                    thought="",
                    action="searxng_search",
                    action_input={"query": query, "categories": categories},
                    observation=observation,
                )
            )
        if _memory is not None:
            _memory.record_search(query=query, observation=observation)
        return observation

    # ✅ CHANGED: Token-aware snippet truncation instead of char slicing
    results: list[SearchResult] = []
    for r in raw_results:
        raw_snippet = r.get("content", "") or ""
        truncated_snippet = truncate_to_tokens(
            raw_snippet, MAX_SNIPPET_TOKENS, model=LLM_MODEL
        )
        results.append(
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=truncated_snippet,
                engine=r.get("engine", "unknown"),
                score=float(r.get("score", 0)),
            )
        )

    lines = [f"Found {len(results)} results for '{query}':\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    URL: {r.url}")
        lines.append(f"    Snippet: {r.snippet}")
        lines.append("")
    observation = "\n".join(lines)

    logger.info(
        "✅ SearXNG returned %d results (snippets token-bounded to %d tokens)",
        len(results),
        MAX_SNIPPET_TOKENS,
    )

    if _memory is not None:
        _memory.record_search(query=query, results=results, observation=observation)

    if _step_tracker is not None:
        _step_tracker.append(
            AgentStep(
                thought="",
                action="searxng_search",
                action_input={
                    "query": query,
                    "categories": categories,
                    "num_results": num_results,
                    "time_range": time_range,
                },
                observation=observation,
            )
        )
    return observation


async def read_url(
    url: str,
    query: str | None = None,
    model: str = LLM_MODEL,
    embed_model: str = EMBED_MODEL_LG,
    rerank_model: str = RERANK_MODEL,
    _step_tracker: list[AgentStep] | None = None,
    _memory: "AccumulationMemory | None" = None,
) -> str:
    """Fetch a URL and return structured, query-relevant content.

    ✅ IMPROVEMENT: When 'query' is provided, uses hybrid_search internally
    to extract only the most relevant sections from the page rather than
    returning generic top chunks.
    ✅ NEW: Updates AccumulationMemory with content and accurate token count.
    """
    if model != _url_extractor.model:
        _url_extractor.model = model
    if embed_model is not None:
        _url_extractor.embed_model = embed_model
    if rerank_model is not None:
        _url_extractor.rerank_model = rerank_model

    content, error = await _url_extractor.extract(url, query=query)

    if error:
        observation = error
        logger.warning("⚠️ read_url failed: %s", error[:100])
    else:
        observation = f"Relevant content from {url}:\n{content}"
        logger.info(
            "✅ read_url success: %d chars extracted for query=%r",
            len(content),
            query[:60] if query else None,
        )

    if _memory is not None and not error:
        content_tokens = count_tokens(content, model=model)
        _memory.record_read(
            url=url,
            content=content,
            tokens=content_tokens,
        )

    if _step_tracker is not None:
        _step_tracker.append(
            AgentStep(
                thought="",
                action="read_url",
                action_input={"url": url, "query": query},
                observation=observation,
                source_url=url,
            )
        )
    return observation


async def synthesize(
    findings: str,
    original_query: str,
    model: str = LLM_MODEL,
    session_id: str | None = None,
    client: AsyncOpenAI | None = None,
    _step_tracker: list[AgentStep] | None = None,
) -> str:
    """Synthesize multiple search findings into a coherent answer.

    ✅ IMPROVEMENT: Uses PromptBudget to guarantee safe prompt assembly,
    preventing silent truncation and HTTP 400 errors from context overflow.
    Dynamically allocates remaining budget to completion tokens.
    """
    logger.info("🧩 Synthesizing findings for query: %r", original_query[:60])

    system_prompt = (
        "Synthesize research findings into a comprehensive, accurate answer. "
        "Use ONLY information from the provided findings. Cite sources by number. "
        "If findings are insufficient, say so explicitly. Do not fabricate."
    )

    budget = PromptBudget(model, max_completion_tokens=1024)
    safe_findings = budget.validate(
        system_prompt=system_prompt,
        query=original_query,
        chunks=[findings],
    )
    alloc = budget.get_allocation(
        system_prompt=system_prompt,
        query=original_query,
        chunks=safe_findings,
    )

    logger.info(
        "💰 Budget allocation: %d/%d tokens used, %d chunks included, %d truncated",
        alloc.total_used,
        alloc.model_ctx,
        alloc.chunks_included,
        alloc.chunks_truncated,
    )

    safe_text = safe_findings[0] if safe_findings else "No findings available."

    messages = [
        {
            "role": "user",
            "content": (
                f"Original Question: {original_query}\nResearch Findings:\n{safe_text}"
            ),
        },
    ]

    safe_completion_tokens = max(256, alloc.model_ctx - alloc.total_used)

    result = await achat(
        prompt_or_messages=messages,
        model=model,
        project_name="react-synthesize",
        temperature=0.3,
        max_tokens=safe_completion_tokens,
        enable_thinking=False,
        capture_content=True,
        session_id=session_id,
        client=client,
    )

    logger.info("✅ Synthesis complete: %d chars", len(result.content))

    if _step_tracker is not None:
        _step_tracker.append(
            AgentStep(
                thought="",
                action="synthesize",
                action_input={"original_query": original_query},
                observation=result.content,
            )
        )
    return result.content


def get_tool_definitions() -> list[dict]:
    """Return OpenAI-format tool definitions for the ReAct agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "searxng_search",
                "description": (
                    "Search the web using SearXNG meta-search engine with built-in caching, "
                    "filtering, and retry logic. Returns titles, URLs, and snippets. "
                    "Supports category filtering, engine selection, and site inclusion/exclusion."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query string",
                        },
                        "categories": {
                            "type": "string",
                            "description": "Search category: general, news, science, it, etc.",
                            "default": "general",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (default 5)",
                            "default": 5,
                        },
                        "time_range": {
                            "type": "string",
                            "description": "Time filter appended to query: day, month, year",
                            "enum": ["day", "month", "year"],
                        },
                        "engines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific search engines to use (e.g., ['google', 'bing'])",
                        },
                        "include_sites": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Limit search to these domains (e.g., ['wikipedia.org'])",
                        },
                        "exclude_sites": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exclude these domains from results",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_url",
                "description": (
                    "Fetch and read the full text content of a web page. "
                    "IMPORTANT: Snippets from search are often incomplete. "
                    "Use this tool to get detailed, verified information. "
                    "Accepts an optional 'query' parameter to extract only "
                    "the most relevant sections from long pages."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL to fetch and read",
                        },
                        "query": {
                            "type": "string",
                            "description": (
                                "Optional: The specific question or topic to focus extraction on. "
                                "Highly recommended for long articles."
                            ),
                        },
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "synthesize",
                "description": (
                    "Synthesize collected research findings into a final comprehensive answer. "
                    "Call this ONLY when you have gathered sufficient information from "
                    "multiple searches and/or page reads to fully answer the original question. "
                    "Pass all accumulated findings as the 'findings' parameter."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "findings": {
                            "type": "string",
                            "description": "All accumulated research findings to synthesize",
                        },
                        "original_query": {
                            "type": "string",
                            "description": "The original user question being answered",
                        },
                    },
                    "required": ["findings", "original_query"],
                },
            },
        },
    ]


def get_tool_registry(
    step_tracker: list[AgentStep] | None = None,
    embed_model: str = EMBED_MODEL_LG,
    rerank_model: str = RERANK_MODEL,
    query_intent: QueryIntent = QueryIntent.UNKNOWN,
    memory: "AccumulationMemory | None" = None,
) -> dict:
    """Return callable tool registry for llm_utils.achat agentic loop.

    Args:
        step_tracker: Optional mutable list that tool wrappers append to.
            Pass this from ReactEngine to get accurate step counts.
        embed_model: Embedding model for read_url hybrid search.
        rerank_model: Rerank model for read_url hybrid search.
        query_intent: Query intent for list-intent guardrail enforcement.
        memory: ✅ NEW: AccumulationMemory instance for real-time context updates.
    """
    registry = {
        "searxng_search": functools.partial(
            searxng_search,
            _step_tracker=step_tracker,
            _query_intent=query_intent,
            _memory=memory,
        ),
        "read_url": functools.partial(
            read_url,
            _step_tracker=step_tracker,
            embed_model=embed_model,
            rerank_model=rerank_model,
            _memory=memory,
        ),
        "synthesize": functools.partial(synthesize, _step_tracker=step_tracker),
    }
    return registry
