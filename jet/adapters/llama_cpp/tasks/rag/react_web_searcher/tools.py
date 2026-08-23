"""ReAct tool implementations: SearXNG search, URL reading, synthesis.

Tool wrappers accept an optional _step_tracker list. When provided by
ReactEngine, each tool call appends an AgentStep so the engine can
report accurate step counts. Without it, tools work standalone.

✅ IMPROVEMENTS:
- synthesize() uses PromptBudget for safe prompt assembly
- read_url() applies SmartChunker + format_chunks_for_rag for structured content
- All tools propagate source metadata into AgentStep for citation
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from jet.adapters.llama_cpp.budget_utils import PromptBudget
from jet.adapters.llama_cpp.llm_utils import achat
from openai import AsyncOpenAI

from .types import AgentStep, SearchResult
from .url_extractor import UrlContextExtractor

logger = logging.getLogger(__name__)

SEARXNG_BASE_URL = "http://localhost:8888"
SEARCH_TIMEOUT = 15.0
MAX_SNIPPET_CHARS = 500

_url_extractor = UrlContextExtractor(model="qwen3.5-uncensored:2b")


async def searxng_search(
    query: str,
    categories: str = "general",
    num_results: int = 5,
    time_range: str | None = None,
    _step_tracker: list[AgentStep] | None = None,
) -> str:
    """Search via SearXNG and return formatted results."""
    logger.info("🔎 SearXNG search: %r (categories=%s)", query[:60], categories)

    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "categories": categories,
        "pageno": 1,
    }
    if time_range:
        params["time_range"] = time_range

    try:
        async with httpx.AsyncClient(timeout=SEARCH_TIMEOUT) as client:
            resp = await client.get(f"{SEARXNG_BASE_URL}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error("❌ SearXNG HTTP error: %s", e)
        observation = (
            f"Search failed: HTTP {e.response.status_code}. Try a different query."
        )
        if _step_tracker is not None:
            _step_tracker.append(
                AgentStep(
                    thought="",
                    action="searxng_search",
                    action_input={"query": query, "categories": categories},
                    observation=observation[:200],
                )
            )
        return observation
    except Exception as e:
        logger.error("❌ SearXNG request failed: %s", e)
        observation = (
            f"Search failed: {e}. Check if SearXNG is running at {SEARXNG_BASE_URL}"
        )
        if _step_tracker is not None:
            _step_tracker.append(
                AgentStep(
                    thought="",
                    action="searxng_search",
                    action_input={"query": query, "categories": categories},
                    observation=observation[:200],
                )
            )
        return observation

    raw_results = data.get("results", [])[:num_results]

    if not raw_results:
        logger.warning("⚠️ No results for query: %r", query[:60])
        observation = "No search results found. Try rephrasing the query."
        if _step_tracker is not None:
            _step_tracker.append(
                AgentStep(
                    thought="",
                    action="searxng_search",
                    action_input={"query": query, "categories": categories},
                    observation=observation[:200],
                )
            )
        return observation

    results: list[SearchResult] = []
    for r in raw_results:
        results.append(
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=(r.get("content", "") or "")[:MAX_SNIPPET_CHARS],
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
    logger.info("✅ SearXNG returned %d results", len(results))

    if _step_tracker is not None:
        _step_tracker.append(
            AgentStep(
                thought="",
                action="searxng_search",
                action_input={
                    "query": query,
                    "categories": categories,
                    "num_results": num_results,
                },
                observation=observation[:200],
            )
        )
    return observation


async def read_url(
    url: str,
    query: str | None = None,  # ✅ NEW: Accept query for targeted extraction
    model: str = "qwen3.5-uncensored:2b",
    _step_tracker: list[AgentStep] | None = None,
) -> str:
    """Fetch a URL and return structured, query-relevant content.

    ✅ IMPROVEMENT: When 'query' is provided, uses hybrid_search internally
    to extract only the most relevant sections from the page rather than
    returning generic top chunks.
    """
    if model != _url_extractor.model:
        _url_extractor.model = model

    # ✅ Pass query to extractor for targeted retrieval
    content, error = await _url_extractor.extract(url, query=query)

    if error:
        observation = error
        logger.warning("⚠️ read_url failed: %s", error[:100])
    else:
        # Content is already chunked/formatted/re-ranked by extractor
        observation = f"Relevant content from {url}:\n{content}"
        logger.info(
            "✅ read_url success: %d chars extracted for query=%r",
            len(content),
            query[:60] if query else None,
        )

    if _step_tracker is not None:
        _step_tracker.append(
            AgentStep(
                thought="",
                action="read_url",
                action_input={"url": url, "query": query},
                observation=observation[:200],
                source_url=url,
            )
        )
    return observation


async def synthesize(
    findings: str,
    original_query: str,
    model: str = "qwen3.5-uncensored:2b",
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

    # ✅ Budget-aware synthesis: validate findings fit within context window
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
                f"Original Question: {original_query}\n\n"
                f"Research Findings:\n{safe_text}"
            ),
        },
    ]

    # ✅ Use remaining budget for completion instead of hardcoded 2048
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
                observation=result.content[:200],
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
                    "Search the web using SearXNG meta-search engine. "
                    "Returns titles, URLs, and snippets. Use this to find "
                    "information about any topic. Supports category filtering "
                    "(general, news, science, etc.) and time ranges (day, month, year)."
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
                            "description": "Time filter: day, month, year, or null for no filter",
                            "enum": ["day", "month", "year"],
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
                            "description": "Optional: The specific question or topic to focus extraction on. Highly recommended for long articles.",
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


def get_tool_registry(step_tracker: list[AgentStep] | None = None) -> dict:
    """Return callable tool registry for llm_utils.achat agentic loop.

    Args:
        step_tracker: Optional mutable list that tool wrappers append to.
            Pass this from ReactEngine to get accurate step counts.
    """
    import functools

    registry = {
        "searxng_search": functools.partial(searxng_search, _step_tracker=step_tracker),
        "read_url": functools.partial(read_url, _step_tracker=step_tracker),
        "synthesize": functools.partial(synthesize, _step_tracker=step_tracker),
    }
    return registry
