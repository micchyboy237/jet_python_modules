"""ReAct tool implementations: SearXNG search, URL reading, synthesis."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from jet.adapters.llama_cpp.chunking_utils import truncate_texts
from jet.adapters.llama_cpp.llm_utils import achat

from .types import SearchResult

logger = logging.getLogger(__name__)

SEARXNG_BASE_URL = "http://localhost:8888"
SEARCH_TIMEOUT = 15.0
READ_TIMEOUT = 10.0
MAX_SNIPPET_CHARS = 500
MAX_PAGE_CHARS = 3000


async def searxng_search(
    query: str,
    categories: str = "general",
    num_results: int = 5,
    time_range: str | None = None,
) -> str:
    """Search via SearXNG and return formatted results.

    Tool for the ReAct agent. Returns a string summary of search results
    that fits within the agent's observation window.
    """
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
        return f"Search failed: HTTP {e.response.status_code}. Try a different query."
    except Exception as e:
        logger.error("❌ SearXNG request failed: %s", e)
        return f"Search failed: {e}. Check if SearXNG is running at {SEARXNG_BASE_URL}"

    raw_results = data.get("results", [])[:num_results]
    if not raw_results:
        logger.warning("⚠️ No results for query: %r", query[:60])
        return "No search results found. Try rephrasing the query."

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

    # Format as readable text for the agent's observation
    lines = [f"Found {len(results)} results for '{query}':\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    URL: {r.url}")
        lines.append(f"    Snippet: {r.snippet}")
        lines.append("")

    observation = "\n".join(lines)
    logger.info("✅ SearXNG returned %d results", len(results))
    return observation


async def read_url(url: str, model: str = "qwen3.5-uncensored:2b") -> str:
    """Fetch a URL and return truncated text content.

    Tool for the ReAct agent. Fetches page content and truncates to fit
    within the agent's context budget.
    """
    logger.info("📄 Reading URL: %s", url[:80])

    try:
        async with httpx.AsyncClient(
            timeout=READ_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "JetReactSearcher/1.0"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"Cannot read non-text content type: {content_type}"
            text = resp.text
    except Exception as e:
        logger.error("❌ Failed to fetch %s: %s", url[:60], e)
        return f"Failed to fetch URL: {e}"

    # Strip HTML tags roughly (full parsing would need beautifulsoup)
    import re

    clean_text = re.sub(r"<[^>]+>", " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    if not clean_text:
        return "Page content is empty or could not be extracted."

    # Truncate to fit agent context
    truncated = truncate_texts(
        clean_text,
        model=model,
        max_tokens=2048,
        strict_sentences=True,
        show_progress=False,
    )
    if isinstance(truncated, list):
        truncated = truncated[0] if truncated else ""

    logger.info("✅ Read %d chars from %s", len(truncated), url[:60])
    return f"Content from {url}:\n\n{truncated}"


async def synthesize(
    findings: str,
    original_query: str,
    model: str = "qwen3.5-uncensored:2b",
) -> str:
    """Synthesize multiple search findings into a coherent answer.

    Tool for the ReAct agent. Called when enough information has been
    gathered to produce a final answer.
    """
    logger.info("🧩 Synthesizing findings for query: %r", original_query[:60])

    messages = [
        {
            "role": "user",
            "content": (
                f"Synthesize the following research findings into a comprehensive, "
                f"accurate answer to the original question.\n\n"
                f"Original Question: {original_query}\n\n"
                f"Research Findings:\n{findings}\n\n"
                f"Instructions:\n"
                f"- Use ONLY information from the findings above\n"
                f"- Cite sources by referencing the result numbers\n"
                f"- If findings are insufficient, say so explicitly\n"
                f"- Be concise but thorough\n"
                f"- Do not fabricate information"
            ),
        },
    ]

    result = await achat(
        prompt_or_messages=messages,
        model=model,
        project_name="react-synthesize",
        temperature=0.3,
        max_tokens=2048,
        enable_thinking=False,
        capture_content=True,
    )

    logger.info("✅ Synthesis complete: %d chars", len(result.content))
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
                    "Use this after finding a promising URL from search results "
                    "to get detailed information beyond the snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL to fetch and read",
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


def get_tool_registry() -> dict:
    """Return callable tool registry for llm_utils.achat agentic loop."""
    return {
        "searxng_search": searxng_search,
        "read_url": read_url,
        "synthesize": synthesize,
    }
