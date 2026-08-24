"""Simple ReAct Agentic Loop using llama.cpp via jet.adapters.

Implements a Reasoning + Acting loop where the LLM explicitly generates
thoughts and selects tools. Reuses existing search and scraping adapters.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import chat
from jet.logger import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

# ─── Tool Definitions (OpenAI Format) ────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information using SearXNG. Returns a list of results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5).",
                        "default": 5,
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
            "description": "Fetch and extract readable text content from a specific URL. Useful for getting details from a search result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to scrape.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]

# ─── Tool Executors ──────────────────────────────────────────────────────────


def _exec_web_search(query: str, count: int = 5) -> str:
    """Execute web search via SearXNG adapter."""
    try:
        from jet.search.searxng import search_searxng

        results = search_searxng(query=query, count=count, use_cache=True)
        if not results:
            return "No search results found."

        formatted = []
        for r in results:
            formatted.append(
                f"- **{r.get('title', 'Untitled')}**\n"
                f"  URL: {r.get('url')}\n"
                f"  Snippet: {r.get('content', '')[:300]}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        logger.error(f"web_search failed: {e}")
        return f"Error executing web_search: {e}"


def _exec_read_url(url: str) -> str:
    """Execute URL scraping via Playwright adapter (sync)."""
    try:
        from jet.scrapers.playwright_utils import scrape_urls_sync

        # Limit content length to avoid context overflow
        MAX_CHARS = 8000

        results = list(
            scrape_urls_sync(
                urls=[url],
                num_parallel=1,
                with_screenshot=False,
                scroll_strategy="none",
                headless=True,
                use_cache=True,
            )
        )

        if not results or results[0].get("status") != "completed":
            status = results[0].get("status", "unknown") if results else "no_result"
            return f"Failed to read URL (status: {status})."

        html = results[0].get("html", "")
        # Basic cleanup: return raw HTML truncated (a real impl would use readability/bs4)
        # For this simple agent, we rely on LLM to parse or truncate aggressively
        if len(html) > MAX_CHARS:
            html = html[:MAX_CHARS] + "\n... [TRUNCATED]"

        return html
    except Exception as e:
        logger.error(f"read_url failed: {e}")
        return f"Error executing read_url: {e}"


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "web_search": _exec_web_search,
    "read_url": _exec_read_url,
}

# ─── ReAct System Prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful AI assistant that uses tools to answer questions.
Follow the ReAct pattern:
1. Thought: Reason about what you need to do next.
2. Action: Call a tool if you need external information.
3. Observation: Analyze the tool output.
4. Repeat until you have enough information.
5. Final Answer: Provide a comprehensive response based on your findings.

Always cite sources when using search results. Be concise but thorough.
"""

# ─── Core Agent Loop ─────────────────────────────────────────────────────────


def run_react_agent(
    query: str,
    model: str = LLM_MODEL,
    max_iterations: int = 8,
    temperature: float = 0.1,
    verbose: bool = True,
) -> str:
    """Run a synchronous ReAct agentic loop.

    Args:
        query: User's question or task.
        model: LLM model identifier.
        max_iterations: Safety cap on tool-call rounds.
        temperature: Sampling temperature (low for reliable tool use).
        verbose: Print thoughts/actions to console.

    Returns:
        Final answer string from the assistant.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    if verbose:
        console.print(
            Panel(f"[bold cyan]Query:[/bold cyan] {query}", title="🤖 ReAct Agent")
        )

    for iteration in range(1, max_iterations + 1):
        logger.info(f"ReAct iteration {iteration}/{max_iterations}")

        # Call LLM (streaming disabled internally for simpler tool handling in loop)
        result = chat(
            prompt_or_messages=messages,
            model=model,
            tools=TOOLS,
            temperature=temperature,
            max_tokens=4096,
            project_name="simple-react-agent",
            capture_content=True,
        )

        # Append assistant message to history
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": result.content or "",
        }

        # Handle tool calls
        if result.has_tool_calls:
            # Note: OpenAI-compatible API expects tool_calls in the message
            # We reconstruct them for the message history
            tc_list = []
            for tc in result.tool_calls:
                tc_list.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                )
            assistant_msg["tool_calls"] = tc_list

            messages.append(assistant_msg)

            # Execute each tool call
            for tc in result.tool_calls:
                if verbose:
                    console.print(
                        f"\n[yellow]💭 Thought:[/yellow] {result.content or '(implicit)'}"
                    )
                    console.print(
                        f"[bold magenta]⚡ Action:[/bold magenta] {tc.name}({json.dumps(tc.arguments)})"
                    )

                executor = TOOL_REGISTRY.get(tc.name)
                if executor:
                    try:
                        observation = executor(**tc.arguments)
                    except Exception as e:
                        observation = f"Tool execution error: {e}"
                        logger.exception(f"Tool {tc.name} raised exception")
                else:
                    observation = f"Unknown tool: {tc.name}"

                if verbose:
                    obs_preview = (
                        observation[:300] + "..."
                        if len(observation) > 300
                        else observation
                    )
                    console.print(f"[dim]👁️ Observation:[/dim] {obs_preview}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": observation,
                    }
                )
        else:
            # No tool calls → final answer
            messages.append(assistant_msg)
            if verbose:
                console.print(f"\n[green]✅ Final Answer:[/green]")
                console.print(Markdown(result.content or ""))
            return result.content or ""

    # Max iterations reached
    fallback = (
        "I've reached the maximum number of reasoning steps without a final answer. Here's what I found so far:\n\n"
        + (result.content or "")
    )
    if verbose:
        console.print(f"\n[red]⚠️ Max iterations ({max_iterations}) reached.[/red]")
        console.print(Markdown(fallback))
    return fallback


# ─── CLI Entry Point ─────────────────────────────────────────────────────────


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple ReAct Agent powered by llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=str,
        help="The question or task for the agent.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help=f"LLM model to use (default: {LLM_MODEL})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum reasoning/tool-use iterations (default: 8)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature; keep low for reliable tool use (default: 0.1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress intermediate thought/action output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    answer = run_react_agent(
        query=args.query,
        model=args.model,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        verbose=not args.quiet,
    )
    # Exit code 0 regardless; answer is printed inside run_react_agent
    sys.exit(0)
