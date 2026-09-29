"""
Binary Strategy RAG Query Rewriting Pipeline.

Implements an adaptive query router that selects between two strategies:
1. REFINE: Clarifies ambiguous or vague queries into a single optimized query.
2. DECOMPOSE: Breaks complex, multi-part queries into independent sub-queries.

Usage Examples:
    # CLI Usage
    python -m jet.adapters.llama_cpp.rag.query_rewrite_binary "Compare AWS and Azure ML costs"
    python -m jet.adapters.llama_cpp.rag.query_rewrite_binary "How does it work?" --context "Previous topic was Kubernetes"

    # Programmatic Usage
    from jet.adapters.llama_cpp.rag.query_rewrite_binary import rewrite_query

    result = rewrite_query("What are the pros and cons of React vs Vue?")
    print(result.strategy)      # "DECOMPOSE"
    print(result.queries)       # ["React pros and cons", "Vue pros and cons", ...]

Expected Span Hierarchy:
    📦 binary-query-rewrite (CHAIN)
    │
    ├── 🧠 analyze_query_intent (LLM)
    │   ├── attr: llm.model_name = "qwen3.5-uncensored:2b"
    │   ├── attr: llm.input_messages = [...]
    │   └── attr: perf.analyze_query_intent.duration_ms = ...
    │
    ├── 🛠️ refine_query (TOOL) [Conditional]
    │   ├── attr: tool.name = "refine_query"
    │   └── attr: tool.parameters = {"original_query": "..."}
    │
    └── 🛠️ decompose_query (TOOL) [Conditional]
        ├── attr: tool.name = "decompose_query"
        └── attr: tool.parameters = {"original_query": "..."}
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Literal

from jet.adapters.llama_cpp.config import LLM_MODEL, PHOENIX_BASE_URL
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet_telemetry import chain, get_service_name, initialize_telemetry, llm, tool
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Structured Output Models
# ---------------------------------------------------------------------------


class RouterDecision(BaseModel):
    """Structured decision from the query analyzer."""

    strategy: Literal["NONE", "REFINE", "DECOMPOSE"] = Field(
        description="Selected rewriting strategy"
    )
    reason: str = Field(description="Brief explanation for the choice")


class RewriteResult(BaseModel):
    """Final output of the binary rewriting pipeline."""

    original_query: str
    strategy: Literal["NONE", "REFINE", "DECOMPOSE"]
    reason: str
    queries: list[str] = Field(description="List of queries to send to the retriever")


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """\
You are a query routing assistant for a RAG system. Analyze the user's query \
and select exactly ONE strategy:

1. NONE: The query is simple, clear, factual, and under 10 words. No rewriting needed.
   Example: "What is Python?"

2. REFINE: The query is ambiguous, vague, uses jargon, or lacks context. \
Rewrite it into a single, clearer, more specific query.
   Example: "How does it work?" → "Explain the working mechanism of Kubernetes control plane."

3. DECOMPOSE: The query is complex, has multiple parts, asks for comparisons, \
or requires multi-step reasoning. Break into 2-4 independent sub-queries.
   Example: "Compare AWS and Azure costs and performance." → \
["AWS pricing models", "Azure pricing models", "AWS vs Azure performance benchmarks"]

Return ONLY valid JSON matching this schema:
{"strategy": "NONE"|"REFINE"|"DECOMPOSE", "reason": "brief explanation"}
"""

REFINE_SYSTEM_PROMPT = """\
You are a search query optimizer. Rewrite the following user query into a \
single, clear, specific, and retrieval-friendly version. Preserve the original \
intent but improve clarity and add missing context.

Return ONLY the rewritten query text, nothing else.\
"""

DECOMPOSE_SYSTEM_PROMPT = """\
You are a query decomposition assistant. Break the following complex query \
into 2-4 simpler, independent sub-queries that can be searched separately. \
Each sub-query should be self-contained and target a distinct aspect of the \
original question.

Return ONLY a JSON array of strings, e.g.: ["sub-query 1", "sub-query 2"]\
"""


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


@llm(model_name=LLM_MODEL)
def _analyze_intent(
    query: str,
    context: str = "",
    output_dir: str | Path | None = None,
) -> RouterDecision:
    """Classify query complexity and select rewriting strategy."""
    user_content = f'Query: "{query}"'
    if context:
        user_content += f"\nConversation Context: {context}"

    result = chat(
        prompt_or_messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        model=LLM_MODEL,
        temperature=0.0,
        enable_thinking=False,
        response_format=RouterDecision,
        project_name=get_service_name(),
        output_dir=output_dir,
    )

    if result.structured and result.structured.success:
        return result.structured.parsed

    # Fallback: parse raw content
    try:
        data = json.loads(result.content)
        return RouterDecision(**data)
    except Exception:
        console.print("[yellow]⚠️ Router fallback: using NONE[/yellow]")
        return RouterDecision(strategy="NONE", reason="Parse failure fallback")


@tool(
    name="refine_query",
    description="Rewrite a vague query into a single optimized query",
)
def _refine_query(
    original_query: str,
    context: str = "",
    output_dir: str | Path | None = None,
) -> str:
    """Generate a single refined query via LLM."""
    user_content = f'Original query: "{original_query}"'
    if context:
        user_content += f"\nContext: {context}"

    result = chat(
        prompt_or_messages=[
            {"role": "system", "content": REFINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        model=LLM_MODEL,
        temperature=0.0,
        enable_thinking=False,
        project_name=get_service_name(),
        output_dir=output_dir,
    )
    return result.content.strip()


@tool(
    name="decompose_query",
    description="Break a complex query into independent sub-queries",
)
def _decompose_query(
    original_query: str,
    context: str = "",
    output_dir: str | Path | None = None,
) -> list[str]:
    """Generate multiple sub-queries via LLM."""
    user_content = f'Original query: "{original_query}"'
    if context:
        user_content += f"\nContext: {context}"

    result = chat(
        prompt_or_messages=[
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        model=LLM_MODEL,
        temperature=0.0,
        enable_thinking=False,
        project_name=get_service_name(),
        output_dir=output_dir,
    )

    try:
        parsed = json.loads(result.content)
        if isinstance(parsed, list):
            return [q.strip() for q in parsed if q.strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: split by newlines
    return [
        line.strip().lstrip("0123456789.-) ")
        for line in result.content.split("\n")
        if line.strip() and not line.strip().startswith(("Sub-queries", "Original"))
    ]


@chain(name="binary-query-rewrite")
def rewrite_query(
    query: str,
    context: str = "",
    output_dir: str | Path | None = None,
) -> RewriteResult:
    """
    Execute the binary strategy rewriting pipeline.

    Args:
        query: Original user query.
        context: Optional conversation history for contextual resolution.

    Returns:
        RewriteResult with selected strategy and generated queries.
    """
    # Step 1: Analyze intent
    decision = _analyze_intent(
        query,
        context,
        output_dir=output_dir,
    )
    console.print(
        f"[bold cyan]Strategy:[/bold cyan] {decision.strategy}  "
        f"[dim]({decision.reason})[/dim]"
    )

    # Step 2: Execute selected strategy
    if decision.strategy == "REFINE":
        refined = _refine_query(
            query,
            context,
            output_dir=output_dir,
        )
        queries = [refined]
    elif decision.strategy == "DECOMPOSE":
        queries = _decompose_query(
            query,
            context,
            output_dir=output_dir,
        )
    else:
        queries = [query]

    return RewriteResult(
        original_query=query,
        strategy=decision.strategy,
        reason=decision.reason,
        queries=queries,
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Binary Strategy RAG Query Rewriting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  %(prog)s "What is Python?"\n'
            '  %(prog)s "Compare AWS and Azure ML costs and performance"\n'
            '  %(prog)s "How does it work?" --context "Topic: Kubernetes"'
        ),
    )
    parser.add_argument("query", type=str, help="User query to rewrite")
    parser.add_argument(
        "-c",
        "--context",
        type=str,
        default="",
        help="Optional conversation context for follow-up queries",
    )
    parser.add_argument(
        "--phoenix-url",
        type=str,
        default=PHOENIX_BASE_URL,
        help="Phoenix observability endpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for trace exports",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    initialize_telemetry(
        service_name="binary-query-rewrite",
        endpoint=args.phoenix_url,
    )

    console.print(
        Panel(
            f"[bold]Query:[/bold] {args.query}\n"
            f"[dim]Context: {args.context or '(none)'}[/dim]",
            title="🔍 Binary Query Rewrite",
            border_style="blue",
        )
    )

    result = rewrite_query(
        query=args.query, context=args.context, output_dir=args.output_dir
    )

    # Display results table
    table = Table(
        title="Rewrite Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column("Query", style="green")
    for i, q in enumerate(result.queries, 1):
        table.add_row(str(i), q)

    console.print()
    console.print(table)
    console.print(f"\n[dim]Traces saved to: {args.output_dir}[/dim]")
