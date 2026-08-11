# jet_python_modules/jet/adapters/llama_cpp/tasks/evaluate_multiple_contexts_relevance.py
"""Evaluate relevance of multiple contexts against a single query using grammar-constrained generation.

Uses GBNF grammar via extra_body_params to guarantee valid JSON array output
with per-context relevance scores. Each result includes the context index,
relevance score (0-2), and validity flag.

NOTE: enable_thinking is FORCED to False because thinking tokens break grammar constraints.
"""

import json
from typing import Optional, TypedDict

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import chat
from jet.logger import logger


class ContextRelevanceResult(TypedDict):
    """Single context relevance evaluation result."""

    context_index: int
    relevance_score: int
    is_valid: bool
    error: Optional[str]


SYSTEM_PROMPT = """\
You are a relevance evaluator. Given a query and multiple numbered contexts, \
evaluate each context's relevance to the query.

Scoring:
- 0: Low relevance (unrelated or barely related)
- 1: Medium relevance (partially addresses the query)
- 2: High relevance (directly and mostly addresses the query)

Return ONLY a JSON array with one object per context, in order. \
Each object must have "context_index" (integer) and "relevance_score" (0, 1, or 2). \
Do NOT include any other text."""


def _build_grammar(num_contexts: int) -> str:
    """Build a dynamic GBNF grammar for exactly num_contexts relevance results.

    Generates a JSON array schema where each element has context_index and
    relevance_score fields. Each context object is defined as a separate named
    rule to avoid GBNF parsing issues with inline concatenation.
    """
    if num_contexts <= 0:
        raise ValueError("num_contexts must be positive")

    lines: list[str] = []

    # Define individual item rules: item0 ::= { "context_index": 0, "relevance_score": score }
    item_refs: list[str] = []
    for i in range(num_contexts):
        rule_name = f"item{i}"
        lines.append(
            f'{rule_name} ::= "{{" ws "\\"context_index\\"" ws ":" ws {i} ws "," '
            f'ws "\\"relevance_score\\"" ws ":" ws score ws "}}"'
        )
        item_refs.append(rule_name)

    # Root rule references each item with explicit comma+ws separators as terminals
    items_sequence = ' ws "," ws '.join(item_refs)
    lines.insert(0, f'root ::= "[" ws {items_sequence} ws "]"')
    lines.append('score ::= "0" | "1" | "2"')
    lines.append("ws ::= [ \\t\\n]*")

    return "\n".join(lines)


def _build_user_prompt(query: str, contexts: list[str]) -> str:
    """Build user prompt with query and indexed contexts."""
    parts = [f"Query: {query}\n\nContexts:"]
    for idx, ctx in enumerate(contexts):
        parts.append(f"[{idx}] {ctx}")
    return "\n".join(parts)


def evaluate_multiple_contexts_relevance(
    query: str,
    contexts: list[str],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> list[ContextRelevanceResult]:
    """Evaluate relevance of multiple contexts against a single query.

    Uses grammar-constrained generation to produce a guaranteed-valid JSON array
    of per-context relevance scores.

    Args:
        query: The user query to evaluate against.
        contexts: List of context strings to assess.
        model: LLM model key. Defaults to LLM_MODEL.
        temperature: Sampling temperature (default: 0.0 for deterministic).
        max_tokens: Max tokens for the JSON response.

    Returns:
        List of ContextRelevanceResult dicts, one per input context.
        If generation fails entirely, returns a list of invalid results.
    """
    resolved_model = model or LLM_MODEL

    # Validate inputs
    if not query.strip():
        logger.error("Query cannot be empty")
        return [
            ContextRelevanceResult(
                context_index=i, relevance_score=0, is_valid=False, error="Empty query"
            )
            for i in range(len(contexts))
        ]
    if not contexts:
        logger.error("Contexts list cannot be empty")
        return []

    logger.info(
        f"evaluate_multiple_contexts_relevance: model={resolved_model}, "
        f"{len(contexts)} contexts, query='{query[:60]}...'"
    )

    # Build dynamic grammar and prompt
    try:
        grammar = _build_grammar(len(contexts))
    except ValueError as e:
        logger.error(f"Grammar build failed: {e}")
        return [
            ContextRelevanceResult(
                context_index=i, relevance_score=0, is_valid=False, error=str(e)
            )
            for i in range(len(contexts))
        ]

    user_prompt = _build_user_prompt(query, contexts)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    logger.debug(f"Grammar preview:\n{grammar[:300]}...")

    # Call with grammar constraint; enable_thinking MUST be False
    try:
        result = chat(
            prompt="",
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=False,
            extra_body_params={"grammar": grammar},
        )
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        return [
            ContextRelevanceResult(
                context_index=i, relevance_score=0, is_valid=False, error=str(e)
            )
            for i in range(len(contexts))
        ]

    raw_output = result.content.strip()
    logger.debug(f"Raw grammar output: '{raw_output[:200]}'")

    # Parse JSON (guaranteed valid by grammar, but guard against edge cases)
    try:
        parsed: list[dict] = json.loads(raw_output)
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse failed despite grammar: {e}"
        logger.error(error_msg)
        return [
            ContextRelevanceResult(
                context_index=i, relevance_score=0, is_valid=False, error=error_msg
            )
            for i in range(len(contexts))
        ]

    # Validate and normalize results
    results: list[ContextRelevanceResult] = []
    seen_indices: set[int] = set()

    for item in parsed:
        idx = item.get("context_index")
        score = item.get("relevance_score")

        if not isinstance(idx, int) or idx < 0 or idx >= len(contexts):
            logger.warning(f"Invalid context_index in result: {item}")
            continue
        if idx in seen_indices:
            logger.warning(f"Duplicate context_index {idx}, skipping")
            continue
        if score not in (0, 1, 2):
            logger.warning(f"Invalid score {score} for index {idx}, clamping to 0")
            score = 0

        seen_indices.add(idx)
        results.append(
            ContextRelevanceResult(
                context_index=idx,
                relevance_score=score,
                is_valid=True,
                error=None,
            )
        )

    # Fill in any missing indices with invalid results
    for i in range(len(contexts)):
        if i not in seen_indices:
            logger.warning(f"Missing result for context_index {i}")
            results.append(
                ContextRelevanceResult(
                    context_index=i,
                    relevance_score=0,
                    is_valid=False,
                    error=f"No result returned for context {i}",
                )
            )

    # Sort by context_index to maintain input order
    results.sort(key=lambda r: r["context_index"])

    valid_count = sum(1 for r in results if r["is_valid"])
    logger.info(f"Evaluation complete: {valid_count}/{len(results)} valid results")
    return results


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    test_query = "What is the capital of France?"
    test_contexts = [
        "The theory of relativity was developed by Albert Einstein.",
        "Paris hosts many tourists in France and is known for the Eiffel Tower.",
        "The capital of France is Paris, located in northern Europe.",
        "Berlin is the capital of Germany, not France.",
    ]

    results = evaluate_multiple_contexts_relevance(test_query, test_contexts)

    score_styles = {0: "dim red", 1: "yellow", 2: "bold green"}
    score_labels = {0: "Low", 1: "Medium", 2: "High"}

    console.print("\n[bold green]Multiple Contexts Relevance Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Context", style="white", max_width=50)
    table.add_column("Score", justify="center", width=12)
    table.add_column("Valid", justify="center", width=6)

    for r in results:
        style = score_styles.get(r["relevance_score"], "dim")
        label = score_labels.get(r["relevance_score"], "?")
        v_str = (
            "[bold green]✓[/bold green]" if r["is_valid"] else "[bold red]✗[/bold red]"
        )
        err = f"\n[dim red]⚠ {r['error']}[/dim red]" if not r["is_valid"] else ""
        ctx_text = test_contexts[r["context_index"]][:60]
        if len(test_contexts[r["context_index"]]) > 60:
            ctx_text += "..."
        table.add_row(
            str(r["context_index"]),
            ctx_text,
            f"[{style}]{r['relevance_score']} ({label})[/{style}]{err}",
            v_str,
        )

    console.print(table)
