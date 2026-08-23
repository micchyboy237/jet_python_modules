# jet_python_modules/jet/adapters/llama_cpp/tasks/evaluate_multiple_contexts_relevance.py
"""Evaluate relevance of multiple contexts against a single query using grammar-constrained generation.

Uses llama.cpp's json_schema_to_grammar via grammar_builder to generate GBNF
from a dynamically-built JSON Schema with prefixItems (tuple validation).
Each context gets a per-position schema with const context_index.

NOTE: enable_thinking is FORCED to False because thinking tokens break grammar constraints.
"""

import json
from typing import Optional, TypedDict

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.helpers.grammar_builder import (
    build_grammar_from_schema,
    validate_grammar,
)
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


def _build_relevance_schema(num_contexts: int) -> dict:
    """Build JSON Schema with prefixItems for fixed-length relevance results.

    Each position has a const context_index matching its array position,
    ensuring the model cannot produce wrong indices.
    """
    if num_contexts <= 0:
        raise ValueError(f"num_contexts must be >= 1, got {num_contexts}")

    item_schemas = [
        {
            "type": "object",
            "properties": {
                "context_index": {"const": i},
                "relevance_score": {"enum": [0, 1, 2]},
            },
            "required": ["context_index", "relevance_score"],
            "additionalProperties": False,
        }
        for i in range(num_contexts)
    ]

    return {
        "type": "array",
        "prefixItems": item_schemas,
        "minItems": num_contexts,
        "maxItems": num_contexts,
    }


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

    Generates GBNF grammar from a dynamic JSON Schema using llama.cpp's
    json_schema_to_grammar converter. Grammar is validated client-side
    when gbnf.dev is available.

    Args:
        query: The user query to evaluate against.
        contexts: List of context strings to assess.
        model: LLM model key. Defaults to LLM_MODEL.
        temperature: Sampling temperature (default: 0.0 for deterministic).
        max_tokens: Max tokens for the JSON response.

    Returns:
        List of ContextRelevanceResult dicts, one per input context, sorted by index.
    """
    resolved_model = model or LLM_MODEL

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

    # Build schema → grammar
    try:
        schema = _build_relevance_schema(len(contexts))
        grammar = build_grammar_from_schema(
            schema,
            prop_order={"context_index": 0, "relevance_score": 1},
        )
    except ValueError as e:
        logger.error(f"Grammar generation failed: {e}")
        return [
            ContextRelevanceResult(
                context_index=i, relevance_score=0, is_valid=False, error=str(e)
            )
            for i in range(len(contexts))
        ]

    # Validate client-side
    validation_error = validate_grammar(grammar)
    if validation_error:
        error_msg = f"Grammar validation failed: {validation_error}"
        logger.error(error_msg)
        return [
            ContextRelevanceResult(
                context_index=i, relevance_score=0, is_valid=False, error=error_msg
            )
            for i in range(len(contexts))
        ]

    logger.debug(f"Grammar OK ({len(grammar)} bytes, {len(contexts)} items)")

    # Call LLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(query, contexts)},
    ]

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
    logger.debug(f"Raw output ({len(raw_output)} chars): '{raw_output[:300]}'")

    # Parse
    try:
        parsed: list[dict] = json.loads(raw_output)
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse failed despite grammar: {e}. Possible truncation."
        logger.error(error_msg)
        return [
            ContextRelevanceResult(
                context_index=i, relevance_score=0, is_valid=False, error=error_msg
            )
            for i in range(len(contexts))
        ]

    # Normalize
    results: list[ContextRelevanceResult] = []
    seen: set[int] = set()

    for item in parsed:
        idx = item.get("context_index")
        score = item.get("relevance_score")

        if not isinstance(idx, int) or idx < 0 or idx >= len(contexts):
            logger.warning(f"Invalid context_index: {item}")
            continue
        if idx in seen:
            logger.warning(f"Duplicate context_index {idx}")
            continue
        if score not in (0, 1, 2):
            logger.warning(f"Invalid score {score} for index {idx}, defaulting to 0")
            score = 0

        seen.add(idx)
        results.append(
            ContextRelevanceResult(
                context_index=idx,
                relevance_score=score,
                is_valid=True,
                error=None,
            )
        )

    for i in range(len(contexts)):
        if i not in seen:
            logger.warning(f"Missing result for context_index {i}")
            results.append(
                ContextRelevanceResult(
                    context_index=i,
                    relevance_score=0,
                    is_valid=False,
                    error=f"No result for context {i}",
                )
            )

    results.sort(key=lambda r: r["context_index"])
    valid_count = sum(1 for r in results if r["is_valid"])
    logger.info(f"Evaluation complete: {valid_count}/{len(results)} valid")
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
