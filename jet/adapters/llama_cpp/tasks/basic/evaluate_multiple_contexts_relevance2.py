# jet_python_modules/jet/adapters/llama_cpp/tasks/evaluate_multiple_contexts_relevance.py
"""Evaluate whether multiple contexts contain answers to a query using grammar-constrained generation.
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
    """Single context answer-containment evaluation result."""

    context_index: int
    has_answer: bool
    is_valid: bool
    error: Optional[str]


SYSTEM_PROMPT = """\
You are an answer-containment evaluator. Given a query and multiple numbered contexts, \
determine whether each context contains information that directly answers the query.
Criteria:
- true: The context contains specific information that answers the query
- false: The context does not contain information that answers the query (even if topically related)
Return ONLY a JSON array with one object per context, in order. \
Each object must have "context_index" (integer) and "has_answer" (boolean). \
Do NOT include any other text."""


def _build_containment_schema(num_contexts: int) -> dict:
    """Build JSON Schema with prefixItems for fixed-length containment results.
    Each position has a const context_index matching its array position,
    ensuring the model cannot produce wrong indices.
    Uses boolean instead of enum for faster grammar and simpler decoding.
    """
    if num_contexts <= 0:
        raise ValueError(f"num_contexts must be >= 1, got {num_contexts}")
    item_schemas = [
        {
            "type": "object",
            "properties": {
                "context_index": {"const": i},
                "has_answer": {"type": "boolean"},
            },
            "required": ["context_index", "has_answer"],
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
    """Evaluate whether multiple contexts contain answers to a single query.
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
                context_index=i, has_answer=False, is_valid=False, error="Empty query"
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
    try:
        schema = _build_containment_schema(len(contexts))
        grammar = build_grammar_from_schema(
            schema,
            prop_order={"context_index": 0, "has_answer": 1},
        )
    except ValueError as e:
        logger.error(f"Grammar generation failed: {e}")
        return [
            ContextRelevanceResult(
                context_index=i, has_answer=False, is_valid=False, error=str(e)
            )
            for i in range(len(contexts))
        ]
    validation_error = validate_grammar(grammar)
    if validation_error:
        error_msg = f"Grammar validation failed: {validation_error}"
        logger.error(error_msg)
        return [
            ContextRelevanceResult(
                context_index=i, has_answer=False, is_valid=False, error=error_msg
            )
            for i in range(len(contexts))
        ]
    logger.debug(f"Grammar OK ({len(grammar)} bytes, {len(contexts)} items)")
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
                context_index=i, has_answer=False, is_valid=False, error=str(e)
            )
            for i in range(len(contexts))
        ]
    raw_output = result.content.strip()
    logger.debug(f"Raw output ({len(raw_output)} chars): '{raw_output[:300]}'")
    try:
        parsed: list[dict] = json.loads(raw_output)
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse failed despite grammar: {e}. Possible truncation."
        logger.error(error_msg)
        return [
            ContextRelevanceResult(
                context_index=i, has_answer=False, is_valid=False, error=error_msg
            )
            for i in range(len(contexts))
        ]
    results: list[ContextRelevanceResult] = []
    seen: set[int] = set()
    for item in parsed:
        idx = item.get("context_index")
        has_answer = item.get("has_answer")
        if not isinstance(idx, int) or idx < 0 or idx >= len(contexts):
            logger.warning(f"Invalid context_index: {item}")
            continue
        if idx in seen:
            logger.warning(f"Duplicate context_index {idx}")
            continue
        if not isinstance(has_answer, bool):
            logger.warning(
                f"Invalid has_answer {has_answer} for index {idx}, defaulting to False"
            )
            has_answer = False
        seen.add(idx)
        results.append(
            ContextRelevanceResult(
                context_index=idx,
                has_answer=has_answer,
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
                    has_answer=False,
                    is_valid=False,
                    error=f"No result for context {i}",
                )
            )
    results.sort(key=lambda r: r["context_index"])
    valid_count = sum(1 for r in results if r["is_valid"])
    positive_count = sum(1 for r in results if r["is_valid"] and r["has_answer"])
    logger.info(
        f"Evaluation complete: {valid_count}/{len(results)} valid, "
        f"{positive_count} contexts contain answers"
    )
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
    console.print(
        "\n[bold green]Multiple Contexts Answer Containment Results[/bold green]"
    )
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Context", style="white", max_width=50)
    table.add_column("Has Answer", justify="center", width=12)
    table.add_column("Valid", justify="center", width=6)
    for r in results:
        has_ans_str = (
            "[bold green]✓ Yes[/bold green]"
            if r["has_answer"]
            else "[dim red]✗ No[/dim red]"
        )
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
            f"{has_ans_str}{err}",
            v_str,
        )
    console.print(table)
