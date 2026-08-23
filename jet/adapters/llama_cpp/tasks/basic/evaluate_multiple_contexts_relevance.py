"""Evaluate whether multiple contexts contain answers to a query using grammar-constrained generation.
Returns ONLY contexts with non-zero answer scores, minimizing output tokens.
Absent indices implicitly have answer_score=0.
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
from jet.libs.llama_cpp.usage.chat_stream_observability import PHOENIX_URL
from jet.logger import logger


class ContextRelevanceResult(TypedDict):
    """Single context answer-containment evaluation result."""

    context_index: int
    answer_score: int
    has_answer: bool
    is_valid: bool
    error: Optional[str]


SYSTEM_PROMPT = """\
You are an answer-containment evaluator. Given a query and numbered contexts, \
return ONLY the contexts that contain information answering the query.
Scoring:
- 1: Partially answers or provides indirect supporting information
- 2: Directly and clearly answers the query
Rules:
- Include ONLY contexts with score 1 or 2
- Do NOT include contexts that are merely topically related but do not answer
- Return a JSON array of objects with "context_index" (integer) and "answer_score" (1 or 2)
- Return an empty array [] if no context answers the query
- Do NOT include any other text"""


def _build_containment_schema(num_contexts: int) -> dict:
    """Build minimal JSON Schema: array of {context_index, answer_score} objects.
    Only non-zero scores are emitted; absent indices imply answer_score=0.
    """
    if num_contexts <= 0:
        raise ValueError(f"num_contexts must be >= 1, got {num_contexts}")
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "context_index": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": num_contexts - 1,
                },
                "answer_score": {"enum": [1, 2]},
            },
            "required": ["context_index", "answer_score"],
            "additionalProperties": False,
        },
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
    project_name: str | None = "eval-context-relevance",
    phoenix_url: str = PHOENIX_URL,
) -> list[ContextRelevanceResult]:
    """Evaluate whether multiple contexts contain answers to a single query.

    Returns one result per input context. Contexts present in the LLM response
    have answer_score 1 or 2; all others have answer_score 0.

    Args:
        query: The user query to evaluate against.
        contexts: List of context strings to assess.
        model: LLM model key. Defaults to LLM_MODEL.
        temperature: Sampling temperature (default: 0.0 for deterministic).
        max_tokens: Max tokens for the JSON response.
        project_name: Phoenix project name for trace grouping. Set to None to disable tracing.
        phoenix_url: Phoenix server base URL for trace links.

    Returns:
        List of ContextRelevanceResult dicts, one per input context, sorted by index.
    """
    resolved_model = model or LLM_MODEL
    num_contexts = len(contexts)

    if not query.strip():
        logger.error("Query cannot be empty")
        return [
            ContextRelevanceResult(
                context_index=i,
                answer_score=0,
                has_answer=False,
                is_valid=False,
                error="Empty query",
            )
            for i in range(num_contexts)
        ]

    if not contexts:
        logger.error("Contexts list cannot be empty")
        return []

    logger.info(
        f"evaluate_multiple_contexts_relevance: model={resolved_model}, "
        f"{num_contexts} contexts, query='{query[:60]}...', project={project_name}"
    )

    try:
        schema = _build_containment_schema(num_contexts)
        grammar = build_grammar_from_schema(
            schema,
            prop_order={"context_index": 0, "answer_score": 1},
        )
    except ValueError as e:
        logger.error(f"Grammar generation failed: {e}")
        return [
            ContextRelevanceResult(
                context_index=i,
                answer_score=0,
                has_answer=False,
                is_valid=False,
                error=str(e),
            )
            for i in range(num_contexts)
        ]

    validation_error = validate_grammar(grammar)
    if validation_error:
        error_msg = f"Grammar validation failed: {validation_error}"
        logger.error(error_msg)
        return [
            ContextRelevanceResult(
                context_index=i,
                answer_score=0,
                has_answer=False,
                is_valid=False,
                error=error_msg,
            )
            for i in range(num_contexts)
        ]

    logger.debug(f"Grammar OK ({len(grammar)} bytes, {num_contexts} contexts)")

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
            project_name=project_name,
            phoenix_url=phoenix_url,
            extra_body_params={"grammar": grammar},
        )
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        return [
            ContextRelevanceResult(
                context_index=i,
                answer_score=0,
                has_answer=False,
                is_valid=False,
                error=str(e),
            )
            for i in range(num_contexts)
        ]

    raw_output = result.content.strip()
    logger.debug(f"Raw output ({len(raw_output)} chars): '{raw_output[:300]}'")

    try:
        parsed: list = json.loads(raw_output)
    except json.JSONDecodeError as e:
        error_msg = f"JSON parse failed despite grammar: {e}. Possible truncation."
        logger.error(error_msg)
        return [
            ContextRelevanceResult(
                context_index=i,
                answer_score=0,
                has_answer=False,
                is_valid=False,
                error=error_msg,
            )
            for i in range(num_contexts)
        ]

    if not isinstance(parsed, list):
        error_msg = f"Expected JSON array, got {type(parsed).__name__}"
        logger.error(error_msg)
        return [
            ContextRelevanceResult(
                context_index=i,
                answer_score=0,
                has_answer=False,
                is_valid=False,
                error=error_msg,
            )
            for i in range(num_contexts)
        ]

    score_map: dict[int, int] = {}
    for item in parsed:
        if not isinstance(item, dict):
            logger.warning(f"Ignoring non-dict entry in response: {item!r}")
            continue
        idx = item.get("context_index")
        score = item.get("answer_score")
        if not isinstance(idx, int) or idx < 0 or idx >= num_contexts:
            logger.warning(f"Ignoring invalid context_index: {item}")
            continue
        if score not in (1, 2):
            logger.warning(f"Ignoring invalid answer_score {score} for index {idx}")
            continue
        if idx in score_map:
            logger.warning(f"Duplicate context_index {idx}, keeping higher score")
            score_map[idx] = max(score_map[idx], score)
        else:
            score_map[idx] = score

    results: list[ContextRelevanceResult] = []
    for i in range(num_contexts):
        score = score_map.get(i, 0)
        results.append(
            ContextRelevanceResult(
                context_index=i,
                answer_score=score,
                has_answer=(score > 0),
                is_valid=True,
                error=None,
            )
        )

    positive_count = len(score_map)
    logger.info(
        f"Evaluation complete: {positive_count}/{num_contexts} contexts contain answers"
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

    score_styles = {0: "dim red", 1: "yellow", 2: "bold green"}
    score_labels = {0: "None", 1: "Partial", 2: "Direct"}

    console.print("\n[bold green]Sparse Answer Containment Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Context", style="white", max_width=50)
    table.add_column("Score", justify="center", width=12)
    table.add_column("Valid", justify="center", width=6)

    for r in results:
        style = score_styles.get(r["answer_score"], "dim")
        label = score_labels.get(r["answer_score"], "?")
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
            f"[{style}]{r['answer_score']} ({label})[/{style}]{err}",
            v_str,
        )

    console.print(table)
