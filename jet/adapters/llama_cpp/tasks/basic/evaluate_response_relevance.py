"""Evaluate response relevance using constrained generative classification.

Equivalent to jet/llm/mlx/tasks/eval/evaluate_response_relevance.py.
Uses llm_utils.chat with logit_bias to constrain output to {0, 1, 2}.
"""

from typing import Optional, TypedDict

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import chat
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.logger import logger


class RelevanceResult(TypedDict):
    relevance_score: int
    is_valid: bool
    error: Optional[str]


VALID_SCORES = ["0", "1", "2"]
SCORE_LABELS = {
    0: ("Irrelevant", "dim red"),
    1: ("Partial", "yellow"),
    2: ("Highly Relevant", "bold green"),
}

SYSTEM_PROMPT = (
    "You are an expert evaluator assessing the relevance of a response to a given query and context. "
    "Based on the query, context, and response provided, assign a relevance score as follows: "
    "0 (irrelevant: the response does not address the query or context), "
    "1 (partially relevant: the response addresses some aspects but is incomplete or tangential), "
    "2 (highly relevant: the response directly and accurately addresses the query and context). "
    "Output only the score (0, 1, or 2) and nothing else."
)


def _build_logit_bias(model: str) -> dict[str, int]:
    """Build logit_bias encouraging only valid score tokens."""
    tokenizer = get_tokenizer(model)
    bias: dict[str, int] = {}
    for choice in VALID_SCORES:
        tokens = tokenizer.encode(choice, add_special_tokens=False)
        if tokens:
            bias[str(tokens[0])] = 100
            logger.debug(f"logit_bias: '{choice}' -> token {tokens[0]} (bias=100)")
    return bias


def evaluate_response_relevance(
    query: str,
    context: str,
    response: str,
    model: str | None = None,
    max_tokens: int = 1,
    temperature: float = 0.1,
) -> RelevanceResult:
    """Evaluate if a response is relevant to the query and context.

    Args:
        query: User query.
        context: Retrieved context provided to the generator.
        response: Generated response to evaluate.
        model: LLM model key. Defaults to LLM_MODEL.
        max_tokens: Max tokens to generate (default: 1).
        temperature: Sampling temperature (default: 0.1).

    Returns:
        RelevanceResult with score (0-2), validity flag, and optional error.
    """
    resolved_model = model or LLM_MODEL

    for name, val in [("Query", query), ("Context", context), ("Response", response)]:
        if not val.strip():
            msg = f"{name} cannot be empty."
            logger.error(msg)
            return RelevanceResult(relevance_score=0, is_valid=False, error=msg)

    logger.info(
        f"evaluate_response_relevance: model={resolved_model}, "
        f"query='{query[:50]}...', response='{response[:50]}...'"
    )

    logit_bias = _build_logit_bias(resolved_model)
    user_content = f"Query: {query}\nContext: {context}\nResponse: {response}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        result = chat(
            prompt="",
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logit_bias=logit_bias,
            stop=["\n"],
        )
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        return RelevanceResult(relevance_score=0, is_valid=False, error=str(e))

    answer = result.content.strip()
    logger.debug(f"Raw model output: '{answer}'")

    if answer not in VALID_SCORES:
        error_msg = f"Output '{answer}' is not a valid relevance score (0-2)."
        logger.error(error_msg)
        return RelevanceResult(relevance_score=0, is_valid=False, error=error_msg)

    score = int(answer)
    label, _ = SCORE_LABELS[score]
    logger.info(f"Response relevance: {score} ({label})")
    return RelevanceResult(relevance_score=score, is_valid=True, error=None)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    test_cases = [
        {
            "query": "What is the capital of France?",
            "context": "France is a country in Western Europe.",
            "response": "The capital of France is Paris.",
            "expected": 2,
        },
        {
            "query": "What is the capital of France?",
            "context": "France is a country in Western Europe.",
            "response": "Paris is a nice city with many tourists.",
            "expected": 1,
        },
        {
            "query": "What is the capital of France?",
            "context": "France is a country in Western Europe.",
            "response": "Machine learning is a subset of artificial intelligence.",
            "expected": 0,
        },
    ]

    console.print("\n[bold green]Response Relevance Evaluation Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Query", style="cyan", max_width=30)
    table.add_column("Response", style="white", max_width=45)
    table.add_column("Score", justify="center", width=18)
    table.add_column("Expected", justify="center", width=18)
    table.add_column("Match", justify="center", width=6)

    for idx, case in enumerate(test_cases, start=1):
        result = evaluate_response_relevance(
            case["query"], case["context"], case["response"]
        )

        s_label, s_style = SCORE_LABELS.get(result["relevance_score"], ("?", "dim"))
        e_label, e_style = SCORE_LABELS.get(case["expected"], ("?", "dim"))
        match = result["is_valid"] and result["relevance_score"] == case["expected"]
        match_str = "[bold green]✓[/bold green]" if match else "[bold red]✗[/bold red]"

        err = (
            f"\n[dim red]⚠ {result['error']}[/dim red]"
            if not result["is_valid"]
            else ""
        )

        table.add_row(
            str(idx),
            case["query"],
            case["response"][:70] + ("..." if len(case["response"]) > 70 else ""),
            f"[{s_style}]{result['relevance_score']} ({s_label})[/{s_style}]{err}",
            f"[{e_style}]{case['expected']} ({e_label})[/{e_style}]",
            match_str,
        )

    console.print(table)
