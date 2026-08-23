"""Evaluate context relevance using constrained generative classification via llama.cpp server.

Equivalent to jet/llm/mlx/tasks/eval/evaluate_context_relevance.py but uses the
OpenAI-compatible chat endpoint instead of local MLX inference. Reuses llm_utils.chat
for constrained generation and token_utils for token ID resolution.
"""

from typing import Optional, TypedDict

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import chat
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.logger import logger


class InvalidInputError(Exception):
    """Raised when query or context is empty or invalid."""

    pass


class InvalidOutputError(Exception):
    """Raised when the generated output is not a valid score."""

    pass


class RelevanceResult(TypedDict):
    relevance_score: int
    is_valid: bool
    error: Optional[str]


VALID_SCORES = ["0", "1", "2"]

SCORE_LABELS = {
    0: ("Low", "dim red"),
    1: ("Medium", "yellow"),
    2: ("High", "bold green"),
}

SYSTEM_PROMPT = """\
Evaluate if the provided context is relevant to the query. Choose one option based on how well the context addresses the query:
0: Low relevance (context is unrelated or barely related to the query)
1: Medium relevance (context partially addresses the query)
2: High relevance (context directly and mostly addresses the query)
Examples:
- Query: "What is the capital of France?"
  - Context: "The theory of relativity was developed by Albert Einstein." -> 0 (completely unrelated)
  - Context: "Paris hosts many tourists in France." -> 1 (mentions Paris but not as capital)
  - Context: "The capital of France is Paris." -> 2 (direct and complete)
Return only the number (0, 1, or 2) without additional text."""


def _validate_inputs(query: str, context: str) -> None:
    """Validates that query and context are non-empty."""
    if not query.strip():
        raise InvalidInputError("Query cannot be empty.")
    if not context.strip():
        raise InvalidInputError("Context cannot be empty.")


def _build_logit_bias(model: str) -> dict[str, int]:
    """Build logit_bias dict that strongly encourages only valid score tokens.

    Resolves token IDs for '0', '1', '2' using the model's tokenizer,
    then assigns a high positive bias to those tokens so the model almost
    always selects one of them when max_tokens=1.
    """
    tokenizer = get_tokenizer(model)
    bias: dict[str, int] = {}
    for choice in VALID_SCORES:
        tokens = tokenizer.encode(choice, add_special_tokens=False)
        if tokens:
            # Use the first token ID; OpenAI API expects string keys
            bias[str(tokens[0])] = 100
            logger.debug(f"logit_bias: '{choice}' -> token {tokens[0]} (bias=100)")
    return bias


def evaluate_context_relevance(
    query: str,
    context: str,
    model: str | None = None,
    max_tokens: int = 1,
    temperature: float = 0.1,
) -> RelevanceResult:
    """Evaluate if retrieved context is relevant to the query using constrained generation.

    This is the llama.cpp adapter equivalent of
    jet/llm/mlx/tasks/eval/evaluate_context_relevance.py. Instead of local MLX
    inference with logits processors, it uses the OpenAI-compatible chat endpoint
    with logit_bias to constrain output to valid score tokens.

    Args:
        query: The user query to evaluate against.
        context: The retrieved context/document to assess.
        model: LLM model key. Defaults to LLM_MODEL from config.
        max_tokens: Maximum tokens to generate (default: 1 for single-score output).
        temperature: Sampling temperature (default: 0.1 for deterministic output).

    Returns:
        RelevanceResult with relevance_score (0-2), is_valid flag, and optional error.
    """
    resolved_model = model or LLM_MODEL

    try:
        _validate_inputs(query, context)
    except InvalidInputError as e:
        logger.error(f"Invalid input: {e}")
        return RelevanceResult(relevance_score=0, is_valid=False, error=str(e))

    logger.info(
        f"evaluate_context_relevance: model={resolved_model}, "
        f"query='{query[:60]}...', context='{context[:60]}...'"
    )

    # Build constrained logit_bias
    logit_bias = _build_logit_bias(resolved_model)

    # Format messages matching MLX reference exactly
    user_content = f"Query: {query}\nContext: {context}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.debug("Sending constrained chat completion request...")
    try:
        result = chat(
            prompt="",  # Not used when messages are provided
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
    logger.info(f"Context relevance: {score} ({label})")

    return RelevanceResult(relevance_score=score, is_valid=True, error=None)


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    test_cases = [
        {
            "query": "What is the capital of France?",
            "context": "The theory of relativity was developed by Albert Einstein.",
            "expected": 0,
        },
        {
            "query": "What is the capital of France?",
            "context": "Paris hosts many tourists in France.",
            "expected": 1,
        },
        {
            "query": "What is the capital of France?",
            "context": "The capital of France is Paris.",
            "expected": 2,
        },
        {
            "query": "Explain gravity",
            "context": "Gravity is a force that attracts two bodies towards each other. "
            "It gives weight to physical objects and is responsible for "
            "the movement of planets around the sun.",
            "expected": 2,
        },
    ]

    console.print("\n[bold green]Context Relevance Evaluation Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Query", style="cyan", no_wrap=False, max_width=35)
    table.add_column("Context", style="white", no_wrap=False, max_width=50)
    table.add_column("Score", justify="center", width=10)
    table.add_column("Expected", justify="center", width=10)
    table.add_column("Match", justify="center", width=6)

    for idx, case in enumerate(test_cases, start=1):
        result = evaluate_context_relevance(case["query"], case["context"])

        score_label, score_style = SCORE_LABELS.get(
            result["relevance_score"], ("Unknown", "dim")
        )
        expected_label, expected_style = SCORE_LABELS.get(
            case["expected"], ("?", "dim")
        )

        match = result["is_valid"] and result["relevance_score"] == case["expected"]
        match_str = "[bold green]✓[/bold green]" if match else "[bold red]✗[/bold red]"

        error_note = ""
        if not result["is_valid"]:
            error_note = f"\n[dim red]⚠ {result['error']}[/dim red]"

        table.add_row(
            str(idx),
            case["query"],
            case["context"][:80] + ("..." if len(case["context"]) > 80 else ""),
            f"[{score_style}]{result['relevance_score']} ({score_label})[/{score_style}]"
            + error_note,
            f"[{expected_style}]{case['expected']} ({expected_label})[/{expected_style}]",
            match_str,
        )

    console.print(table)
