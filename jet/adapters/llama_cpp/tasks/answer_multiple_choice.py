"""Answer multiple-choice questions using constrained generative selection.

Equivalent to jet/llm/mlx/tasks/answer_multiple_choice.py.
Uses llm_utils.chat with logit_bias to constrain output to valid choice tokens.
With temperature=0.0 + moderate logit_bias, the model deterministically selects
the highest-probability valid token (equivalent to MLX argmax).
"""

from typing import Optional, TypedDict

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import chat
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.logger import logger


class AnswerResult(TypedDict):
    answer: str
    token_id: int
    is_valid: bool
    method: str
    error: Optional[str]


LOGIT_BIAS_VALUE = (
    5  # Moderate bias: encourages valid tokens without causing repetition loops
)


def _validate_inputs(question: str, choices: list[str]) -> None:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not choices:
        raise ValueError("Choices cannot be empty.")


def _create_system_prompt(choices: list[str]) -> str:
    options_text = "\n".join(choices)
    return (
        "Answer the following question by choosing exactly ONE option from the list below. "
        "Return ONLY the exact text of your chosen option and nothing else.\n"
        f"Options:\n{options_text}"
    )


def _build_choice_logit_bias(
    tokenizer, choices: list[str]
) -> tuple[dict[str, int], dict[int, str]]:
    """Build logit_bias for first tokens of each choice."""
    bias: dict[str, int] = {}
    token_to_choice: dict[int, str] = {}
    for choice in choices:
        tokens = tokenizer.encode(choice, add_special_tokens=False)
        if tokens:
            tid = tokens[0]
            bias[str(tid)] = LOGIT_BIAS_VALUE
            token_to_choice[tid] = choice
            logger.debug(
                f"logit_bias: '{choice}' -> token {tid} (bias={LOGIT_BIAS_VALUE})"
            )
    return bias, token_to_choice


def answer_multiple_choice(
    question: str,
    choices: list[str],
    model: str | None = None,
    max_tokens: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> AnswerResult:
    """Answer a multiple-choice question using constrained generation.

    Args:
        question: The question to answer.
        choices: List of valid answer choices.
        model: LLM model key. Defaults to LLM_MODEL.
        max_tokens: Max tokens to generate (default: 1 for single-token answers).
        temperature: Sampling temperature (default: 0.0 for deterministic).
        top_p: Nucleus sampling threshold (default: 0.9).

    Returns:
        AnswerResult with selected answer, validity flag, and optional error.
    """
    resolved_model = model or LLM_MODEL

    try:
        _validate_inputs(question, choices)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return AnswerResult(
            answer="", token_id=-1, is_valid=False, method="chat", error=str(e)
        )

    logger.info(
        f"answer_multiple_choice: model={resolved_model}, "
        f"{len(choices)} choices, question='{question[:60]}...'"
    )

    tokenizer = get_tokenizer(resolved_model)
    system_prompt = _create_system_prompt(choices)
    logit_bias, token_to_choice = _build_choice_logit_bias(tokenizer, choices)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        result = chat(
            prompt="",
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logit_bias=logit_bias,
            stop=["\n"],
        )
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        return AnswerResult(
            answer="", token_id=-1, is_valid=False, method="chat", error=str(e)
        )

    answer = result.content.strip()
    logger.debug(f"Raw model output: '{answer}'")

    # Match against choices (exact or prefix match for multi-token choices)
    matched_choice = None
    matched_token_id = -1
    for choice in choices:
        if answer == choice or choice.startswith(answer):
            matched_choice = choice
            tokens = tokenizer.encode(choice, add_special_tokens=False)
            matched_token_id = tokens[0] if tokens else -1
            break

    if matched_choice is None:
        error_msg = f"Output '{answer}' is not one of the provided choices: {choices}"
        logger.error(error_msg)
        return AnswerResult(
            answer="", token_id=-1, is_valid=False, method="chat", error=error_msg
        )

    logger.info(f"Selected answer: '{matched_choice}'")
    return AnswerResult(
        answer=matched_choice,
        token_id=matched_token_id,
        is_valid=True,
        method="chat",
        error=None,
    )


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    test_cases = [
        {
            "question": "What is the capital of France?",
            "choices": ["London", "Berlin", "Paris", "Madrid"],
            "expected": "Paris",
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
            "expected": "Mars",
        },
        {
            "question": "What is 2 + 2?",
            "choices": ["3", "4", "5", "6"],
            "expected": "4",
        },
    ]

    console.print("\n[bold green]Multiple Choice Answer Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Question", style="cyan", max_width=40)
    table.add_column("Choices", style="white", max_width=35)
    table.add_column("Answer", justify="center", width=12)
    table.add_column("Expected", justify="center", width=12)
    table.add_column("Match", justify="center", width=6)

    for idx, case in enumerate(test_cases, start=1):
        result = answer_multiple_choice(case["question"], case["choices"])

        match = result["is_valid"] and result["answer"] == case["expected"]
        match_str = "[bold green]✓[/bold green]" if match else "[bold red]✗[/bold red]"
        ans_style = (
            "bold green" if match else ("yellow" if result["is_valid"] else "dim red")
        )

        err = (
            f"\n[dim red]⚠ {result['error']}[/dim red]"
            if not result["is_valid"]
            else ""
        )

        table.add_row(
            str(idx),
            case["question"],
            ", ".join(case["choices"]),
            f"[{ans_style}]{result['answer'] or 'N/A'}[/{ans_style}]{err}",
            case["expected"],
            match_str,
        )

    console.print(table)
