"""Answer multiple-choice questions returning the choice key.

Equivalent to jet/llm/mlx/tasks/answer_multiple_choice_with_key.py.
Parses 'Key) Text' formatted choices, biases key tokens via logit_bias,
and returns both the selected key and resolved text.
Uses moderate logit_bias (5) + max_tokens=1 to avoid repetition loops.
"""

import re
from typing import Optional, TypedDict

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import chat
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.logger import logger


class AnswerResult(TypedDict):
    answer_key: str
    token_id: int
    is_valid: bool
    method: str
    error: Optional[str]
    text: str
    prob: float


CHOICE_PATTERN = re.compile(r"^\s*([a-zA-Z0-9]+)[\)\.\:]\s*(.+?)\s*$")
LOGIT_BIAS_VALUE = 5


def _parse_choices(choices: list[str]) -> tuple[dict[str, str], list[str]]:
    """Parse 'A) Text' format into key→text mapping and text list."""
    key_to_text: dict[str, str] = {}
    texts: list[str] = []
    for choice in choices:
        m = CHOICE_PATTERN.match(choice.strip())
        if not m:
            raise ValueError(
                f"Choice '{choice}' does not match expected format (e.g., 'A) Text')"
            )
        key, text = m.groups()
        if not key or not text.strip():
            raise ValueError(f"Choice '{choice}' has empty key or text")
        if key in key_to_text:
            raise ValueError(f"Duplicate key '{key}' in choices")
        key_to_text[key] = text.strip()
        texts.append(text.strip())
    return key_to_text, texts


def _create_system_prompt(choices: list[str]) -> str:
    options_text = "\n".join(choices)
    return (
        "Answer the following question by choosing exactly ONE option. "
        "Return ONLY the option key (e.g., A, B, 1, 2) and nothing else.\n"
        f"Options:\n{options_text}"
    )


def _build_key_logit_bias(
    tokenizer, keys: list[str]
) -> tuple[dict[str, int], dict[int, str]]:
    """Build logit_bias for key tokens only."""
    bias: dict[str, int] = {}
    token_to_key: dict[int, str] = {}
    for key in keys:
        tokens = tokenizer.encode(key, add_special_tokens=False)
        if tokens:
            tid = tokens[0]
            bias[str(tid)] = LOGIT_BIAS_VALUE
            token_to_key[tid] = key
            logger.debug(
                f"logit_bias: key '{key}' -> token {tid} (bias={LOGIT_BIAS_VALUE})"
            )
    return bias, token_to_key


def answer_multiple_choice_with_key(
    question: str,
    choices: list[str],
    model: str | None = None,
    max_tokens: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> AnswerResult:
    """Answer a multiple-choice question, returning the choice key.

    Args:
        question: The question to answer.
        choices: List of choices in 'Key) Text' format (e.g., 'A) Paris').
        model: LLM model key. Defaults to LLM_MODEL.
        max_tokens: Max tokens to generate (default: 1 for single key).
        temperature: Sampling temperature (default: 0.0 for deterministic).
        top_p: Nucleus sampling threshold (default: 0.9).

    Returns:
        AnswerResult with answer_key, resolved text, validity flag, and optional error.
        prob is always 0.0 (raw logits unavailable via OpenAI-compatible API).
    """
    resolved_model = model or LLM_MODEL

    if not question.strip():
        return AnswerResult(
            answer_key="",
            token_id=-1,
            is_valid=False,
            method="chat",
            error="Question cannot be empty.",
            text="",
            prob=0.0,
        )
    if not choices:
        return AnswerResult(
            answer_key="",
            token_id=-1,
            is_valid=False,
            method="chat",
            error="Choices cannot be empty.",
            text="",
            prob=0.0,
        )

    try:
        key_to_text, _ = _parse_choices(choices)
    except ValueError as e:
        logger.error(f"Invalid choice format: {e}")
        return AnswerResult(
            answer_key="",
            token_id=-1,
            is_valid=False,
            method="chat",
            error=str(e),
            text="",
            prob=0.0,
        )

    valid_keys = list(key_to_text.keys())
    logger.info(
        f"answer_mc_with_key: model={resolved_model}, "
        f"{len(valid_keys)} keys, question='{question[:60]}...'"
    )

    tokenizer = get_tokenizer(resolved_model)
    system_prompt = _create_system_prompt(choices)
    logit_bias, token_to_key = _build_key_logit_bias(tokenizer, valid_keys)

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
            answer_key="",
            token_id=-1,
            is_valid=False,
            method="chat",
            error=str(e),
            text="",
            prob=0.0,
        )

    raw_output = result.content.strip()
    logger.debug(f"Raw model output: '{raw_output}'")

    # Match output to a valid key (exact or prefix)
    matched_key = None
    matched_token_id = -1
    for key in valid_keys:
        if raw_output == key or key.startswith(raw_output):
            matched_key = key
            tokens = tokenizer.encode(key, add_special_tokens=False)
            matched_token_id = tokens[0] if tokens else -1
            break

    if matched_key is None:
        error_msg = f"Output '{raw_output}' is not a valid choice key: {valid_keys}"
        logger.error(error_msg)
        return AnswerResult(
            answer_key="",
            token_id=-1,
            is_valid=False,
            method="chat",
            error=error_msg,
            text="",
            prob=0.0,
        )

    resolved_text = key_to_text[matched_key]
    logger.info(f"Selected: key='{matched_key}', text='{resolved_text}'")

    return AnswerResult(
        answer_key=matched_key,
        token_id=matched_token_id,
        is_valid=True,
        method="chat",
        error=None,
        text=resolved_text,
        prob=0.0,  # Not available without raw logits
    )


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    test_cases = [
        {
            "question": "What is the capital of France?",
            "choices": ["A) London", "B) Paris", "C) Berlin", "D) Madrid"],
            "expected_key": "B",
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "choices": ["1) Venus", "2) Mars", "3) Jupiter", "4) Saturn"],
            "expected_key": "2",
        },
        {
            "question": "What is 2 + 2?",
            "choices": ["X) 3", "Y) 4", "Z) 5"],
            "expected_key": "Y",
        },
    ]

    console.print("\n[bold green]Multiple Choice With Key Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Question", style="cyan", max_width=35)
    table.add_column("Key", justify="center", width=6)
    table.add_column("Text", style="white", max_width=25)
    table.add_column("Expected", justify="center", width=8)
    table.add_column("Match", justify="center", width=6)

    for idx, case in enumerate(test_cases, start=1):
        result = answer_multiple_choice_with_key(case["question"], case["choices"])

        match = result["is_valid"] and result["answer_key"] == case["expected_key"]
        match_str = "[bold green]✓[/bold green]" if match else "[bold red]✗[/bold red]"
        key_style = (
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
            f"[{key_style}]{result['answer_key'] or 'N/A'}[/{key_style}]",
            f"{result['text']}{err}",
            case["expected_key"],
            match_str,
        )

    console.print(table)
