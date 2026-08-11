"""Answer multiple-choice questions allowing multiple selections.

Equivalent to jet/llm/mlx/tasks/answer_multiple_choice_multiple_selections.py.
Uses llm_utils.chat with newline-separated key output. Parses keys from response
and validates against parsed choice format (e.g., 'A) Text', '1) Text').
"""

import re
from typing import Optional, TypedDict

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import chat
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.logger import logger


class AnswerResult(TypedDict):
    answer_keys: list[str]
    token_ids: list[int]
    is_valid: bool
    method: str
    error: Optional[str]
    texts: list[str]
    prob: dict[str, float]


CHOICE_PATTERN = re.compile(r"^\s*([a-zA-Z0-9]+)[\)\.\:]\s*(.+?)\s*$")
LOGIT_BIAS_VALUE = 5  # Moderate bias to avoid repetition loops


def _parse_choices(choices: list[str]) -> tuple[dict[str, str], list[str]]:
    key_to_text: dict[str, str] = {}
    texts: list[str] = []
    for choice in choices:
        m = CHOICE_PATTERN.match(choice.strip())
        if not m:
            raise ValueError(
                f"Choice '{choice}' does not match expected format (e.g., 'A) Text')"
            )
        key, text = m.groups()
        if key in key_to_text:
            raise ValueError(f"Duplicate key '{key}' in choices")
        key_to_text[key] = text.strip()
        texts.append(text.strip())
    return key_to_text, texts


def _create_system_prompt(
    choices: list[str], max_selections: Optional[int] = None
) -> str:
    limit_note = f" Select AT MOST {max_selections} options." if max_selections else ""
    return (
        f"Select one or more options that best answer the question.{limit_note} "
        "Return each selected option KEY on its own line. Do NOT repeat keys. "
        "Do NOT add any other text.\n\n"
        f"Options:\n{'\\n'.join(choices)}"
    )


def _build_key_logit_bias(
    tokenizer, keys: list[str]
) -> tuple[dict[str, int], dict[int, str]]:
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
    nl_tokens = tokenizer.encode("\n", add_special_tokens=False)
    if nl_tokens:
        bias[str(nl_tokens[0])] = LOGIT_BIAS_VALUE
        logger.debug(
            f"logit_bias: newline -> token {nl_tokens[0]} (bias={LOGIT_BIAS_VALUE})"
        )
    return bias, token_to_key


def answer_multiple_choice_multiple_selections(
    question: str,
    choices: list[str],
    model: str | None = None,
    max_tokens: int = 20,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_selections: Optional[int] = None,
) -> AnswerResult:
    """Answer a multiple-choice question allowing multiple selections."""
    resolved_model = model or LLM_MODEL

    if not question.strip():
        return AnswerResult(
            answer_keys=[],
            token_ids=[],
            is_valid=False,
            method="chat",
            error="Question cannot be empty.",
            texts=[],
            prob={},
        )
    if not choices:
        return AnswerResult(
            answer_keys=[],
            token_ids=[],
            is_valid=False,
            method="chat",
            error="Choices cannot be empty.",
            texts=[],
            prob={},
        )

    try:
        key_to_text, _ = _parse_choices(choices)
    except ValueError as e:
        logger.error(f"Invalid choice format: {e}")
        return AnswerResult(
            answer_keys=[],
            token_ids=[],
            is_valid=False,
            method="chat",
            error=str(e),
            texts=[],
            prob={},
        )

    valid_keys = list(key_to_text.keys())
    logger.info(
        f"answer_multiple_choice_multi: model={resolved_model}, "
        f"{len(valid_keys)} keys, max_sel={max_selections}, "
        f"question='{question[:60]}...'"
    )

    tokenizer = get_tokenizer(resolved_model)
    system_prompt = _create_system_prompt(choices, max_selections)
    logit_bias, _ = _build_key_logit_bias(tokenizer, valid_keys)

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
        )
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        return AnswerResult(
            answer_keys=[],
            token_ids=[],
            is_valid=False,
            method="chat",
            error=str(e),
            texts=[],
            prob={},
        )

    raw_output = result.content.strip()
    logger.debug(f"Raw model output: '{raw_output}'")

    # Parse keys: take unique valid keys in order, stop at max_selections
    selected_keys: list[str] = []
    seen: set[str] = set()
    for line in raw_output.split("\n"):
        key = line.strip()
        if not key:
            continue
        if key in valid_keys and key not in seen:
            selected_keys.append(key)
            seen.add(key)
        elif key in seen:
            # Stop at first repeated key (model is looping)
            logger.debug(f"Stopping parse at repeated key '{key}'")
            break
        if max_selections and len(selected_keys) >= max_selections:
            break

    if not selected_keys:
        error_msg = f"No valid keys parsed from output: '{raw_output[:100]}'"
        logger.error(error_msg)
        return AnswerResult(
            answer_keys=[],
            token_ids=[],
            is_valid=False,
            method="chat",
            error=error_msg,
            texts=[],
            prob={},
        )

    selected_texts = [key_to_text[k] for k in selected_keys]
    token_ids = []
    for key in selected_keys:
        tokens = tokenizer.encode(key, add_special_tokens=False)
        token_ids.append(tokens[0] if tokens else -1)

    logger.info(f"Selected keys: {selected_keys}, texts: {selected_texts}")
    return AnswerResult(
        answer_keys=selected_keys,
        token_ids=token_ids,
        is_valid=True,
        method="chat",
        error=None,
        texts=selected_texts,
        prob={},
    )


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    test_cases = [
        {
            "question": "Which animals are mammals?",
            "choices": ["1) Dolphin", "2) Crocodile", "3) Python", "4) Whale"],
            "expected_keys": ["1", "4"],
        },
        {
            "question": "Which is a primary color?",
            "choices": ["A) Red", "B) Green", "C) Blue", "D) Orange"],
            "expected_keys": ["A", "C"],
        },
        {
            "question": "Which number is even?",
            "choices": ["X) 1", "Y) 2", "Z) 3"],
            "expected_keys": ["Y"],
        },
    ]

    console.print("\n[bold green]Multiple Selection Answer Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Question", style="cyan", max_width=35)
    table.add_column("Choices", style="white", max_width=30)
    table.add_column("Selected", justify="center", width=12)
    table.add_column("Expected", justify="center", width=12)
    table.add_column("Match", justify="center", width=6)

    for idx, case in enumerate(test_cases, start=1):
        result = answer_multiple_choice_multiple_selections(
            case["question"], case["choices"]
        )

        match = result["is_valid"] and sorted(result["answer_keys"]) == sorted(
            case["expected_keys"]
        )
        match_str = "[bold green]✓[/bold green]" if match else "[bold red]✗[/bold red]"
        sel_style = (
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
            ", ".join(c.split(")")[0] for c in case["choices"]),
            f"[{sel_style}]{', '.join(result['answer_keys']) or 'N/A'}[/{sel_style}]{err}",
            ", ".join(case["expected_keys"]),
            match_str,
        )

    console.print(table)
