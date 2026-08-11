"""Answer multiple-choice questions with per-choice context.

Equivalent to jet/llm/mlx/tasks/answer_multiple_choice_with_context.py.
Each choice has an associated context string injected into the system prompt.
Uses llm_utils.chat with logit_bias for constrained single-answer selection.
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


CHOICE_PATTERN = re.compile(r"^\s*([a-zA-Z0-9]+)[\)\.\:]\s*(.+?)\s*$")
LOGIT_BIAS_VALUE = 5


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


def _create_system_prompt(choices: list[str], contexts: list[str]) -> str:
    formatted_options = []
    for choice, context in zip(choices, contexts):
        formatted_options.append(f"{choice}\nContext: {context}")
    return (
        "Answer the following question by choosing exactly ONE option. "
        "Return ONLY the exact text of your chosen option and nothing else.\n"
        f"Options:\n{'\\n'.join(formatted_options)}"
    )


def _build_choice_logit_bias(
    tokenizer, choice_texts: list[str]
) -> tuple[dict[str, int], dict[str, list[int]]]:
    bias: dict[str, int] = {}
    choice_token_map: dict[str, list[int]] = {}
    for text in choice_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        choice_token_map[text] = tokens
        if tokens:
            bias[str(tokens[0])] = LOGIT_BIAS_VALUE
            logger.debug(
                f"logit_bias: '{text[:40]}...' -> token {tokens[0]} (bias={LOGIT_BIAS_VALUE})"
            )
    return bias, choice_token_map


def answer_multiple_choice_with_context(
    question: str,
    choices: list[str],
    contexts: list[str],
    model: str | None = None,
    max_tokens: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> AnswerResult:
    """Answer a multiple-choice question with per-choice context."""
    resolved_model = model or LLM_MODEL

    if not question.strip():
        return AnswerResult(
            answer_key="",
            token_id=-1,
            is_valid=False,
            method="chat",
            error="Question cannot be empty.",
        )
    if not choices:
        return AnswerResult(
            answer_key="",
            token_id=-1,
            is_valid=False,
            method="chat",
            error="Choices cannot be empty.",
        )
    if not contexts:
        return AnswerResult(
            answer_key="",
            token_id=-1,
            is_valid=False,
            method="chat",
            error="Contexts cannot be empty.",
        )
    if len(choices) != len(contexts):
        return AnswerResult(
            answer_key="",
            token_id=-1,
            is_valid=False,
            method="chat",
            error=f"Choices ({len(choices)}) and contexts ({len(contexts)}) count mismatch.",
        )

    try:
        key_to_text, choice_texts = _parse_choices(choices)
    except ValueError as e:
        logger.error(f"Invalid choice format: {e}")
        return AnswerResult(
            answer_key="", token_id=-1, is_valid=False, method="chat", error=str(e)
        )

    logger.info(
        f"answer_mc_with_context: model={resolved_model}, "
        f"{len(choices)} choices, question='{question[:60]}...'"
    )

    tokenizer = get_tokenizer(resolved_model)
    system_prompt = _create_system_prompt(choices, contexts)
    logit_bias, choice_token_map = _build_choice_logit_bias(tokenizer, choice_texts)

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
            answer_key="", token_id=-1, is_valid=False, method="chat", error=str(e)
        )

    answer = result.content.strip()
    logger.debug(f"Raw model output: '{answer}'")

    matched_text = None
    for text in choice_texts:
        if answer == text or text.startswith(answer):
            matched_text = text
            break

    if matched_text is None:
        error_msg = f"Output '{answer}' is not one of the provided choices."
        logger.error(error_msg)
        return AnswerResult(
            answer_key="", token_id=-1, is_valid=False, method="chat", error=error_msg
        )

    answer_key = next((k for k, v in key_to_text.items() if v == matched_text), "")
    tokens = choice_token_map.get(matched_text, [])
    token_id = tokens[0] if tokens else -1

    logger.info(f"Selected answer: key='{answer_key}', text='{matched_text}'")
    return AnswerResult(
        answer_key=answer_key,
        token_id=token_id,
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
            "question": "Which city is the capital of France?",
            "choices": ["A) London", "B) Paris", "C) Berlin"],
            "contexts": [
                "London is the capital of the United Kingdom.",
                "Paris is located in northern France and serves as its capital.",
                "Berlin is the capital of Germany.",
            ],
            "expected_key": "B",
        },
        {
            "question": "Which animal lives in water?",
            "choices": ["1) Eagle", "2) Dolphin", "3) Lion"],
            "contexts": [
                "Eagles are birds of prey that fly in the sky.",
                "Dolphins are marine mammals that live in oceans and rivers.",
                "Lions are large cats that live on the African savanna.",
            ],
            "expected_key": "2",
        },
    ]

    console.print("\n[bold green]Multiple Choice with Context Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", justify="center", style="dim", width=3)
    table.add_column("Question", style="cyan", max_width=35)
    table.add_column("Answer Key", justify="center", width=10)
    table.add_column("Answer Text", style="white", max_width=30)
    table.add_column("Expected", justify="center", width=10)
    table.add_column("Match", justify="center", width=6)

    for idx, case in enumerate(test_cases, start=1):
        result = answer_multiple_choice_with_context(
            case["question"], case["choices"], case["contexts"]
        )

        match = result["is_valid"] and result["answer_key"] == case["expected_key"]
        match_str = "[bold green]✓[/bold green]" if match else "[bold red]✗[/bold red]"
        ans_style = (
            "bold green" if match else ("yellow" if result["is_valid"] else "dim red")
        )

        try:
            kt, _ = _parse_choices(case["choices"])
            ans_text = kt.get(result["answer_key"], "N/A")
        except Exception:
            ans_text = "N/A"

        err = (
            f"\n[dim red]⚠ {result['error']}[/dim red]"
            if not result["is_valid"]
            else ""
        )

        table.add_row(
            str(idx),
            case["question"],
            f"[{ans_style}]{result['answer_key'] or 'N/A'}[/{ans_style}]",
            f"{ans_text}{err}",
            case["expected_key"],
            match_str,
        )

    console.print(table)
