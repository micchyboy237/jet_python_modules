from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.helpers.answer_multiple_choice import (
    ModelComponents, load_model_components, validate_method,
    log_prompt_details, format_chat_messages, encode_choices,
    setup_generation_parameters, generate_answer_stream, generate_answer_step
)
from jet.logger import logger


class AnswerResult(TypedDict):
    answer: str
    token_id: int
    is_valid: bool
    method: str
    error: Optional[str]


def create_labeled_system_prompt(choices: List[str]) -> str:
    """Creates a formatted system prompt with labeled choices (e.g., A), B), C))."""
    labeled_choices = [f"{chr(65+i)}) {choice}" for i,
                       choice in enumerate(choices)]
    return f"Answer the following question by choosing one of the labeled options provided (e.g., 'A', 'B'). Return only the letter of the chosen option.\nOptions:\n{'\n'.join(labeled_choices)}"


def validate_labeled_answer(answer: str, num_choices: int) -> None:
    """Validates that the answer is a single letter corresponding to a choice."""
    valid_letters = [chr(65+i) for i in range(num_choices)]
    if answer not in valid_letters:
        raise ValueError(f"Output '{answer}' is not a valid option letter.")


def create_choice_token_map(tokenizer, choices: List[str]) -> Dict[str, List[int]]:
    """Encodes each choice label (A, B, etc.) into tokens."""
    choice_token_map = {}
    for i in range(len(choices)):
        label = chr(65+i)  # A, B, C, etc.
        tokens = tokenizer.encode(label, add_special_tokens=False)
        choice_token_map[label] = tokens
        logger.log(f"Tokens for '{label}':", tokens, colors=["GRAY", "ORANGE"])
    return choice_token_map


def answer_multiple_choice_with_key(
    question: str,
    choices: List[str],
    model_path: ModelType,
    method: str = "stream_generate",
    max_tokens: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> AnswerResult:
    """
    Answers a multiple choice question with labeled choices (e.g., A), B)).

    Args:
        question: The question to be answered
        choices: List of possible answer choices
        model_path: Path to the model
        method: Generation method ("stream_generate" or "generate_step")
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        AnswerResult containing the selected option letter and metadata
    """
    try:
        validate_method(method)
        model_components = load_model_components(model_path)
        system_prompt = create_labeled_system_prompt(choices)
        log_prompt_details(system_prompt, question, model_path)
        messages = format_chat_messages(system_prompt, question)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        choice_token_map = create_choice_token_map(
            model_components.tokenizer, choices)
        logits_processors, sampler, stop_tokens = setup_generation_parameters(
            model_components.tokenizer, choice_token_map, temperature, top_p
        )

        valid_letters = [chr(65+i) for i in range(len(choices))]
        if method == "stream_generate":
            answer, token_id, _ = generate_answer_stream(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens, valid_letters
            )
        else:
            answer, token_id, _ = generate_answer_step(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens, valid_letters
            )

        validate_labeled_answer(answer, len(choices))

        return AnswerResult(
            answer=answer,
            token_id=token_id,
            is_valid=True,
            method=method,
            error=None
        )
    except Exception as e:
        return AnswerResult(
            answer="",
            token_id=-1,
            is_valid=False,
            method=method,
            error=str(e)
        )
