from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.token_utils import tokenize_strings
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import TokenizerWrapper

mx.random.seed(42)


class ModelLoadError(Exception):
    """Raised when model or tokenizer loading fails."""
    pass


class InvalidMethodError(Exception):
    """Raised when an invalid generation method is specified."""
    pass


class PromptFormattingError(Exception):
    """Raised when question or contexts are empty or invalid."""
    pass


class TokenEncodingError(Exception):
    """Raised when token encoding fails."""
    pass


class GenerationError(Exception):
    """Raised when generation fails during processing."""
    pass


class InvalidOutputError(Exception):
    """Raised when the generated answer is not 'Yes' or 'No'."""
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class AnswerResult(TypedDict):
    answers: List[str]
    token_ids: List[int]
    is_valid: bool
    method: str
    error: Optional[str]


class ModelComponents:
    """Encapsulates model and tokenizer for easier management."""

    def __init__(self, model, tokenizer: TokenizerWrapper):
        self.model = model
        self.tokenizer = tokenizer


def load_model_components(model_path: LLMModelType) -> ModelComponents:
    """Loads model and tokenizer from the specified path."""
    try:
        model, tokenizer = load(resolve_model(model_path))
        return ModelComponents(model, tokenizer)
    except Exception as e:
        raise ModelLoadError(f"Error loading model or tokenizer: {e}")


def validate_method(method: str) -> None:
    """Validates the generation method."""
    if method != "generate_step":
        raise InvalidMethodError(
            "Invalid method specified. Only 'generate_step' is supported.")


def validate_inputs(question: str, contexts: List[str]) -> None:
    """Validates that question and contexts are non-empty."""
    if not question.strip():
        raise PromptFormattingError("Question cannot be empty.")
    if not contexts:
        raise PromptFormattingError("Contexts cannot be empty.")
    for context in contexts:
        if not context.strip():
            raise PromptFormattingError("Context cannot be empty.")


def create_system_prompt(context: str) -> str:
    """Creates a system prompt for a single context."""
    return f"Answer the following question with only 'Yes' or 'No' based on the provided context. Ensure accuracy.\nContext: {context}"


def log_prompt_details(system_prompt: str, question: str, model_path: LLMModelType) -> None:
    """Logs system prompt, tokenized system prompt, and user question for debugging."""
    logger.gray("System:")
    logger.debug(system_prompt)
    logger.gray("Tokenized System:")
    logger.debug(tokenize_strings(system_prompt, model_path))
    logger.gray("User:")
    logger.debug(question)
    logger.newline()


def format_chat_messages(system_prompt: str, question: str) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]


def setup_generation_parameters(
    tokenizer: TokenizerWrapper,
    temperature: float,
    top_p: float
) -> tuple:
    """Sets up logit bias, logits processors, sampler, and stop tokens for generation."""
    yes_tokens = tokenizer.encode("Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode("No", add_special_tokens=False)
    yes_token = yes_tokens[0] if yes_tokens else None
    no_token = no_tokens[0] if no_tokens else None
    if yes_token is None or no_token is None:
        raise TokenEncodingError("Failed to encode 'Yes' or 'No' tokens.")
    logger.log("Token for 'Yes':", yes_tokens[0], colors=["GRAY", "ORANGE"])
    logger.log("Token for 'No':", no_tokens[0], colors=["GRAY", "ORANGE"])
    logit_bias = {
        yes_token: 0.0,
        no_token: 0.0,
        **{i: -1e9 for i in range(tokenizer.vocab_size) if i not in [yes_token, no_token]}
    }
    logits_processors = make_logits_processors(logit_bias=logit_bias)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    stop_tokens = tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
    return logits_processors, sampler, stop_tokens, yes_token, no_token


def generate_answer_step(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int]
) -> tuple[str, int]:
    """Generates an answer using generate_step method."""
    answer = ""
    token_id = -1
    input_ids = mx.array(model_components.tokenizer.encode(
        formatted_prompt, add_special_tokens=False))
    prompt_cache = None
    try:
        for token, _ in generate_step(
            model=model_components.model,
            prompt=input_ids,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            sampler=sampler,
            prompt_cache=prompt_cache
        ):
            if token in stop_tokens:
                raise InvalidOutputError("Generated token is a stop token.")
            answer = model_components.tokenizer.decode([token]).strip()
            token_id = token
            break
    except Exception as e:
        raise GenerationError(f"Error during generate_step: {e}")
    if answer.lower() not in ["yes", "no"]:
        raise InvalidOutputError(f"Output '{answer}' is not 'Yes' or 'No'.")
    return answer, token_id


def answer_multiple_yes_no_with_context(
    question: str,
    contexts: List[str],
    model_path: LLMModelType,
    method: str = "generate_step",
    max_tokens: int = 1,
    temperature: float = 0.1,
    top_p: float = 0.1
) -> AnswerResult:
    """Answers a yes/no question for multiple contexts, returning a list of answers."""
    # Allow specific exceptions to propagate for test purposes
    validate_method(method)
    validate_inputs(question, contexts)
    if max_tokens == 0 or max_tokens < -1:
        raise ValueError("Max tokens can only be -1 or a positive integer.")

    try:
        model_components = load_model_components(model_path)
        answers = []
        token_ids = []
        for context in contexts:
            system_prompt = create_system_prompt(context)
            log_prompt_details(system_prompt, question, model_path)
            messages = format_chat_messages(system_prompt, question)
            formatted_prompt = model_components.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            logits_processors, sampler, stop_tokens, _, _ = setup_generation_parameters(
                model_components.tokenizer, temperature, top_p
            )
            answer, token_id = generate_answer_step(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens
            )
            answers.append(answer)
            token_ids.append(token_id)
        return AnswerResult(
            answers=answers,
            token_ids=token_ids,
            is_valid=True,
            method=method,
            error=None
        )
    except (ModelLoadError, TokenEncodingError, GenerationError, InvalidOutputError) as e:
        logger.error(f"Error in answer_multiple_yes_no_with_context: {str(e)}")
        return AnswerResult(
            answers=[],
            token_ids=[],
            is_valid=False,
            method=method,
            error=str(e)
        )
