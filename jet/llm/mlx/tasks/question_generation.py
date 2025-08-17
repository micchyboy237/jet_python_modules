from typing import List, Dict, Optional, TypedDict
from jet.models.model_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.token_utils import tokenize_strings
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate, generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import TokenizerWrapper

mx.random.seed(42)


class ModelLoadError(Exception):
    pass


class InvalidMethodError(Exception):
    pass


class PromptFormattingError(Exception):
    pass


class GenerationError(Exception):
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class QuestionGenerationResult(TypedDict):
    questions: str
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
    valid_methods = ["stream_generate", "generate_step"]
    if method not in valid_methods:
        raise InvalidMethodError(
            f"Invalid method specified: {method}. Valid methods: {valid_methods}")


def log_prompt_details(system_prompt: str, text: str, model_path: LLMModelType) -> None:
    """Logs system prompt, tokenized system prompt, and input text for debugging."""
    logger.gray("System:")
    logger.debug(system_prompt)
    logger.gray("Tokenized System:")
    logger.debug(tokenize_strings(system_prompt, model_path))
    logger.gray("Input Text:")
    logger.debug(text)
    logger.newline()


def format_chat_messages(system_prompt: str, text: str) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]


def setup_generation_parameters(
    tokenizer: TokenizerWrapper,
    temperature: float,
    top_p: float
) -> tuple:
    """Sets up logits processors, sampler, and stop tokens for generation."""
    logits_processors = make_logits_processors()
    sampler = make_sampler(temp=temperature, top_p=top_p)
    stop_tokens = tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
    return logits_processors, sampler, stop_tokens


def generate_questions_stream(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int]
) -> tuple[str, List[int]]:
    """Generates questions using stream_generate method."""
    questions = ""
    token_ids = []
    for output in stream_generate(
        model=model_components.model,
        tokenizer=model_components.tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        logits_processors=logits_processors,
        sampler=sampler
    ):
        if output.token in stop_tokens:
            break
        token_ids.append(output.token)
        questions = model_components.tokenizer.decode(token_ids)
    return questions, token_ids


def generate_questions_step(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int]
) -> tuple[str, List[int]]:
    """Generates questions using generate_step method."""
    questions = ""
    token_ids = []
    input_ids = mx.array(model_components.tokenizer.encode(
        formatted_prompt, add_special_tokens=False))
    prompt_cache = None
    for token, _ in generate_step(
        model=model_components.model,
        prompt=input_ids,
        max_tokens=max_tokens,
        logits_processors=logits_processors,
        sampler=sampler,
        prompt_cache=prompt_cache
    ):
        if token in stop_tokens:
            break
        token_ids.append(token)
        questions = model_components.tokenizer.decode(token_ids)
    return questions, token_ids


def question_generation(
    text: str,
    model_path: LLMModelType = "llama-3.2-3b-instruct-4bit",
    method: str = "stream_generate",
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
    system_prompt: str = "Generate a list of relevant questions based on the following text. Each question should be concise and directly related to the content."
) -> QuestionGenerationResult:
    """Generates questions based on the input text."""
    try:
        if not text.strip():
            raise PromptFormattingError("Text cannot be empty.")
        validate_method(method)
        model_components = load_model_components(model_path)
        log_prompt_details(system_prompt, text, model_path)
        messages = format_chat_messages(system_prompt, text)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        logits_processors, sampler, stop_tokens = setup_generation_parameters(
            model_components.tokenizer, temperature, top_p
        )
        if method == "stream_generate":
            questions, token_ids = generate_questions_stream(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens
            )
        else:
            questions, token_ids = generate_questions_step(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens
            )
        return QuestionGenerationResult(
            questions=questions.strip(),
            token_ids=token_ids,
            is_valid=True,
            method=method,
            error=None
        )
    except Exception as e:
        return QuestionGenerationResult(
            questions="",
            token_ids=[],
            is_valid=False,
            method=method,
            error=str(e)
        )
