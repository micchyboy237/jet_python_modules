from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate, generate_step
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import TokenizerWrapper

mx.random.seed(42)

# Custom exceptions


class ModelLoadError(Exception):
    pass


class InvalidMethodError(Exception):
    pass


class PromptFormattingError(Exception):
    pass


class GenerationError(Exception):
    pass


class InvalidOutputError(Exception):
    pass

# Type definitions


class ChatMessage(TypedDict):
    role: str
    content: str


class TextToSQLResult(TypedDict):
    sql_query: str
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


def create_system_prompt(schema: str) -> str:
    """Creates a formatted system prompt for text-to-SQL conversion."""
    return (
        f"You are a SQL expert. Convert the given natural language query into a valid SQL query based on the provided database schema. "
        f"Provide only the SQL query without any additional explanation.\n\nSchema:\n{schema}"
    )


def log_prompt_details(system_prompt: str, input_text: str, model_path: LLMModelType) -> None:
    """Logs system prompt and input text for debugging."""
    logger.gray("System:")
    logger.debug(system_prompt)
    logger.gray("User:")
    logger.debug(input_text)
    logger.newline()


def format_chat_messages(system_prompt: str, input_text: str) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text}
    ]


def generate_response_stream(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int]
) -> str:
    """Generates a response using stream_generate method."""
    response = ""
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
        response += output.text
    return response.strip()


def generate_response_step(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int]
) -> str:
    """Generates a response using generate_step method."""
    response = ""
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
        response += model_components.tokenizer.decode([token])
    return response.strip()


def validate_response(response: str) -> None:
    """Validates that the response is a non-empty SQL query."""
    if not response.strip():
        raise InvalidOutputError("SQL query is empty.")
    if not response.strip().upper().startswith("SELECT"):
        raise InvalidOutputError(
            "Generated query is not a valid SQL SELECT query.")


def text_to_sql(
    input_text: str,
    schema: str,
    model_path: LLMModelType,
    method: str = "stream_generate",
    max_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> TextToSQLResult:
    """
    Converts a natural language query into a SQL query based on the provided schema.

    Args:
        input_text (str): The natural language query to convert.
        schema (str): The database schema to base the SQL query on.
        model_path (LLMModelType): Path to the model or model identifier.
        method (str): Generation method ("stream_generate" or "generate_step").
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling parameter.

    Returns:
        TextToSQLResult: Dictionary containing the SQL query and metadata.
    """
    try:
        if not input_text.strip():
            raise PromptFormattingError("Input text cannot be empty.")
        if not schema.strip():
            raise PromptFormattingError("Schema cannot be empty.")
        if max_tokens <= 0:
            raise ValueError("Max tokens must be a positive integer.")

        validate_method(method)
        model_components = load_model_components(model_path)
        system_prompt = create_system_prompt(schema)
        log_prompt_details(system_prompt, input_text, model_path)

        messages = format_chat_messages(system_prompt, input_text)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        logits_processors = make_logits_processors()
        sampler = make_sampler(temp=temperature, top_p=top_p)
        stop_tokens = model_components.tokenizer.encode(
            "\n\n") + list(model_components.tokenizer.eos_token_ids)

        if method == "stream_generate":
            response = generate_response_stream(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens
            )
        else:
            response = generate_response_step(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens
            )

        validate_response(response)

        return TextToSQLResult(
            sql_query=response,
            is_valid=True,
            method=method,
            error=None
        )
    except Exception as e:
        return TextToSQLResult(
            sql_query="",
            is_valid=False,
            method=method,
            error=str(e)
        )
