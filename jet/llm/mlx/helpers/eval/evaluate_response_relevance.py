from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.models import resolve_model
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import TokenizerWrapper
from mlx_lm.generate import generate_step


class ModelLoadError(Exception):
    """Raised when model or tokenizer loading fails."""
    pass


class InvalidInputError(Exception):
    """Raised when query or response is empty or invalid."""
    pass


class InvalidOutputError(Exception):
    """Raised when the generated output is not a valid score."""
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class RelevanceResult(TypedDict):
    relevance_score: int
    is_valid: bool
    error: Optional[str]


class ModelComponents:
    """Encapsulates model and tokenizer for easier management."""

    def __init__(self, model, tokenizer: TokenizerWrapper):
        self.model = model
        self.tokenizer = tokenizer


def load_model_components(model_path: ModelType) -> ModelComponents:
    """Loads model and tokenizer from the specified path."""
    try:
        model, tokenizer = load(resolve_model(model_path))
        return ModelComponents(model, tokenizer)
    except Exception as e:
        raise ModelLoadError(f"Error loading model or tokenizer: {e}")


def create_system_prompt() -> str:
    """Creates a system prompt for evaluating response relevance."""
    return """Evaluate if the provided response is relevant to the query. Choose one option based on how well the response addresses the query:
0: very low (response is completely unrelated or off-topic)
1: low (response has minimal relation to the query)
2: medium (response partially addresses the query)
3: high (response mostly addresses the query)
4: very high (response fully and directly addresses the query)

Return only the number (0, 1, 2, 3, or 4) without additional text."""


def format_chat_messages(query: str, response: str) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    user_content = f"Query: {query}\nResponse: {response}"
    return [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": user_content}
    ]


def validate_inputs(query: str, response: str) -> None:
    """Validates that query and response are non-empty."""
    if not query.strip():
        raise InvalidInputError("Query cannot be empty.")
    if not response.strip():
        raise InvalidInputError("Response cannot be empty.")


def evaluate_response_relevance(
    query: str,
    response: str,
    model_path: ModelType,
    max_tokens: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> RelevanceResult:
    """Evaluates if the LLM response is relevant to the query."""
    try:
        validate_inputs(query, response)
        model_components = load_model_components(model_path)

        # Define valid outputs
        valid_outputs = ["0", "1", "2", "3", "4"]
        choice_token_map = {choice: model_components.tokenizer.encode(
            choice, add_special_tokens=False) for choice in valid_outputs}
        # Log token map
        for choice, tokens in choice_token_map.items():
            logger.log(f"Token for '{choice}':",
                       tokens, colors=["GRAY", "ORANGE"])
        logit_bias = {
            tokens[0]: 0.0 for tokens in choice_token_map.values() if tokens}
        logits_processors = [lambda tokens, logits: logits +
                             mx.array([logit_bias.get(i, -1e9) for i in range(logits.shape[-1])])]
        sampler = mx.random.categorical
        stop_tokens = model_components.tokenizer.encode(
            "\n") + list(model_components.tokenizer.eos_token_ids)

        # Format prompt
        messages = format_chat_messages(query, response)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate answer
        input_ids = mx.array(model_components.tokenizer.encode(
            formatted_prompt, add_special_tokens=False))
        answer = ""
        for token, _ in generate_step(
            model=model_components.model,
            prompt=input_ids,
            max_tokens=max_tokens,
            logits_processors=logits_processors,
            sampler=lambda logits: sampler(
                logits / (temperature if temperature > 0 else 1.0)),
            prompt_cache=None
        ):
            if token in stop_tokens:
                break
            answer = model_components.tokenizer.decode([token]).strip()
            break

        # Validate answer
        if answer not in valid_outputs:
            raise InvalidOutputError(
                f"Output '{answer}' is not a valid relevance score (0-4).")

        return RelevanceResult(
            relevance_score=int(answer),
            is_valid=True,
            error=None
        )
    except Exception as e:
        logger.error(f"Error in evaluate_response_relevance: {str(e)}")
        return RelevanceResult(
            relevance_score=0,
            is_valid=False,
            error=str(e)
        )
