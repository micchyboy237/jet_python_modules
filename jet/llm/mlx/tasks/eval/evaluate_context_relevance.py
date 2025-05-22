from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.mlx_types import LLMModelType
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
    """Raised when query or context is empty or invalid."""
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


def load_model_components(model_path: LLMModelType) -> ModelComponents:
    """Loads model and tokenizer from the specified path."""
    try:
        model, tokenizer = load(resolve_model(model_path))
        return ModelComponents(model, tokenizer)
    except Exception as e:
        raise ModelLoadError(f"Error loading model or tokenizer: {e}")


def create_system_prompt() -> str:
    """Creates a system prompt for evaluating context relevance."""
    return """Evaluate if the provided context is relevant to the query. Choose one option based on how well the context addresses the query:
0: Low relevance (context is unrelated or barely related to the query)
1: Medium relevance (context partially addresses the query)
2: High relevance (context directly and mostly addresses the query)

Examples:
- Query: "What is the capital of France?"
  - Context: "The theory of relativity was developed by Albert Einstein." -> 0 (completely unrelated)
  - Context: "Paris hosts many tourists in France." -> 1 (mentions Paris but not as capital)
  - Context: "The capital of France is Paris." -> 2 (direct and complete)

Return only the number (0, 1, or 2) without additional text."""


def format_chat_messages(query: str, context: str) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    user_content = f"Query: {query}\nContext: {context}"
    return [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": user_content}
    ]


def validate_inputs(query: str, context: str) -> None:
    """Validates that query and context are non-empty."""
    if not query.strip():
        raise InvalidInputError("Query cannot be empty.")
    if not context.strip():
        raise InvalidInputError("Context cannot be empty.")


def evaluate_context_relevance(
    query: str,
    context: str,
    model_path: LLMModelType,
    max_tokens: int = 1,
    temperature: float = 0.1,
) -> RelevanceResult:
    """Evaluates if the retrieved context is relevant to the query."""
    try:
        validate_inputs(query, context)
        model_components = load_model_components(model_path)

        # Define valid outputs
        valid_outputs = ["0", "1", "2"]
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
        messages = format_chat_messages(query, context)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate answer and capture logits
        input_ids = mx.array(model_components.tokenizer.encode(
            formatted_prompt, add_special_tokens=False))
        answer = ""
        for token, logits in generate_step(
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
            # Compute softmax probabilities for valid scores
            valid_token_ids = [choice_token_map[choice][0]
                               for choice in valid_outputs]
            valid_logits = logits[valid_token_ids]
            probs = mx.softmax(valid_logits).tolist()
            prob_dict = {choice: round(prob, 4)
                         for choice, prob in zip(valid_outputs, probs)}
            logger.log(
                f"Probabilities for query '{query}' and context '{context}':",
                prob_dict,
                colors=["GRAY", "CYAN"]
            )
            answer = model_components.tokenizer.decode([token]).strip()
            break

        # Validate answer
        if answer not in valid_outputs:
            raise InvalidOutputError(
                f"Output '{answer}' is not a valid relevance score (0-2).")

        return RelevanceResult(
            relevance_score=int(answer),
            is_valid=True,
            error=None
        )
    except Exception as e:
        logger.error(f"Error in evaluate_context_relevance: {str(e)}")
        return RelevanceResult(
            relevance_score=0,
            is_valid=False,
            error=str(e)
        )
