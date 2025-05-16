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
    """Raised when query or contexts are empty or invalid."""
    pass


class InvalidOutputError(Exception):
    """Raised when the generated output is not a valid index."""
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class DocScore(TypedDict):
    doc_idx: int
    score: float


class SearchResult(TypedDict):
    results: List[DocScore]
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


def create_system_prompt(contexts: List[str]) -> str:
    """Creates a system prompt for selecting the most relevant context."""
    return (
        "Given a query and a list of contexts, evaluate the relevance of each context. "
        "Output the index (0-based) of the most relevant context without additional text.\n"
        f"Contexts:\n" +
        "\n".join(f"[{i}] {context}" for i, context in enumerate(contexts))
    )


def format_chat_messages(query: str, contexts: List[str]) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    system_prompt = create_system_prompt(contexts)
    user_content = f"Query: {query}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def validate_inputs(query: str, contexts: List[str], top_n: int) -> None:
    """Validates that query, contexts, and top_n are valid."""
    if not query.strip():
        raise InvalidInputError("Query cannot be empty.")
    if not contexts:
        raise InvalidInputError("Contexts cannot be empty.")
    if top_n < 1:
        raise InvalidInputError("top_n must be at least 1.")
    if top_n > len(contexts):
        raise InvalidInputError(
            f"top_n ({top_n}) cannot exceed number of contexts ({len(contexts)}).")
    for i, context in enumerate(contexts):
        if not context.strip():
            raise InvalidInputError(f"Context at index {i} cannot be empty.")


def compute_confidence_scores(
    model,
    input_ids: mx.array,
    choice_token_map: Dict[str, List[int]]
) -> Dict[str, float]:
    """Computes normalized confidence scores from model logits for each choice."""
    try:
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]
        logger.debug(f"Input IDs shape: {input_ids.shape}")
        model_output = model(input_ids)
        logger.debug(f"Model output shape: {model_output.shape}")
        if len(model_output.shape) != 3:
            raise ValueError(
                f"Unexpected model output shape: {model_output.shape}")
        logits = model_output[0, -1]
        probs = mx.softmax(logits, axis=-1)
        logger.debug(
            f"Softmax probabilities (min, max): {float(probs.min()), float(probs.max())}")
        raw_confidence_scores = {}
        for choice, tokens in choice_token_map.items():
            if tokens:
                token_probs = [float(probs[token_id]) for token_id in tokens]
                raw_confidence_scores[choice] = sum(
                    token_probs) / len(token_probs) if token_probs else 0.0
                logger.debug(
                    f"Choice: {choice}, Token IDs: {tokens}, Raw Prob: {raw_confidence_scores[choice]}")
        total_prob = sum(raw_confidence_scores.values())
        if total_prob == 0:
            logger.warning("Total probability is zero, returning raw scores")
            return raw_confidence_scores
        normalized_confidence_scores = {
            choice: prob / total_prob for choice, prob in raw_confidence_scores.items()
        }
        return normalized_confidence_scores
    except Exception as e:
        logger.error(f"Error computing confidence scores: {str(e)}")
        return {}


def search_contexts_by_index(
    query: str,
    contexts: List[str],
    model_path: ModelType,
    top_n: int = 1,
    max_tokens: int = 1,
    temperature: float = 0.1,
    top_p: float = 0.9
) -> SearchResult:
    """Searches contexts and returns the top N most relevant context indices with confidence scores."""
    try:
        validate_inputs(query, contexts, top_n)
        model_components = load_model_components(model_path)

        # Create valid outputs as string indices
        valid_outputs = [str(i) for i in range(len(contexts))]
        choice_token_map = {
            choice: model_components.tokenizer.encode(
                choice, add_special_tokens=False)
            for choice in valid_outputs
        }
        for choice, tokens in choice_token_map.items():
            logger.log(f"Token for index '{choice}':",
                       tokens, colors=["GRAY", "ORANGE"])

        # Setup generation parameters
        logit_bias = {
            tokens[0]: 0.0 for tokens in choice_token_map.values() if tokens}
        logits_processors = [
            lambda tokens, logits: logits + mx.array(
                [logit_bias.get(i, -1e9) for i in range(logits.shape[-1])]
            )
        ]
        sampler = mx.random.categorical
        stop_tokens = model_components.tokenizer.encode(
            "\n") + list(model_components.tokenizer.eos_token_ids)

        # Format and encode prompt
        messages = format_chat_messages(query, contexts)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = mx.array(model_components.tokenizer.encode(
            formatted_prompt, add_special_tokens=False))

        # Generate answer
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
            valid_token_ids = [choice_token_map[choice][0]
                               for choice in valid_outputs]
            valid_logits = logits[valid_token_ids]
            probs = mx.softmax(valid_logits).tolist()
            prob_dict = {choice: round(prob, 4)
                         for choice, prob in zip(valid_outputs, probs)}
            logger.log(
                f"Probabilities for query '{query}':",
                prob_dict,
                colors=["GRAY", "CYAN"]
            )
            answer = model_components.tokenizer.decode([token]).strip()
            break

        # Validate output
        if answer not in valid_outputs:
            raise InvalidOutputError(
                f"Output '{answer}' is not a valid context index (0-{len(contexts)-1})."
            )

        # Compute confidence scores for ranking
        confidence_scores = compute_confidence_scores(
            model_components.model, input_ids, choice_token_map
        )
        if not confidence_scores:
            logger.warning(
                "No confidence scores computed, returning generated answer only")
            return SearchResult(
                results=[{"doc_idx": int(answer), "score": 1.0}],
                is_valid=True,
                error=None
            )

        # Sort contexts by confidence score
        results = [
            {"doc_idx": int(choice), "score": score}
            for choice, score in sorted(
                confidence_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
        ]

        # Ensure generated answer is included in top_n if it's not already
        answer_index = int(answer)
        if not any(ctx["doc_idx"] == answer_index for ctx in results):
            results = results[:top_n-1] + [
                {"doc_idx": answer_index,
                    "score": confidence_scores.get(answer, 0.0)}
            ]

        logger.debug(f"Results: {results}")
        return SearchResult(
            results=results,
            is_valid=True,
            error=None
        )
    except Exception as e:
        logger.error(f"Error in search_contexts_by_index: {str(e)}")
        return SearchResult(
            results=[],
            is_valid=False,
            error=str(e)
        )
