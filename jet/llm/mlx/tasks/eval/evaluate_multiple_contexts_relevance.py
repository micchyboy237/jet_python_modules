import json
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
    """Raised when query or contexts are empty or invalid."""
    pass


class InvalidOutputError(Exception):
    """Raised when the generated output is not a valid score."""
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class ContextRelevanceResult(TypedDict):
    query: str
    context: str
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
    """Creates a system prompt for evaluating context relevance with few-shot examples."""
    examples = [
        {
            "query": "What is the capital of France?",
            "context": "The capital of France is Paris.",
            "score": "2"
        },
        {
            "query": "What is the capital of France?",
            "context": "Paris is a popular tourist destination.",
            "score": "1"
        },
        {
            "query": "What is the capital of France?",
            "context": "Einstein developed the theory of relativity.",
            "score": "0"
        }
    ]
    examples_str = "\n".join(
        f"Query: {ex['query']}\nContext: {ex['context']}\nScore: {ex['score']}"
        for ex in examples
    )
    return (
        f"You are an assistant that evaluates the relevance of a context to a query.\n"
        f"Respond strictly with a score of '0' (not relevant), '1' (somewhat relevant), or '2' (highly relevant).\n"
        f"\nExamples:\n{examples_str}\n\n"
        f"Given the query and context, provide the relevance score."
    )


def format_chat_messages(query: str, context: str, system_prompt: Optional[str] = None) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    user_content = f"Query: {query}\nContext: {context}\nScore:"
    return [
        {"role": "system", "content": system_prompt or create_system_prompt()},
        {"role": "user", "content": user_content}
    ]


def validate_inputs(query: str, contexts: List[str]) -> None:
    """Validates that query and contexts are non-empty."""
    if not query.strip():
        raise InvalidInputError("Query cannot be empty.")
    if not contexts:
        raise InvalidInputError("Contexts list cannot be empty.")
    for context in contexts:
        if not context.strip():
            raise InvalidInputError(
                f"Context cannot be empty for query: {query}")


def evaluate_multiple_contexts_relevance(
    query: str,
    contexts: List[str],
    model_path: LLMModelType,
    max_tokens: int = 1,
    temperature: float = 0.1,
    system_prompt: Optional[str] = None
) -> List[ContextRelevanceResult]:
    """Evaluates the relevance of multiple contexts for a single query."""
    try:
        validate_inputs(query, contexts)
        model_components = load_model_components(model_path)
        valid_outputs = ["0", "1", "2"]
        choice_token_map = {
            choice: model_components.tokenizer.encode(
                choice, add_special_tokens=False)
            for choice in valid_outputs
        }
        for choice, tokens in choice_token_map.items():
            logger.log(f"Token for '{choice}':",
                       tokens, colors=["GRAY", "ORANGE"])

        logit_bias = {
            tokens[0]: 0.0 for tokens in choice_token_map.values() if tokens
        }
        logit_bias.update(
            {i: -1e9 for i in range(model_components.tokenizer.vocab_size)
             if i not in [tokens[0] for tokens in choice_token_map.values()]}
        )
        logits_processors = [
            lambda tokens, logits: logits +
            mx.array([logit_bias.get(i, -1e9)
                     for i in range(logits.shape[-1])])
        ]
        sampler = mx.random.categorical
        stop_tokens = model_components.tokenizer.encode(
            "\n") + list(model_components.tokenizer.eos_token_ids)

        results = []
        for context in contexts:
            try:
                messages = format_chat_messages(query, context, system_prompt)
                formatted_prompt = model_components.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
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
                        raise InvalidOutputError(
                            "Generated token is a stop token.")
                    answer = model_components.tokenizer.decode([token]).strip()
                    break

                if answer not in valid_outputs:
                    raise InvalidOutputError(
                        f"Output '{answer}' is not a valid relevance score (0-2) for context: {context}")

                valid_token_ids = [choice_token_map[choice][0]
                                   for choice in valid_outputs]
                valid_logits = logits[valid_token_ids]
                probs = mx.softmax(valid_logits).tolist()
                prob_dict = {choice: round(prob, 4)
                             for choice, prob in zip(valid_outputs, probs)}
                logger.log(
                    f"Probabilities for:\nQuery: '{json.dumps(query)[:100]}'\nContext: '{json.dumps(context)[:100]}'",
                    f"\n{prob_dict}",
                    colors=["GRAY", "CYAN"]
                )

                results.append(ContextRelevanceResult(
                    query=query,
                    context=context,
                    relevance_score=int(answer),
                    is_valid=True,
                    error=None
                ))
            except Exception as e:
                logger.error(
                    f"Error processing context '{context[:100]}': {str(e)}")
                results.append(ContextRelevanceResult(
                    query=query,
                    context=context,
                    relevance_score=0,
                    is_valid=False,
                    error=str(e)
                ))
        return results
    except (ModelLoadError, InvalidInputError) as e:
        raise
    except Exception as e:
        logger.error(
            f"Error in evaluate_multiple_contexts_relevance: {str(e)}")
        raise
