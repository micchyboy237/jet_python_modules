import json
from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.tasks.utils import ModelComponents, load_model_components
from jet.logger import logger
import mlx.core as mx
from mlx_lm.generate import generate_step


class InvalidInputError(Exception):
    """Raised when query, context, or response is empty or invalid."""
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


def create_system_prompt() -> str:
    """Creates a system prompt for evaluating response relevance."""
    return (
        "You are an expert evaluator assessing the relevance of a response to a given query and context. "
        "Based on the query, context, and response provided, assign a relevance score as follows: "
        "0 (irrelevant: the response does not address the query or context), "
        "1 (partially relevant: the response addresses some aspects of the query or context but is incomplete or tangential), "
        "2 (highly relevant: the response directly and accurately addresses the query and context). "
        "Output only the score (0, 1, or 2) and nothing else."
    )


def format_chat_messages(query: str, context: str, response: str) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    user_content = f"Query: {query}\nContext: {context}\nResponse: {response}"
    return [
        {"role": "system", "content": create_system_prompt()},
        {"role": "user", "content": user_content}
    ]


def validate_inputs(query: str, context: str, response: str) -> None:
    """Validates that query, context, and response are non-empty."""
    if not query.strip():
        raise InvalidInputError("Query cannot be empty.")
    if not context.strip():
        raise InvalidInputError("Context cannot be empty.")
    if not response.strip():
        raise InvalidInputError("Response cannot be empty.")


def evaluate_response_relevance(
    query: str,
    context: str,
    response: str,
    model_path: LLMModelType | ModelComponents,
    max_tokens: int = 1,
    temperature: float = 0.1,
) -> RelevanceResult:
    """Evaluates if the response is relevant to the query and context."""
    try:
        validate_inputs(query, context, response)
        model_components = model_path if isinstance(
            model_path, ModelComponents) else load_model_components(model_path)
        valid_outputs = ["0", "1", "2"]
        choice_token_map = {choice: model_components.tokenizer.encode(
            choice, add_special_tokens=False) for choice in valid_outputs}
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
        messages = format_chat_messages(query, context, response)
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
                break
            valid_token_ids = [choice_token_map[choice][0]
                               for choice in valid_outputs]
            valid_logits = logits[valid_token_ids]
            probs = mx.softmax(valid_logits).tolist()
            prob_dict = {choice: round(prob, 4)
                         for choice, prob in zip(valid_outputs, probs)}
            logger.log(
                f"Probabilities for:\nQuery: '{json.dumps(query)[:100]}'\nContext: '{json.dumps(context)[:100]}'\nResponse: '{json.dumps(response)[:100]}'",
                f"\n{prob_dict}",
                colors=["GRAY", "CYAN"]
            )
            answer = model_components.tokenizer.decode([token]).strip()
            break
        if answer not in valid_outputs:
            raise InvalidOutputError(
                f"Output '{answer}' is not a valid relevance score (0-2).")
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
