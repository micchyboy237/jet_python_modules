from typing import List, Dict, Optional, TypedDict
from jet.llm.mlx.mlx_types import ModelType
from jet.llm.mlx.helpers.answer_multiple_choice import (
    ModelComponents, load_model_components, validate_method,
    create_system_prompt, log_prompt_details, format_chat_messages,
    encode_choices, validate_answer
)
from jet.logger import logger
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate, generate_step
from mlx_lm.sample_utils import make_logits_processors


class MultipleChoiceConfidenceResult(TypedDict):
    answer: str
    token_id: int
    is_valid: bool
    method: str
    error: Optional[str]
    confidence_scores: Dict[str, float]


def setup_confidence_generation_parameters(
    tokenizer,  # Assuming TokenizerWrapper type
    choice_token_map: Dict[str, List[int]],
    temperature: float
) -> tuple:
    """Sets up logits processor for confidence score calculation."""
    logit_bias = {tokens[0]: 0.0 for choice,
                  tokens in choice_token_map.items() if tokens}
    logits_processors = make_logits_processors(logit_bias=logit_bias)
    stop_tokens = tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
    return logits_processors, stop_tokens


def compute_confidence_scores(
    model,
    input_ids: mx.array,
    choice_token_map: Dict[str, List[int]]
) -> Dict[str, float]:
    """Computes confidence scores from model logits for each choice."""
    try:
        # Ensure input_ids has a batch dimension
        if len(input_ids.shape) == 1:
            input_ids = input_ids[None, :]  # Add batch dimension: (1, seq_len)

        # Log input shape for debugging
        logger.debug(f"Input IDs shape: {input_ids.shape}")

        # Get model output
        model_output = model(input_ids)

        # Log output shape for debugging
        logger.debug(f"Model output shape: {model_output.shape}")

        # Ensure output has expected shape (batch_size, sequence_length, vocab_size)
        if len(model_output.shape) != 3:
            raise ValueError(
                f"Unexpected model output shape: {model_output.shape}")

        # Get logits for the last token
        logits = model_output[0, -1]  # Shape: (vocab_size,)

        # Apply softmax to get probabilities
        probs = mx.softmax(logits, axis=-1)
        logger.debug(
            f"Softmax probabilities (min, max): {float(probs.min()), float(probs.max())}")

        confidence_scores = {}
        for choice, tokens in choice_token_map.items():
            if tokens:
                # For multi-token choices, average the probabilities
                token_probs = [float(probs[token_id]) for token_id in tokens]
                confidence_scores[choice] = sum(
                    token_probs) / len(token_probs) if token_probs else 0.0
                logger.debug(
                    f"Choice: {choice}, Token IDs: {tokens}, Prob: {confidence_scores[choice]}")

        return confidence_scores

    except Exception as e:
        logger.error(f"Error computing confidence scores: {str(e)}")
        return {}


def answer_multiple_choice_with_confidence(
    question: str,
    choices: List[str],
    model_path: ModelType,
    method: str = "stream_generate",
    max_tokens: int = 10,
    temperature: float = 0.1  # Adjusted from 0.0 to 0.1
) -> MultipleChoiceConfidenceResult:
    """
    Answers a multiple choice question and provides confidence scores for each choice.
    """
    try:
        validate_method(method)
        model_components = load_model_components(model_path)
        system_prompt = create_system_prompt(choices)
        log_prompt_details(system_prompt, question, model_path)
        messages = format_chat_messages(system_prompt, question)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Log formatted prompt for debugging
        logger.debug(f"Formatted prompt: {formatted_prompt}")
        choice_token_map = encode_choices(model_components.tokenizer, choices)
        logits_processors, stop_tokens = setup_confidence_generation_parameters(
            model_components.tokenizer, choice_token_map, temperature
        )

        if method == "stream_generate":
            answer, token_id, confidence_scores = generate_answer_with_confidence_stream(
                model_components, formatted_prompt, max_tokens,
                logits_processors, stop_tokens, choices, choice_token_map
            )
        else:
            answer, token_id, confidence_scores = generate_answer_with_confidence_step(
                model_components, formatted_prompt, max_tokens,
                logits_processors, stop_tokens, choices, choice_token_map
            )

        validate_answer(answer, choices)

        return MultipleChoiceConfidenceResult(
            answer=answer,
            token_id=token_id,
            is_valid=True,
            method=method,
            error=None,
            confidence_scores=confidence_scores
        )
    except Exception as e:
        logger.error(
            f"Error in answer_multiple_choice_with_confidence: {str(e)}")
        return MultipleChoiceConfidenceResult(
            answer="",
            token_id=-1,
            is_valid=False,
            method=method,
            error=str(e),
            confidence_scores={}
        )


def generate_answer_with_confidence_stream(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors,
    stop_tokens: List[int],
    choices: List[str],
    choice_token_map: Dict[str, List[int]]
) -> tuple[str, int, Dict[str, float]]:
    """Generates answer and confidence scores using stream_generate method."""
    answer = ""
    token_id = -1
    generated_tokens = []
    input_ids = mx.array(model_components.tokenizer.encode(formatted_prompt))

    # Ensure input_ids has batch dimension
    if len(input_ids.shape) == 1:
        input_ids = input_ids[None, :]

    # Compute confidence scores before generation
    confidence_scores = compute_confidence_scores(
        model_components.model, input_ids, choice_token_map
    )

    for output in stream_generate(
        model=model_components.model,
        tokenizer=model_components.tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        logits_processors=logits_processors
    ):
        if output.token in stop_tokens:
            break
        generated_tokens.append(output.token)
        token_id = output.token
        answer = model_components.tokenizer.decode(generated_tokens).strip()
        if answer in choices:
            break

    return answer, token_id, confidence_scores


def answer_multiple_choice_with_confidence(
    question: str,
    choices: List[str],
    model_path: ModelType,
    method: str = "stream_generate",
    max_tokens: int = 10,
    temperature: float = 0.0
) -> MultipleChoiceConfidenceResult:
    """
    Answers a multiple choice question and provides confidence scores for each choice.

    Args:
        question: The question to be answered
        choices: List of possible answer choices
        model_path: Path to the model
        method: Generation method ("stream_generate" or "generate_step")
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature

    Returns:
        MultipleChoiceConfidenceResult containing answer and confidence scores
    """
    try:
        validate_method(method)
        model_components = load_model_components(model_path)
        system_prompt = create_system_prompt(choices)
        log_prompt_details(system_prompt, question, model_path)
        messages = format_chat_messages(system_prompt, question)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        choice_token_map = encode_choices(model_components.tokenizer, choices)
        logits_processors, stop_tokens = setup_confidence_generation_parameters(
            model_components.tokenizer, choice_token_map, temperature
        )

        if method == "stream_generate":
            answer, token_id, confidence_scores = generate_answer_with_confidence_stream(
                model_components, formatted_prompt, max_tokens,
                logits_processors, stop_tokens, choices, choice_token_map
            )
        else:
            answer, token_id, confidence_scores = generate_answer_with_confidence_step(
                model_components, formatted_prompt, max_tokens,
                logits_processors, stop_tokens, choices, choice_token_map
            )

        validate_answer(answer, choices)

        return MultipleChoiceConfidenceResult(
            answer=answer,
            token_id=token_id,
            is_valid=True,
            method=method,
            error=None,
            confidence_scores=confidence_scores
        )
    except Exception as e:
        return MultipleChoiceConfidenceResult(
            answer="",
            token_id=-1,
            is_valid=False,
            method=method,
            error=str(e),
            confidence_scores={}
        )
