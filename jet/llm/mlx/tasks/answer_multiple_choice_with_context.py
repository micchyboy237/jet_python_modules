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
import re

mx.random.seed(42)


class ModelLoadError(Exception):
    """Raised when model or tokenizer loading fails."""
    pass


class InvalidMethodError(Exception):
    """Raised when an invalid generation method is specified."""
    pass


class InvalidOutputError(Exception):
    """Raised when the generated answer is not in the provided choices."""
    pass


class InvalidChoiceFormatError(Exception):
    """Raised when a choice does not match the expected format."""
    pass


class InvalidInputError(Exception):
    """Raised when question, choices, or contexts are empty or invalid."""
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class AnswerResult(TypedDict):
    answer_key: str
    token_id: int
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
    valid_methods = ["generate_step"]
    if method not in valid_methods:
        raise InvalidMethodError(
            f"Invalid method specified: {method}. Valid methods: {valid_methods}")


def validate_inputs(question: str, choices: List[str], contexts: List[str]) -> None:
    """Validates that question, choices, and contexts are non-empty and have matching lengths."""
    if not question.strip():
        raise InvalidInputError("Question cannot be empty.")
    if not choices:
        raise InvalidInputError("Choices cannot be empty.")
    if not contexts:
        raise InvalidInputError("Contexts cannot be empty.")
    if len(choices) != len(contexts):
        raise InvalidInputError(
            f"Number of choices ({len(choices)}) must match number of contexts ({len(contexts)}).")


def parse_choices(choices: List[str]) -> tuple[Dict[str, str], List[str]]:
    """Parses choices into a key-to-choice dictionary and a list of choice texts."""
    key_to_choice = {}
    choice_texts = []
    pattern = re.compile(r'^\s*([a-zA-Z0-9]+)[\)\.\:]\s*(.+?)\s*$')
    for choice in choices:
        if not choice.strip():
            raise InvalidChoiceFormatError(
                f"Choice '{choice}' is empty or invalid")
        match = pattern.match(choice)
        if not match:
            raise InvalidChoiceFormatError(
                f"Choice '{choice}' does not match expected format (e.g., 'A) Text', '1) Text', 'A. Text', 'A: Text')")
        key, text = match.groups()
        if not key or not text.strip():
            raise InvalidChoiceFormatError(
                f"Choice '{choice}' has empty key or text")
        if key in key_to_choice:
            raise InvalidChoiceFormatError(
                f"Duplicate key '{key}' found in choices")
        key_to_choice[key] = text.strip()
        choice_texts.append(text.strip())
    return key_to_choice, choice_texts


def create_system_prompt(choices: List[str], contexts: List[str]) -> str:
    """Creates a formatted system prompt with choices and their corresponding contexts."""
    formatted_options = []
    for choice, context in zip(choices, contexts):
        formatted_options.append(f"{choice}\nContext: {context}")
    return f"Answer the following question by choosing one of the options provided without any additional text.\nOptions:\n{'\n'.join(formatted_options)}"


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


def encode_choices(tokenizer: TokenizerWrapper, choice_texts: List[str]) -> Dict[str, List[int]]:
    """Encodes each choice text into tokens and logs the results."""
    choice_token_map = {}
    for choice in choice_texts:
        tokens = tokenizer.encode(choice, add_special_tokens=False)
        choice_token_map[choice] = tokens
        logger.log(f"Tokens for '{choice}':",
                   tokens, colors=["GRAY", "ORANGE"])
    return choice_token_map


def setup_generation_parameters(
    tokenizer: TokenizerWrapper,
    choice_token_map: Dict[str, List[int]],
    temperature: float,
    top_p: float
) -> tuple:
    """Sets up logit bias, logits processors, sampler, and stop tokens for generation."""
    logit_bias = {tokens[0]: 0.0 for choice,
                  tokens in choice_token_map.items() if tokens}
    logits_processors = make_logits_processors(logit_bias=logit_bias)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    stop_tokens = tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
    return logits_processors, sampler, stop_tokens


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


def generate_answer_step(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int],
    choice_texts: List[str]
) -> tuple[str, int, List[int]]:
    """Generates an answer using generate_step method."""
    answer = ""
    token_id = -1
    generated_tokens = []
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
        generated_tokens.append(token)
        token_id = token
        answer = model_components.tokenizer.decode(generated_tokens).strip()
        if answer in choice_texts:
            break
    return answer, token_id, generated_tokens


def validate_answer(answer: str, choice_texts: List[str]) -> None:
    """Validates that the generated answer is one of the provided choice texts."""
    if answer not in choice_texts:
        raise InvalidOutputError(
            f"Output '{answer}' is not one of the provided choices: {choice_texts}")


def answer_multiple_choice_with_context(
    question: str,
    choices: List[str],
    contexts: List[str],
    model_path: LLMModelType,
    method: str = "generate_step",
    max_tokens: int = 10,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> AnswerResult:
    """Answers a multiple-choice question with provided contexts for each choice."""
    try:
        validate_method(method)
        validate_inputs(question, choices, contexts)
        model_components = load_model_components(model_path)
        key_to_choice, choice_texts = parse_choices(choices)
        logger.debug(f"Parsed choices: {key_to_choice}")
        system_prompt = create_system_prompt(choices, contexts)
        log_prompt_details(system_prompt, question, model_path)
        messages = format_chat_messages(system_prompt, question)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        choice_token_map = encode_choices(
            model_components.tokenizer, choice_texts)
        logits_processors, sampler, stop_tokens = setup_generation_parameters(
            model_components.tokenizer, choice_token_map, temperature, top_p
        )
        answer, token_id, _ = generate_answer_step(
            model_components, formatted_prompt, max_tokens,
            logits_processors, sampler, stop_tokens, choice_texts
        )
        input_ids = mx.array(
            model_components.tokenizer.encode(formatted_prompt))
        confidence_scores = compute_confidence_scores(
            model_components.model, input_ids, choice_token_map
        )
        if confidence_scores:
            most_confident_choice = max(
                confidence_scores, key=confidence_scores.get)
            logger.debug(f"Normalized confidence scores: {confidence_scores}")
            logger.debug(
                f"Most confident choice: {most_confident_choice} ({confidence_scores[most_confident_choice]})")
            if answer != most_confident_choice:
                logger.warning(
                    f"Generated answer '{answer}' differs from most confident choice '{most_confident_choice}'. Overriding.")
                answer = most_confident_choice
                token_id = choice_token_map[most_confident_choice][0] if choice_token_map[most_confident_choice] else -1
        validate_answer(answer, choice_texts)
        answer_key = next(
            key for key, text in key_to_choice.items() if text == answer)
        logger.debug(f"Selected answer: {answer}, Key: {answer_key}")
        return AnswerResult(
            answer_key=answer_key,
            token_id=token_id,
            is_valid=True,
            method=method,
            error=None
        )
    except Exception as e:
        logger.error(f"Error in answer_multiple_choice_with_context: {str(e)}")
        return AnswerResult(
            answer_key="",
            token_id=-1,
            is_valid=False,
            method=method,
            error=str(e)
        )
