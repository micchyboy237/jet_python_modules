from typing import List, Optional, Tuple
from typing import List, Dict, Optional, Tuple, TypedDict
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
    """Raised when question or choices are empty or invalid."""
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class AnswerResult(TypedDict):
    answer_keys: List[str]
    token_ids: List[int]
    is_valid: bool
    method: str
    error: Optional[str]
    texts: List[str]
    prob: Dict[str, float]  # Dictionary for per-choice probabilities


class ModelComponents:
    """Encapsulates model and tokenizer for easier management."""

    def __init__(self, model, tokenizer: TokenizerWrapper):
        self.model = model
        self.tokenizer = tokenizer
        self.choices = None


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


def validate_inputs(question: str, choices: List[str]) -> None:
    """Validates that question and choices are non-empty."""
    if not question.strip():
        raise InvalidInputError("Question cannot be empty.")
    if not choices:
        raise InvalidInputError("Choices cannot be empty.")


def parse_choices(choices: List[str]) -> tuple[Dict[str, str], List[str]]:
    """
    Parses choices into a dictionary mapping keys to choice texts and a list of choice texts.
    Supports flexible key formats (e.g., 'A)', '1)', 'A.', 'A:') using regex.
    """
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


def create_system_prompt(choices: List[str], max_selections: Optional[int] = None) -> str:
    """Creates a formatted system prompt with few-shot examples for multiple-choice selection."""
    instruction = (
        f"Select one or more options that best answer the question. "
        f"Return each selected option key on a new line (e.g., 'A\nB\nC') without additional text.\n\n"
        f"Examples:\n"
        f"Question: Which animals are mammals?\n"
        f"Options:\n1) Dolphin\n2) Crocodile\n3) Python\n4) Whale\n"
        f"Answer:\n1\n4\n\n"
        f"Question: Which is a primary color?\n"
        f"Options:\nA) Red\nB) Green\nC) Blue\nD) Orange\n"
        f"Answer:\nA\nC\n\n"
        f"Question: Which number is even?\n"
        f"Options:\nX) 1\nY) 2\nZ) 3\n"
        f"Answer:\nY\n\n"
    )
    if max_selections:
        instruction = (
            f"Select up to {max_selections} options that best answer the question. "
            f"Return each selected option key on a new line (e.g., 'A\nB\nC') without additional text.\n\n"
            f"Examples:\n"
            f"Question: Which animals are mammals? (Select up to 2)\n"
            f"Options:\n1) Dolphin\n2) Crocodile\n3) Python\n4) Whale\n"
            f"Answer:\n1\n4\n\n"
            f"Question: Which is a primary color? (Select up to 1)\n"
            f"Options:\nA) Red\nB) Green\nC) Blue\nD) Orange\n"
            f"Answer:\nA\n\n"
            f"Question: Which numbers are even? (Select up to 2)\n"
            f"Options:\nX) 1\nY) 2\nZ) 4\n"
            f"Answer:\nY\nZ\n\n"
        )
    return f"{instruction}Options:\n{'\n'.join(choices)}"


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
    stop_tokens = list(tokenizer.eos_token_ids)
    return logits_processors, sampler, stop_tokens


def compute_confidence_scores(
    model,
    input_ids: mx.array,
    choice_token_map: Dict[str, List[int]],
    selected_keys: List[str],
    key_to_choice: Dict[str, str]
) -> Dict[str, float]:
    """Computes normalized confidence scores for each selected choice."""
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
        confidence_scores = {}
        for key in selected_keys:
            text = key_to_choice.get(key, "")
            tokens = choice_token_map.get(text, [])
            if tokens:
                token_probs = [float(probs[token_id]) for token_id in tokens]
                confidence_scores[key] = sum(
                    token_probs) / len(token_probs) if token_probs else 0.0
                logger.debug(
                    f"Choice: {key} ({text}), Token IDs: {tokens}, Prob: {confidence_scores[key]}")
        total_prob = sum(confidence_scores.values())
        if total_prob == 0:
            logger.warning("Total probability is zero, returning raw scores")
            return confidence_scores
        normalized_confidence_scores = {
            key: prob / total_prob for key, prob in confidence_scores.items()
        }
        return normalized_confidence_scores
    except Exception as e:
        logger.error(f"Error computing confidence scores: {str(e)}")
        return {}


def generate_answer_step(
    model_components: 'ModelComponents',
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int],
    choice_texts: List[str],
    max_selections: Optional[int] = None,
    top_k: int = 5
) -> Tuple[List[str], List[int], List[int], List[mx.array], List[mx.array]]:
    """
    Generates multiple answers using generate_step method, logging probabilities and top-k indices.

    Args:
        model_components: Model components including tokenizer and model.
        formatted_prompt (str): The input prompt.
        max_tokens (int): Maximum number of tokens to generate.
        logits_processors (list): List of logits processor functions.
        sampler: Sampler function for token selection.
        stop_tokens (List[int]): List of token IDs that stop generation.
        choice_texts (List[str]): List of valid choice texts.
        max_selections (Optional[int]): Maximum number of selections to generate.
        top_k (int): Number of top probabilities to return indices for. Default: 5.

    Returns:
        Tuple[List[str], List[int], List[int], List[mx.array], List[mx.array]]:
            - Generated answers.
            - Token IDs.
            - Generated tokens.
            - Log probabilities for each token.
            - Top-k indices for each token's log probabilities.
    """
    generated_tokens = []
    token_ids = []
    logprobs_list = []
    top_k_indices_list = []
    input_ids = mx.array(model_components.tokenizer.encode(
        formatted_prompt, add_special_tokens=False))
    prompt_cache = None
    key_to_choice, _ = parse_choices(model_components.choices)
    valid_keys = list(key_to_choice.keys())

    for token, logprobs in generate_step(
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
        token_ids.append(token)
        logprobs_list.append(logprobs)
        top_k_indices = mx.argsort(
            logprobs, axis=-1)[::-1][:top_k]  # Fixed argsort
        top_k_indices_list.append(top_k_indices)
        answer = model_components.tokenizer.decode(generated_tokens).strip()
        current_answers = [a.strip() for a in answer.split("\n") if a.strip()]
        if max_selections and len(current_answers) >= max_selections:
            break

    return current_answers, token_ids, generated_tokens, logprobs_list, top_k_indices_list


def print_probs_and_indices(
    token_ids: List[int],
    logprobs_list: List[mx.array],
    top_k_indices_list: List[mx.array],
    tokenizer,
    token_mappings: dict
):
    """
    Prints log probabilities and top-k indices for each generated token.

    Args:
        token_ids (List[int]): List of generated token IDs.
        logprobs_list (List[mx.array]): List of log probabilities for each token.
        top_k_indices_list (List[mx.array]): List of top-k indices for each token.
        tokenizer: Tokenizer to decode token IDs.
        token_mappings (dict): Mapping of text to token IDs (e.g., {'Red': [6161], ...}).
    """
    # Reverse token mappings for lookup
    id_to_text = {}
    for text, tokens in token_mappings.items():
        for token in tokens:
            id_to_text[token] = text

    print("\nToken Probabilities and Top-k Indices:")
    for i, (token, logprobs, top_k_indices) in enumerate(zip(token_ids, logprobs_list, top_k_indices_list)):
        token_text = id_to_text.get(token, tokenizer.decode([token]))
        print(f"\nStep {i + 1}: Token = {token} ({token_text})")
        print(f"  Log Probabilities (first 5): {logprobs[:5].tolist()}")
        print(f"  Top-k Indices: {top_k_indices.tolist()}")
        # Print decoded tokens for top-k indices
        top_k_tokens = top_k_indices.tolist()
        top_k_texts = [id_to_text.get(idx, tokenizer.decode([idx]))
                       for idx in top_k_tokens]
        print(f"  Top-k Tokens: {top_k_texts}")
        # Print log probabilities for top-k tokens
        top_k_probs = logprobs[top_k_indices].tolist()
        print(f"  Top-k Log Probabilities: {top_k_probs}")


def validate_answer(answers: List[str], valid_keys: List[str]) -> None:
    """Validates that all generated answers are in the provided choice keys."""
    for answer in answers:
        if answer not in valid_keys:
            raise InvalidOutputError(
                f"Output '{answer}' is not one of the provided choice keys: {valid_keys}")


def answer_multiple_choice_multiple_selections(
    question: str,
    choices: List[str],
    model_path: LLMModelType,
    method: str = "generate_step",
    max_tokens: int = 20,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_selections: Optional[int] = None
) -> AnswerResult:
    validate_method(method)
    validate_inputs(question, choices)
    model_components = load_model_components(model_path)
    model_components.choices = choices
    key_to_choice, choice_texts = parse_choices(choices)
    logger.debug(f"Parsed choices: {key_to_choice}")
    system_prompt = create_system_prompt(choices, max_selections)
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
    answers, token_ids, generated_tokens, logprobs_list, top_k_indices_list = generate_answer_step(
        model_components, formatted_prompt, max_tokens,
        logits_processors, sampler, stop_tokens, choice_texts, max_selections
    )
    # Create complete token mappings from choice_token_map
    token_mappings = {}
    for text, tokens in choice_token_map.items():
        token_mappings[text] = tokens
    print_probs_and_indices(
        token_ids=token_ids,
        logprobs_list=logprobs_list,
        top_k_indices_list=top_k_indices_list,
        tokenizer=model_components.tokenizer,
        token_mappings=token_mappings
    )
    validate_answer(answers, list(key_to_choice.keys()))
    answer_texts = [key_to_choice.get(answer, "") for answer in answers]
    input_ids = mx.array(
        model_components.tokenizer.encode(formatted_prompt))
    confidence_scores = compute_confidence_scores(
        model_components.model, input_ids, choice_token_map, answers, key_to_choice
    )
    logger.debug(f"Normalized confidence scores: {confidence_scores}")
    logger.debug(f"Selected answers: {answers}, Texts: {answer_texts}")
    if not answers and confidence_scores:
        top_choices = sorted(
            confidence_scores, key=confidence_scores.get, reverse=True)
        answers = [
            key for key, text in key_to_choice.items()
            if text in top_choices[:max_selections or len(top_choices)]
        ]
        answer_texts = [
            text for text in top_choices[:max_selections or len(top_choices)]
        ]
        token_ids = [choice_token_map.get(text, [-1])[0]
                     for text in answer_texts]
    return AnswerResult(
        answer_keys=answers,
        token_ids=token_ids,
        is_valid=True,
        method=method,
        error=None,
        texts=answer_texts,
        prob=confidence_scores
    )
