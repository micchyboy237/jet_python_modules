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


class InvalidOutputError(Exception):
    pass


class ChatMessage(TypedDict):
    role: str
    content: str


class EntailmentResult(TypedDict):
    label: str
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
    valid_methods = ["stream_generate", "generate_step"]
    if method not in valid_methods:
        raise InvalidMethodError(
            f"Invalid method specified: {method}. Valid methods: {valid_methods}")


def create_system_prompt(labels: List[str]) -> str:
    """Creates a formatted system prompt with the given entailment labels."""
    return f"Determine the relationship between the premise and hypothesis by choosing one of the options provided without any additional text.\nOptions:\n{'\n'.join(labels)}"


def log_prompt_details(system_prompt: str, premise: str, hypothesis: str, model_path: LLMModelType) -> None:
    """Logs system prompt, tokenized system prompt, premise, and hypothesis for debugging."""
    logger.gray("System:")
    logger.debug(system_prompt)
    logger.gray("Tokenized System:")
    logger.debug(tokenize_strings(system_prompt, model_path))
    logger.gray("Premise:")
    logger.debug(premise)
    logger.gray("Hypothesis:")
    logger.debug(hypothesis)
    logger.newline()


def format_chat_messages(system_prompt: str, premise: str, hypothesis: str) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    user_content = f"Premise: {premise}\nHypothesis: {hypothesis}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def encode_labels(tokenizer: TokenizerWrapper, labels: List[str]) -> Dict[str, List[int]]:
    """Encodes each label into tokens and logs the results."""
    label_token_map = {}
    for label in labels:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        label_token_map[label] = tokens
        logger.log(f"Tokens for '{label}':",
                   tokens, colors=["GRAY", "ORANGE"])
    return label_token_map


def setup_generation_parameters(
    tokenizer: TokenizerWrapper,
    label_token_map: Dict[str, List[int]],
    temperature: float,
    top_p: float
) -> tuple:
    """Sets up logit bias, logits processors, sampler, and stop tokens for generation."""
    logit_bias = {tokens[0]: 0.0 for label,
                  tokens in label_token_map.items() if tokens}
    logits_processors = make_logits_processors(logit_bias=logit_bias)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    stop_tokens = tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
    return logits_processors, sampler, stop_tokens


def generate_label_stream(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int],
    labels: List[str]
) -> tuple[str, int, List[int]]:
    """Generates a label using stream_generate method."""
    label = ""
    token_id = -1
    generated_tokens = []
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
        generated_tokens.append(output.token)
        token_id = output.token
        label = model_components.tokenizer.decode(generated_tokens)
        if label in labels:
            break
    return label, token_id, generated_tokens


def generate_label_step(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int],
    labels: List[str]
) -> tuple[str, int, List[int]]:
    """Generates a label using generate_step method."""
    label = ""
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
        label = model_components.tokenizer.decode(generated_tokens)
        if label in labels:
            break
    return label, token_id, generated_tokens


def validate_label(label: str, labels: List[str]) -> None:
    """Validates that the generated label is one of the provided options."""
    if label not in labels:
        raise InvalidOutputError(
            f"Output '{label}' is not one of the provided labels.")


def text_entailment(
    premise: str,
    hypothesis: str,
    labels: List[str] = ["Entailment", "Contradiction", "Neutral"],
    model_path: LLMModelType = "llama-3.2-3b-instruct-4bit",
    method: str = "stream_generate",
    max_tokens: int = 10,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> EntailmentResult:
    """Evaluates the entailment relationship between a premise and hypothesis."""
    try:
        validate_method(method)
        model_components = load_model_components(model_path)
        system_prompt = create_system_prompt(labels)
        log_prompt_details(system_prompt, premise, hypothesis, model_path)
        messages = format_chat_messages(system_prompt, premise, hypothesis)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        label_token_map = encode_labels(model_components.tokenizer, labels)
        logits_processors, sampler, stop_tokens = setup_generation_parameters(
            model_components.tokenizer, label_token_map, temperature, top_p
        )
        if method == "stream_generate":
            label, token_id, _ = generate_label_stream(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens, labels
            )
        else:
            label, token_id, _ = generate_label_step(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens, labels
            )
        validate_label(label, labels)
        return EntailmentResult(
            label=label,
            token_id=token_id,
            is_valid=True,
            method=method,
            error=None
        )
    except Exception as e:
        return EntailmentResult(
            label="",
            token_id=-1,
            is_valid=False,
            method=method,
            error=str(e)
        )
