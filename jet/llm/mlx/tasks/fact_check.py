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


class FactCheckResult(TypedDict):
    verdict: str
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


def create_system_prompt(verdicts: List[str]) -> str:
    """Creates a formatted system prompt with the given verdict options."""
    return f"Evaluate the truthfulness of the following statement by choosing one of the options provided without any additional text.\nOptions:\n{'\n'.join(verdicts)}"


def log_prompt_details(system_prompt: str, statement: str, model_path: LLMModelType) -> None:
    """Logs system prompt, tokenized system prompt, and input statement for debugging."""
    logger.gray("System:")
    logger.debug(system_prompt)
    logger.gray("Tokenized System:")
    logger.debug(tokenize_strings(system_prompt, model_path))
    logger.gray("Statement:")
    logger.debug(statement)
    logger.newline()


def format_chat_messages(system_prompt: str, statement: str) -> List[ChatMessage]:
    """Formats the system and user messages for the chat template."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": statement}
    ]


def encode_verdicts(tokenizer: TokenizerWrapper, verdicts: List[str]) -> Dict[str, List[int]]:
    """Encodes each verdict into tokens and logs the results."""
    verdict_token_map = {}
    for verdict in verdicts:
        tokens = tokenizer.encode(verdict, add_special_tokens=False)
        verdict_token_map[verdict] = tokens
        logger.log(f"Tokens for '{verdict}':",
                   tokens, colors=["GRAY", "ORANGE"])
    return verdict_token_map


def setup_generation_parameters(
    tokenizer: TokenizerWrapper,
    verdict_token_map: Dict[str, List[int]],
    temperature: float,
    top_p: float
) -> tuple:
    """Sets up logit bias, logits processors, sampler, and stop tokens for generation."""
    logit_bias = {tokens[0]: 0.0 for verdict,
                  tokens in verdict_token_map.items() if tokens}
    logits_processors = make_logits_processors(logit_bias=logit_bias)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    stop_tokens = tokenizer.encode("\n") + list(tokenizer.eos_token_ids)
    return logits_processors, sampler, stop_tokens


def generate_verdict_stream(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int],
    verdicts: List[str]
) -> tuple[str, int, List[int]]:
    """Generates a verdict using stream_generate method."""
    verdict = ""
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
        verdict = model_components.tokenizer.decode(generated_tokens)
        if verdict in verdicts:
            break
    return verdict, token_id, generated_tokens


def generate_verdict_step(
    model_components: ModelComponents,
    formatted_prompt: str,
    max_tokens: int,
    logits_processors: list,
    sampler,
    stop_tokens: List[int],
    verdicts: List[str]
) -> tuple[str, int, List[int]]:
    """Generates a verdict using generate_step method."""
    verdict = ""
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
        verdict = model_components.tokenizer.decode(generated_tokens)
        if verdict in verdicts:
            break
    return verdict, token_id, generated_tokens


def validate_verdict(verdict: str, verdicts: List[str]) -> None:
    """Validates that the generated verdict is one of the provided options."""
    if verdict not in verdicts:
        raise InvalidOutputError(
            f"Output '{verdict}' is not one of the provided verdicts.")


def fact_check(
    statement: str,
    verdicts: List[str] = ["True", "False", "Uncertain"],
    model_path: LLMModelType = "llama-3.2-3b-instruct-4bit",
    method: str = "stream_generate",
    max_tokens: int = 10,
    temperature: float = 0.0,
    top_p: float = 0.9
) -> FactCheckResult:
    """Evaluates the truthfulness of a statement."""
    try:
        validate_method(method)
        model_components = load_model_components(model_path)
        system_prompt = create_system_prompt(verdicts)
        log_prompt_details(system_prompt, statement, model_path)
        messages = format_chat_messages(system_prompt, statement)
        formatted_prompt = model_components.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        verdict_token_map = encode_verdicts(
            model_components.tokenizer, verdicts)
        logits_processors, sampler, stop_tokens = setup_generation_parameters(
            model_components.tokenizer, verdict_token_map, temperature, top_p
        )
        if method == "stream_generate":
            verdict, token_id, _ = generate_verdict_stream(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens, verdicts
            )
        else:
            verdict, token_id, _ = generate_verdict_step(
                model_components, formatted_prompt, max_tokens,
                logits_processors, sampler, stop_tokens, verdicts
            )
        validate_verdict(verdict, verdicts)
        return FactCheckResult(
            verdict=verdict,
            token_id=token_id,
            is_valid=True,
            method=method,
            error=None
        )
    except Exception as e:
        return FactCheckResult(
            verdict="",
            token_id=-1,
            is_valid=False,
            method=method,
            error=str(e)
        )
