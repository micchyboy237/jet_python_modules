from jet.llm.mlx.mlx_types import LLMModelType
from jet.llm.mlx.models import AVAILABLE_MODELS, MODEL_CONTEXTS, MODEL_EMBEDDING_TOKENS
from mlx_lm import stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizer
from typing import Union, List, Dict, Optional, Literal
import mlx.core as mx


def get_model_max_tokens(model: LLMModelType, max_kv_size: Optional[int] = None) -> int:
    """
    Retrieve the maximum token length of the model (input + output tokens).

    Args:
        model (str): The model name or identifier.

    Returns:
        int: The maximum token length.
    """
    # Check if model is a key in AVAILABLE_MODELS
    if model in AVAILABLE_MODELS:
        model_name = model
    # Check if model is a value in AVAILABLE_MODELS
    else:
        model_name = next(
            (key for key, value in AVAILABLE_MODELS.items() if value == model), None)

    if model_name is None:
        raise ValueError(f"Model {model} not found in AVAILABLE_MODELS")

    max_tokens = get_hidden_size(model_name)

    # If max_kv_size is specified and smaller, it limits the token length
    if max_kv_size is not None and max_kv_size < max_tokens:
        max_tokens = max_kv_size
        print(
            f"Max token length limited by max_kv_size: {max_tokens}")

    return max_tokens


def get_max_context_length(model: LLMModelType) -> int:
    """
    Retrieve the maximum context length of the model (input + output tokens).

    Args:
        model (nn.Module): The MLX model.
        max_kv_size (Optional[int]): The maximum key-value cache size, if specified.

    Returns:
        int: The maximum context length (in tokens).
    """
    return MODEL_CONTEXTS[model]


def get_hidden_size(model: LLMModelType) -> int:
    """
    Retrieve the hidden size (embedding dimension) of the model.

    Args:
        model (nn.Module): The MLX model.

    Returns:
        int: The hidden size of the model.

    Raises:
        AttributeError: If neither hidden_size nor n_embd is found in the model configuration.
    """
    return MODEL_EMBEDDING_TOKENS[model]


def get_prompt_token_count(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    add_special_tokens: bool = True
) -> int:
    """
    Calculate the token count for a given prompt.

    Args:
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt (string, token array, or token list).
        add_special_tokens (bool): Whether to add special tokens (e.g., BOS) during encoding.

    Returns:
        int: The number of tokens in the prompt.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if isinstance(prompt, str):
        tokens = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)
    elif isinstance(prompt, mx.array):
        tokens = prompt
    else:
        tokens = mx.array(prompt)

    return tokens.size if isinstance(tokens, mx.array) else len(tokens)


def get_messages_token_count(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    messages: List[Dict[str, str]],
    chat_template_config: Optional[Dict] = None,
    add_special_tokens: bool = False,
    continue_final_message: bool = False,
    add_generation_prompt: bool = True
) -> int:
    """
    Calculate the token count for a list of messages, applying the chat template if available.

    Args:
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        messages (List[Dict[str, str]]): List of messages with 'role' and 'content' keys.
        chat_template_config (Optional[Dict]): Additional config for chat template.
        add_special_tokens (bool): Whether to add special tokens during encoding.
        continue_final_message (bool): Whether to continue the final message (for prefill).
        add_generation_prompt (bool): Whether to add a generation prompt.

    Returns:
        int: The total number of tokens for the messages.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    chat_template_config = chat_template_config or {}

    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **chat_template_config
        )
        tokens = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)
    else:
        prompt = "".join(message["content"] for message in messages)
        tokens = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)

    return tokens.size if isinstance(tokens, mx.array) else len(tokens)


def get_individual_message_token_counts(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    messages: List[Dict[str, str]],
    add_special_tokens: bool = False
) -> List[Dict[str, Union[str, int]]]:
    """
    Calculate the token count for each message individually.

    Args:
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        messages (List[Dict[str, str]]): List of messages with 'role' and 'content' keys.
        add_special_tokens (bool): Whether to add special tokens during encoding.

    Returns:
        List[Dict[str, Union[str, int]]]: List of dictionaries with 'role', 'content', and 'token_count'.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    result = []
    for message in messages:
        tokens = tokenizer.encode(
            message["content"], add_special_tokens=add_special_tokens)
        token_count = tokens.size if isinstance(
            tokens, mx.array) else len(tokens)
        result.append({
            "role": message["role"],
            "content": message["content"],
            "token_count": token_count
        })
    return result


def get_response_token_count(
    model: 'nn.Module',
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 100,
    **kwargs
) -> tuple[str, int]:
    """
    Calculate the token count for the generated response.

    Args:
        model (nn.Module): The MLX model.
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt.
        max_tokens (int): Maximum number of tokens to generate.
        **kwargs: Additional arguments passed to stream_generate (e.g., sampler, draft_model).

    Returns:
        tuple[str, int]: The generated text and the number of tokens in the response.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    text = ""
    response_token_count = 0

    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, **kwargs):
        text += response.text
        response_token_count = response.generation_tokens
        if response.finish_reason in ["stop", "length"]:
            break

    return text, response_token_count


__all__ = [
    "get_model_max_tokens",
    "get_max_context_length",
    "get_hidden_size",
    "get_prompt_token_count",
    "get_messages_token_count",
    "get_individual_message_token_counts",
    "get_response_token_count",
]
