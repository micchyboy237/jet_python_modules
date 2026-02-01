import json
from collections.abc import Callable
from typing import Literal, overload

import tiktoken
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS, LLAMACPP_VALUES
from jet.adapters.llama_cpp.utils import resolve_model_value
from jet.logger import logger
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


def get_tokenizer(
    model_name: LLAMACPP_VALUES | None = None,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast | tiktoken.Encoding:
    """
    Get a tokenizer for a given model name, or fall back to tiktoken if none/model not found.
    """
    if model_name is None:
        return tiktoken.get_encoding("cl100k_base")

    try:
        model_id = resolve_model_value(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return tokenizer
    except Exception as e:
        logger.warning(
            f'Model "{model_name}" not found, falling back to tiktoken cl100k_base: {str(e)}'
        )
        return tiktoken.get_encoding("cl100k_base")


def get_tokenizer_fn(
    model_name: LLAMACPP_VALUES | None = None, add_special_tokens: bool = False
) -> Callable[[str | list[str], bool], list[int] | list[list[int]]]:
    """
    Returns a tokenizer function for the specified model that tokenizes input text.

    Args:
        model_name (LLAMACPP_VALUES | None): The name of the model to get the tokenizer for. If None, uses tiktoken's "cl100k_base" tokenizer.
        add_special_tokens (bool, optional): Whether to add special tokens during tokenization. Defaults to False.

    Returns:
        Callable: A function that tokenizes input text (str or list[str]) into tokens (list[int] or list[list[int]]).

    Raises:
        ValueError: If the model_name is invalid or tokenizer cannot be loaded.
    """
    tokenizer = get_tokenizer(model_name)

    def tokenize_fn(
        text: str | list[str], add_special_tokens: bool = add_special_tokens
    ) -> list[int] | list[list[int]]:
        if isinstance(text, list):
            if isinstance(tokenizer, tiktoken.Encoding):
                return tokenizer.encode_batch(text)
            return tokenizer.batch_encode_plus(
                text, add_special_tokens=add_special_tokens, return_tensors=None
            )["input_ids"]
        else:
            if isinstance(tokenizer, tiktoken.Encoding):
                return tokenizer.encode(text)
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    return tokenize_fn


def tokenize(
    text: str | dict | list[str] | list[dict] = "",
    model_name: LLAMACPP_VALUES | None = None,
    add_special_tokens: bool = False,
) -> list[int] | list[list[int]]:
    tokenizer = get_tokenizer(model_name)

    if isinstance(text, list):
        # --- CHAT-AWARE PATH ---
        # Detect chat-style messages: list[dict] with "role"
        if text and isinstance(text[0], dict) and "role" in text[0]:
            # Normalize tool calls into content for token counting
            normalized_messages = []
            for msg in text:
                if "tool_calls" in msg:
                    tool_payload = json.dumps(
                        msg["tool_calls"], ensure_ascii=False, sort_keys=True
                    )
                    normalized_messages.append(
                        {
                            "role": msg.get("role", "assistant"),
                            "content": tool_payload,
                        }
                    )
                else:
                    normalized_messages.append(
                        {
                            "role": msg.get("role"),
                            "content": msg.get("content", ""),
                        }
                    )

            if isinstance(tokenizer, tiktoken.Encoding):
                raise ValueError(
                    "Chat-style tokenization requires a model-specific tokenizer "
                    "with apply_chat_template; tiktoken cannot account for roles "
                    "or message framing."
                )

            if not hasattr(tokenizer, "apply_chat_template"):
                raise ValueError("Tokenizer does not support chat templates.")

            return tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=True,
                add_generation_prompt=False,
            )

        texts = []
        for t in text:
            if isinstance(t, dict):
                texts.append(str(t.get("content", t)))
            else:
                texts.append(str(t))

        if isinstance(tokenizer, tiktoken.Encoding):
            tokenized = tokenizer.encode_batch(
                texts, allowed_special="all" if add_special_tokens else set()
            )
        else:
            tokenized = tokenizer.batch_encode_plus(
                texts, return_tensors=None, add_special_tokens=add_special_tokens
            )
            tokenized = tokenized["input_ids"]
        return tokenized
    else:
        if isinstance(text, dict):
            text_str = str(text.get("content", text))
        else:
            text_str = str(text)
        if isinstance(tokenizer, tiktoken.Encoding):
            tokenized = tokenizer.encode(
                text_str, allowed_special="all" if add_special_tokens else set()
            )
        else:
            tokenized = tokenizer.encode(
                text_str, add_special_tokens=add_special_tokens
            )
        return tokenized


TokenizableInput = str | dict | list[str] | list[dict]


@overload
def count_tokens(
    text: TokenizableInput,
    model: LLAMACPP_KEYS | None = None,
    prevent_total: Literal[False] = False,
    add_special_tokens: bool = False,
) -> int: ...


@overload
def count_tokens(
    text: TokenizableInput,
    model: LLAMACPP_KEYS | None = None,
    prevent_total: Literal[True] = True,
    add_special_tokens: bool = False,
) -> list[int]: ...


def count_tokens(
    text: TokenizableInput,
    model: LLAMACPP_KEYS | None = None,
    prevent_total: bool = False,
    add_special_tokens: bool = False,
) -> int | list[int]:
    if not text:
        return 0

    # Chat-style input is fully tokenized as a single sequence
    if (
        isinstance(text, list)
        and text
        and isinstance(text[0], dict)
        and "role" in text[0]
    ):
        tokenized = tokenize(text, model, add_special_tokens)
        return len(tokenized)

    tokenized = tokenize(text, model, add_special_tokens)
    if isinstance(text, (str, dict)):
        return len(tokenized)
    else:
        token_counts = [len(item) for item in tokenized]
        return sum(token_counts) if not prevent_total else token_counts
