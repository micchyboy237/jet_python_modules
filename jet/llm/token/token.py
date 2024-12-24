from typing import Literal, Optional
import tiktoken
from jet.llm.token.token_image import calculate_img_tokens
from jet.llm.llm_types import Message
from jet.llm.ollama import (
    OLLAMA_HF_MODELS,
    count_tokens as count_ollama_tokens,
    get_token_max_length,
)


def tokenizer():
    encoding = tiktoken.get_encoding("cl100k_base")
    return encoding


def token_counter(
    text: str | list[str] | list[Message] = None,
    model: Optional[Literal[tuple(OLLAMA_HF_MODELS.keys())]] = "mistral",
) -> int:
    if not text:
        raise ValueError("text cannot both be None")

    if model not in OLLAMA_HF_MODELS:
        raise ValueError(f"Model can only be one of the ff: {
                         [tuple(OLLAMA_HF_MODELS.keys())]}")

    return count_ollama_tokens(OLLAMA_HF_MODELS[model], text)
