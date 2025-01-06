from typing import Literal, Optional
from jet.logger import logger
from llama_index.core.base.llms.types import ChatMessage
import tiktoken
from jet.llm.llm_types import Message
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from jet.llm.ollama.models import (
    OLLAMA_HF_MODELS,
    OLLAMA_MODEL_EMBEDDING_TOKENS,
)


def get_ollama_models():
    """Lazy loading of Ollama models to avoid circular imports"""

    return OLLAMA_HF_MODELS, OLLAMA_MODEL_EMBEDDING_TOKENS


def tokenizer():
    encoding = tiktoken.get_encoding("cl100k_base")
    return encoding


def get_tokenizer(model_name: str) -> PreTrainedTokenizer | PreTrainedTokenizerFast | tiktoken.Encoding:
    try:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_name)
        return tokenizer
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding


def tokenize(model_name: str, text: str | list[str] | list[dict]):
    tokenizer = get_tokenizer(model_name)

    if isinstance(text, list):
        texts = [str(t) for t in text]

        if isinstance(tokenizer, tiktoken.Encoding):
            tokenized = tokenizer.encode_batch(texts)
        else:
            tokenized = tokenizer.batch_encode_plus(texts, return_tensors=None)
            tokenized = tokenized["input_ids"]
        return tokenized
    else:
        tokens = tokenizer.encode(str(text))
        return tokens


def token_counter(
    text: str | list[str] | list[ChatMessage],
    model: Optional[str] = "mistral",
    prevent_total: bool = False
) -> int | list[int]:
    if not text:
        raise ValueError("text cannot be None")

    # Get models only when needed
    OLLAMA_HF_MODELS, _ = get_ollama_models()

    if model not in OLLAMA_HF_MODELS:
        raise ValueError(f"Model can only be one of the ff: {
                         [tuple(OLLAMA_HF_MODELS.keys())]}")

    tokenized = tokenize(OLLAMA_HF_MODELS[model], text)
    if isinstance(text, str):
        return len(tokenized)
    else:
        token_counts = [len(item) for item in tokenized]
        return sum(token_counts) if not prevent_total else token_counts


def get_model_max_tokens(
    model: Optional[str] = "mistral",
) -> int:
    # Get models only when needed
    _, OLLAMA_MODEL_EMBEDDING_TOKENS = get_ollama_models()

    if model not in OLLAMA_MODEL_EMBEDDING_TOKENS:
        raise ValueError(f"Model can only be one of the ff: {
                         [tuple(OLLAMA_MODEL_EMBEDDING_TOKENS.keys())]}")

    return OLLAMA_MODEL_EMBEDDING_TOKENS[model]


def filter_texts(
    text: str | list[str] | list[dict] | list[ChatMessage],
    model: Optional[Literal[tuple(OLLAMA_HF_MODELS.keys())]] = "mistral",
    max_tokens: Optional[int | float] = None,
) -> str | list[str] | list[dict] | list[ChatMessage]:
    tokenizer = get_tokenizer(OLLAMA_HF_MODELS[model])
    if isinstance(max_tokens, float) and max_tokens < 1:
        max_tokens = int(get_model_max_tokens(model) * max_tokens)
    else:
        max_tokens = max_tokens or get_model_max_tokens(model)

    if isinstance(text, str):
        token_count = token_counter(text, model)
        if token_count <= max_tokens:
            return [text]

        # Split into manageable chunks
        tokens = tokenize(OLLAMA_HF_MODELS[model], text)
        return tokenizer.decode(tokens[0:max_tokens], skip_special_tokens=False)
    else:
        if isinstance(text[0], str):
            filtered_texts = []
            current_token_count = 0

            # Precompute token counts for all text in a single batch for efficiency
            text_token_counts = token_counter(text, model, prevent_total=True)

            for text, token_count in zip(text, text_token_counts):
                # Check if adding this text will exceed the max_tokens limit
                if current_token_count + token_count <= max_tokens:
                    filtered_texts.append(text)
                    current_token_count += token_count
                else:
                    break  # Stop early since texts are already sorted by score

            return filtered_texts
        else:
            messages = text.copy()
            token_count = token_counter(str(messages), model)

            if isinstance(token_count, int) and token_count <= max_tokens:
                return messages

            # Remove messages one by one from second to last up to second
            while len(messages) > 2 and isinstance(token_count, int) and token_count > max_tokens:
                messages.pop(-2)  # Remove second to last message
                token_count = token_counter(str(messages), model)

            return messages


if __name__ == "__main__":
    models = ["llama3.1"]
    ollama_models = {}
    sample_text = "Text 1, Text 2"
    sample_texts = ["Text 1", "Text 2"]

    logger.info("Count tokens for: str")
    for model_name in models:
        result = token_counter(sample_text, model_name)
        logger.log("Count:", result, colors=["DEBUG", "SUCCESS"])

    logger.info("Count batch tokens for: list[str]")
    for model_name in models:
        result = token_counter(sample_texts, model_name)
        logger.log("Count:", result, colors=["DEBUG", "SUCCESS"])
