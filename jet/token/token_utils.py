from typing import Callable, Literal, Optional, TypedDict, Union
from jet.logger import logger
from llama_index.core.base.llms.types import ChatMessage
import tiktoken
from jet.llm.llm_types import Message
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from jet.llm.models import (
    OLLAMA_HF_MODEL_NAMES,
    OLLAMA_HF_MODELS,
    OLLAMA_MODEL_EMBEDDING_TOKENS,
    OLLAMA_MODEL_NAMES,
)


def get_ollama_models():
    """Lazy loading of Ollama models to avoid circular imports"""

    return OLLAMA_HF_MODELS, OLLAMA_MODEL_EMBEDDING_TOKENS


def get_ollama_tokenizer(model_name: str | OLLAMA_MODEL_NAMES | OLLAMA_HF_MODEL_NAMES) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    if model_name in OLLAMA_MODEL_NAMES.__args__:
        model_name = OLLAMA_HF_MODELS[model_name]

    if model_name in OLLAMA_HF_MODEL_NAMES.__args__:
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_name)
        return tokenizer

    raise ValueError(f"Model \"{model_name}\" not found")


def get_tokenizer(model_name: str | OLLAMA_MODEL_NAMES | OLLAMA_HF_MODEL_NAMES) -> PreTrainedTokenizer | PreTrainedTokenizerFast | tiktoken.Encoding:
    if model_name in OLLAMA_MODEL_NAMES.__args__:
        model_name = OLLAMA_HF_MODELS[model_name]

    if model_name in OLLAMA_HF_MODEL_NAMES.__args__:
        return get_ollama_tokenizer(model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding


def tokenize(model_name: str | OLLAMA_MODEL_NAMES, text: str | list[str] | list[dict]):
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
    text: str | list[str] | list[ChatMessage] | list[Message],
    model: Optional[str | OLLAMA_MODEL_NAMES] = "mistral",
    prevent_total: bool = False
) -> int | list[int]:
    if not text:
        return 0

    tokenized = tokenize(model, text)
    if isinstance(text, str):
        return len(tokenized)
    else:
        token_counts = [len(item) for item in tokenized]
        return sum(token_counts) if not prevent_total else token_counts


class TokenCountsInfoResult(TypedDict):
    tokens: int
    text: str


class TokenCountsInfo(TypedDict):
    average: float
    max: int
    min: int
    results: list[TokenCountsInfoResult]


def get_token_counts_info(texts: list[str], model: OLLAMA_MODEL_NAMES) -> TokenCountsInfo:
    token_counts: list[int] = token_counter(texts, model, prevent_total=True)
    total_count = sum(token_counts)
    avg_count = round(total_count / len(token_counts),
                      2) if token_counts else 0.0  # Rounded average
    results: list[TokenCountsInfoResult] = [{"tokens": count, "text": text}
                                            for count, text in zip(token_counts, texts)]

    return {
        "min": min(token_counts) if token_counts else 0,
        "max": max(token_counts) if token_counts else 0,
        "average": avg_count,
        "results": sorted(results, key=lambda x: x["tokens"])
    }


def get_model_max_tokens(
    model: Optional[str | OLLAMA_MODEL_NAMES] = "mistral",
) -> int:
    if model in OLLAMA_MODEL_EMBEDDING_TOKENS:
        return OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer.model_max_length
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.max_token_value


def filter_texts(
    text: str | list[str] | list[ChatMessage] | list[Message],
    model: str | OLLAMA_MODEL_NAMES = "mistral",
    max_tokens: Optional[int | float] = None,
) -> str | list[str] | list[dict] | list[ChatMessage]:
    if not max_tokens:
        max_tokens = 0.5

    tokenizer = get_tokenizer(OLLAMA_HF_MODELS[model])
    if isinstance(max_tokens, float) and max_tokens < 1:
        max_tokens = int(
            get_model_max_tokens(model) * max_tokens)
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

            for t, token_count in zip(text, text_token_counts):
                # Check if adding this text will exceed the max_tokens limit
                if current_token_count + token_count <= max_tokens:
                    filtered_texts.append(t)
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


def group_texts(
    text: str | list[str] | list[ChatMessage] | list[Message],
    model: str | OLLAMA_MODEL_NAMES = "mistral",
    max_tokens: Optional[int | float] = None,
) -> list[list[str]]:
    if not max_tokens:
        max_tokens = 0.5

    tokenizer = get_tokenizer(OLLAMA_HF_MODELS[model])
    if isinstance(max_tokens, float) and max_tokens < 1:
        max_tokens = int(get_model_max_tokens(model) * max_tokens)
    else:
        max_tokens = max_tokens or get_model_max_tokens(model)

    if isinstance(text, str):
        tokens = tokenize(OLLAMA_HF_MODELS[model], text)
        grouped_texts = []

        for i in range(0, len(tokens), max_tokens):
            chunk = tokens[i:i + max_tokens]
            grouped_texts.append(tokenizer.decode(
                chunk, skip_special_tokens=False))

        return grouped_texts

    elif isinstance(text, list) and isinstance(text[0], str):
        grouped_texts = []
        current_group = []
        current_token_count = 0

        text_token_counts = token_counter(text, model, prevent_total=True)

        for t, token_count in zip(text, text_token_counts):
            if current_token_count + token_count > max_tokens:
                grouped_texts.append(current_group)
                current_group = []
                current_token_count = 0

            current_group.append(t)
            current_token_count += token_count

        if current_group:
            grouped_texts.append(current_group)

        return grouped_texts

    else:
        raise TypeError("Unsupported input type for group_texts")


def calculate_num_predict_ctx(prompt: str | list[str] | list[ChatMessage] | list[Message], model: str = "llama3.1", *, system: str = "", max_prediction_ratio: float = 0.75):
    user_tokens: int = token_counter(prompt, model)
    system_tokens: int = token_counter(system, model)
    prompt_tokens = user_tokens + system_tokens
    num_predict = int(prompt_tokens * max_prediction_ratio)
    num_ctx = prompt_tokens + num_predict

    model_max_tokens = OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    if num_ctx > model_max_tokens:
        raise ValueError({
            "prompt_tokens": prompt_tokens,
            "num_predict": num_predict,
            "error": f"Context window size ({num_ctx}) exceeds model's maximum tokens ({model_max_tokens})",
        })

    return {
        "user_tokens": user_tokens,
        "system_tokens": system_tokens,
        "prompt_tokens": prompt_tokens,
        "num_predict": num_predict,
        "num_ctx": num_ctx,
    }


def truncate_texts(texts: str | list[str], model: str, max_tokens: int) -> list[str]:
    """
    Truncates texts that exceed the max_tokens limit.

    Args:
        texts (str | list[str]): A list of texts to be truncated.
        model (str): The model name for tokenization.
        max_tokens (int): The maximum number of tokens allowed per text.

    Returns:
        list[str]: A list of truncated texts.
    """
    tokenizer = get_tokenizer(model)

    if isinstance(texts, str):
        texts = [texts]

    tokenized_texts = tokenizer.batch_encode_plus(texts, return_tensors=None)
    tokenized_texts = tokenized_texts["input_ids"]
    truncated_texts = []

    for text, tokens in zip(texts, tokenized_texts):
        if len(tokens) > max_tokens:
            truncated_text = tokenizer.decode(
                tokens[:max_tokens], skip_special_tokens=True)
            truncated_texts.append(truncated_text)
        else:
            truncated_texts.append(text)

    return truncated_texts


def split_texts(
    texts: str | list[str],
    model: str | OLLAMA_MODEL_NAMES,
    chunk_size: Optional[int] = None,
    chunk_overlap: int = 0,
    *,
    buffer: int = 0
) -> list[str]:
    """
    Splits a list of texts into smaller chunks based on chunk_size, chunk_overlap, and buffer.

    Args:
        texts (str | list[str]): List of input texts to be split.
        model (str): Model name for tokenization.
        chunk_size (int): Maximum tokens allowed per chunk.
        chunk_overlap (int): Number of overlapping tokens between chunks.
        buffer (int, optional): Extra space reserved to avoid exceeding chunk_size. Default is 0.

    Returns:
        list[str]: A list of split text chunks.
    """
    if not chunk_size:
        chunk_size = OLLAMA_MODEL_EMBEDDING_TOKENS[model]

    if chunk_size <= chunk_overlap:
        raise ValueError(
            f"Chunk size ({chunk_size}) must be greater than chunk overlap ({chunk_overlap})")

    effective_max_tokens = max(chunk_size - buffer, 1)  # Ensure positive value
    if effective_max_tokens <= chunk_overlap:
        raise ValueError(
            f"Effective max tokens ({effective_max_tokens}) must be greater than chunk overlap ({chunk_overlap})")

    tokenizer = get_tokenizer(model)
    split_chunks = []

    if isinstance(texts, str):
        texts = [texts]

    for text in texts:
        tokens = tokenizer.encode(text) if hasattr(
            tokenizer, "encode") else tokenizer(text)
        total_tokens = len(tokens)

        if total_tokens <= effective_max_tokens:
            split_chunks.append(text)
            continue

        start = 0
        while start < total_tokens:
            end = min(start + effective_max_tokens, total_tokens)
            chunk_tokens = tokens[start:end]
            try:
                chunk_text = tokenizer.decode(
                    chunk_tokens, skip_special_tokens=True)
            except:
                chunk_text = tokenizer.decode(chunk_tokens)

            chunk_text = chunk_text.strip()

            if chunk_text:  # Ensure non-empty chunks are added
                split_chunks.append(chunk_text)

            if end == total_tokens:
                break
            start = max(end - chunk_overlap, 0)  # Prevent negative index

    logger.debug(
        f"Split {len(texts)} texts into {len(split_chunks)} chunks (buffer={buffer}, overlap={chunk_overlap}).")
    return split_chunks


def group_texts(
    texts: Union[list[str], list[ChatMessage], list[Message]],
    model: str,
    max_tokens: int
) -> list[list[str]]:
    """
    Groups texts into batches without exceeding a maximum token limit.

    Args:
        texts (list[str] | list[ChatMessage] | list[Message]): The texts to group.
        model (str): The model name for tokenization.
        max_tokens (int): The maximum allowed tokens per group.

    Returns:
        list[list[str]]: A list of grouped text batches.
    """
    if not texts:
        return []

    # Token count lookup
    text_token_counts = token_counter(texts, model, prevent_total=True)
    grouped_texts, current_group = [], []
    current_token_count = 0

    for text, token_count in zip(texts, text_token_counts):
        # If adding a new text exceeds the limit, store current batch
        if current_token_count + token_count > max_tokens:
            grouped_texts.append(current_group)
            current_group = []
            current_token_count = 0

        current_group.append(text)
        current_token_count += token_count

    if current_group:
        grouped_texts.append(current_group)

    return grouped_texts


if __name__ == "__main__":
    from jet.file.utils import load_file
    from jet.search.formatters import clean_string
    from jet.transformers.formatters import format_json

    # models = ["llama3.1"]
    models = ["paraphrase-MiniLM-L12-v2"]
    ollama_models = {}

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    data = load_file(data_file)
    docs = []
    for item in data:
        cleaned_sentence = clean_string(item)
        docs.append(cleaned_sentence)

    sample_text = "Text 1, Text 2"
    sample_texts = docs

    logger.info("Count tokens for: str")
    for model_name in models:
        result = token_counter(sample_text, model_name)
        logger.log("Count:", format_json(result), colors=["DEBUG", "SUCCESS"])

    logger.info("Count tokens info for: str")
    for model_name in models:
        splitted_texts = split_texts(
            docs, model_name, chunk_size=200, chunk_overlap=50)
        result = get_token_counts_info(splitted_texts, model_name)
        logger.log("Count:", format_json(result), colors=["DEBUG", "SUCCESS"])
