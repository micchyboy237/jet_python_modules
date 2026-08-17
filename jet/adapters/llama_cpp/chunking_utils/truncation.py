# jet_python_modules/jet/adapters/llama_cpp/chunking_utils/truncation.py
from typing import overload

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.logger import logger
from tqdm import tqdm


@overload
def truncate_texts(
    texts: str,
    model: str | LLAMACPP_KEYS = ...,
    max_tokens: int | None = ...,
    strict_sentences: bool = ...,
    show_progress: bool = ...,
) -> str: ...


@overload
def truncate_texts(
    texts: list[str],
    model: str | LLAMACPP_KEYS = ...,
    max_tokens: int | None = ...,
    strict_sentences: bool = ...,
    show_progress: bool = ...,
) -> list[str]: ...


def truncate_texts(
    texts: str | list[str],
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    max_tokens: int | None = None,
    strict_sentences: bool = True,
    show_progress: bool = True,
) -> str | list[str]:
    """Truncate texts to a maximum token count, preserving sentence boundaries when possible.

    Based on text_chunker.py's truncate_texts_fast approach with these advantages:
    - Uses split_sentences_with_separators() for single-pass splitting with separators included
    - No manual separator extraction or sentence position hunting after splitting
    - No parallel sentence/separator array management
    - Simple "".join() reconstruction since separators are already part of sentences
    - Returns matching type: string input → string output, list input → list output

    Args:
        texts: Single text or list of texts to truncate.
        model: Model key for tokenizer (default: LLM_MODEL).
        max_tokens: Maximum tokens to keep. If None, uses model's context size.
        strict_sentences: If True, preserve sentence boundaries. If False, truncate at token level.
        show_progress: Show progress bar during batch processing.

    Returns:
        Truncated text string (if input was str) or list of truncated strings (if input was list).
        Empty strings from empty inputs are preserved for string input, filtered for list input.
    """
    from jet.wordnet.sentence import split_sentences_with_separators

    single_input = isinstance(texts, str)
    if single_input:
        texts = [texts]

    if max_tokens is None:
        try:
            from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size

            ctx_info = get_model_ctx_embd_size(model)
            max_tokens = ctx_info["ctx"]
            logger.debug(f"Using model context size as max_tokens: {max_tokens}")
        except Exception as e:
            logger.warning(
                f"Could not get context size for {model}, using default 2048: {e}"
            )
            max_tokens = 2048

    tokenizer = get_tokenizer(model)
    results = []

    if show_progress and len(texts) > 1:
        text_iter = tqdm(texts, desc="Truncating texts", unit="doc")
    else:
        text_iter = texts

    for text in text_iter:
        if not text or not text.strip():
            if single_input:
                results.append("")
            continue

        original_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        if original_tokens <= max_tokens:
            results.append(text.strip())
            logger.debug(
                f"Text fits within limit ({original_tokens}/{max_tokens} tokens), no truncation needed"
            )
            continue

        if not strict_sentences:
            tokens = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            truncated = tokenizer.decode(
                tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            results.append(truncated)
            logger.debug(
                f"Token-level truncation: {original_tokens} → {max_tokens} tokens"
            )
            continue

        sentences = split_sentences_with_separators(text)
        if not sentences:
            logger.debug(f"No sentences found in text: {text[:50]}...")
            if single_input:
                results.append("")
            continue

        current_tokens = 0
        kept_sentences = []
        total_sentences = len(sentences)

        for sentence in sentences:
            sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
            sentence_len = len(sentence_tokens)

            if current_tokens + sentence_len > max_tokens:
                logger.debug(
                    f"Truncating at sentence boundary: {current_tokens}/{max_tokens} tokens, "
                    f"next sentence adds {sentence_len} tokens "
                    f"(kept {len(kept_sentences)}/{total_sentences} sentences)"
                )
                break

            kept_sentences.append(sentence)
            current_tokens += sentence_len

        if kept_sentences:
            truncated = "".join(kept_sentences).strip()
            results.append(truncated)
            logger.debug(
                f"Sentence-boundary truncation: {original_tokens} → {current_tokens} tokens, "
                f"{len(kept_sentences)}/{total_sentences} sentences kept"
            )
        else:
            logger.debug(
                f"No complete sentence fits within {max_tokens} tokens, "
                f"falling back to token-level truncation of first sentence"
            )
            first_sentence = sentences[0]
            tokens = tokenizer.encode(first_sentence, add_special_tokens=False)[
                :max_tokens
            ]
            truncated = tokenizer.decode(
                tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            results.append(truncated)

    non_empty = sum(1 for r in results if r)
    logger.info(
        f"Truncation complete: {len(texts)} input(s) → {len(results)} output(s) "
        f"({non_empty} non-empty)"
    )

    return results[0] if single_input else results
