import uuid
from typing import Callable

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.logger import logger
from jet.models.chunkers import chunk_headers_by_hierarchy
from jet.wordnet.sentence import split_sentences

from .tokenization import _decode_tokens, _get_size_fn
from .types import MarkdownChunkResult, OverlapStrategy


def _create_tokenizer_adapter(size_fn: Callable) -> Callable:
    """Adapt LlamaCPP size_fn to the interface expected by markdown_hierarchy_chunking."""

    def _adapter(text):
        if isinstance(text, list):
            return [len(ids) for ids in size_fn(text)]
        return size_fn(text)

    return _adapter


def _apply_sentence_overlap(
    prev_content: str,
    next_content: str,
    overlap_size: int,
) -> str:
    """Prepend last N sentences from previous chunk body to next chunk body."""
    if overlap_size <= 0 or not prev_content:
        return next_content
    sentences = split_sentences(prev_content)
    if not sentences:
        return next_content
    overlap_sents = sentences[-overlap_size:]
    overlap_text = " ".join(overlap_sents)
    if not overlap_text.strip():
        return next_content
    return f"{overlap_text} {next_content}"


def _apply_token_overlap(
    prev_content: str,
    next_content: str,
    header: str,
    overlap_size: int,
    size_fn: Callable,
    max_chunk_size: int,
    model: str | LLAMACPP_KEYS,
) -> tuple[str, int]:
    """Prepend token-level overlap from previous chunk body to next chunk body.

    Returns (adjusted_content, adjusted_token_count).
    Header tokens are excluded from overlap extraction but counted toward budget.
    Trims overlap from oldest end if combined content exceeds budget.
    """
    if overlap_size <= 0 or not prev_content:
        return next_content, len(size_fn(next_content))

    prev_tokens = size_fn(prev_content)
    if not prev_tokens:
        return next_content, len(size_fn(next_content))

    overlap_ids = prev_tokens[-overlap_size:]
    overlap_text = _decode_tokens(overlap_ids, model)

    combined_body = f"{overlap_text} {next_content}"
    combined_tokens = size_fn(combined_body)

    header_tokens = len(size_fn(header))
    available = max_chunk_size - header_tokens

    trim_count = 0
    while len(combined_tokens) > available and overlap_ids:
        overlap_ids = overlap_ids[1:]
        trim_count += 1
        overlap_text = _decode_tokens(overlap_ids, model)
        combined_body = f"{overlap_text} {next_content}"
        combined_tokens = size_fn(combined_body)

    if trim_count > 0:
        logger.debug(
            f"Token overlap trimmed {trim_count} tokens to fit budget "
            f"(requested={overlap_size}, used={len(overlap_ids)})"
        )

    return combined_body, len(combined_tokens)


def chunk_markdown_hierarchy(
    markdown_text: str,
    chunk_size: int = 512,
    overlap_strategy: OverlapStrategy = "sentence",
    overlap_size: int = 1,
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    buffer: int = 0,
    min_chunk_size: int = 32,
    show_progress: bool = True,
) -> list[str]:
    """Chunk markdown text respecting header hierarchy and token limits.

    Returns flat list of chunk strings for simple use cases.
    For rich metadata, use chunk_markdown_hierarchy_with_data().
    """
    results = chunk_markdown_hierarchy_with_data(
        markdown_text=markdown_text,
        chunk_size=chunk_size,
        overlap_strategy=overlap_strategy,
        overlap_size=overlap_size,
        model=model,
        buffer=buffer,
        min_chunk_size=min_chunk_size,
        show_progress=show_progress,
    )
    return [r["content"] for r in results]


def chunk_markdown_hierarchy_with_data(
    markdown_text: str | list[str],
    chunk_size: int = 512,
    overlap_strategy: OverlapStrategy = "sentence",
    overlap_size: int = 1,
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    ids: list[str] | None = None,
    buffer: int = 0,
    min_chunk_size: int = 32,
    show_progress: bool = True,
) -> list[MarkdownChunkResult]:
    """Chunk markdown with full hierarchy metadata and configurable overlap.

    Args:
        markdown_text: Single markdown string or list of markdown documents.
        chunk_size: Maximum tokens per chunk (including header tokens).
        overlap_strategy: How to bridge adjacent chunks.
            - "sentence": Prepend last N sentences from previous chunk (default).
            - "token": Prepend last N tokens from previous chunk.
            - "none": No overlap between chunks.
        overlap_size: Number of sentences (for "sentence") or tokens (for "token")
            to carry over. Ignored when overlap_strategy="none".
        model: Model key for LlamaCPP tokenizer.
        ids: Optional document IDs for multi-document input.
        buffer: Reserved token space to avoid exceeding chunk_size.
        min_chunk_size: Minimum tokens for a chunk to be kept standalone.
        show_progress: Show progress bar for multi-document processing.

    Returns:
        List of MarkdownChunkResult with hierarchy and source indices.
    """
    if isinstance(markdown_text, str):
        markdown_text = [markdown_text]
        doc_indices = [0]
    else:
        doc_indices = list(range(len(markdown_text)))

    effective_chunk_size = chunk_size - buffer
    size_fn = _get_size_fn(model)
    tokenizer_adapter = _create_tokenizer_adapter(size_fn)

    # Engine uses word-level overlap internally; we disable it and handle
    # overlap at the adapter level for strategy consistency
    engine_overlap = 0

    all_results: list[MarkdownChunkResult] = []

    from tqdm import tqdm

    doc_iter = tqdm(
        zip(doc_indices, markdown_text),
        total=len(markdown_text),
        desc="Chunking markdown hierarchy",
        disable=not show_progress,
    )

    for doc_index, text in doc_iter:
        if not text.strip():
            continue

        doc_id = ids[doc_index] if ids and doc_index < len(ids) else str(uuid.uuid4())

        try:
            raw_chunks = chunk_headers_by_hierarchy(
                markdown_text=text,
                chunk_size=effective_chunk_size,
                tokenizer=tokenizer_adapter,
                split_fn=split_sentences,
                overlap_tokens=engine_overlap,
            )
        except ValueError as e:
            logger.error(f"Sentence alignment failed for doc {doc_id}: {e}")
            continue

        adapted_chunks: list[MarkdownChunkResult] = []
        prev_content = ""

        for chunk in raw_chunks:
            content = chunk["content"]
            num_tokens = chunk["num_tokens"]

            # Apply overlap based on strategy
            if overlap_strategy == "sentence" and prev_content:
                content = _apply_sentence_overlap(prev_content, content, overlap_size)
                num_tokens = len(size_fn(content))
            elif overlap_strategy == "token" and prev_content:
                content, num_tokens = _apply_token_overlap(
                    prev_content,
                    content,
                    chunk["header"],
                    overlap_size,
                    size_fn,
                    effective_chunk_size,
                    model,
                )

            # Skip undersized chunks unless they're the only content
            if num_tokens < min_chunk_size and len(raw_chunks) > 1:
                logger.debug(
                    f"Skipping undersized chunk ({num_tokens} < {min_chunk_size}): "
                    f"{chunk['header'][:30]}..."
                )
                continue

            adapted: MarkdownChunkResult = {
                "id": chunk["id"],
                "doc_id": doc_id,
                "doc_index": doc_index,
                "chunk_index": chunk["chunk_index"],
                "num_tokens": num_tokens,
                "content": content,
                "start_idx": chunk["metadata"]["start_idx"],
                "end_idx": chunk["metadata"]["end_idx"],
                "line_idx": 0,
                "overlap_start_idx": None,
                "overlap_end_idx": None,
                "header": chunk["header"],
                "parent_header": chunk.get("parent_header"),
                "level": chunk["level"],
                "parent_level": chunk.get("parent_level"),
                "parent_id": chunk.get("parent_id"),
                "section_index": chunk["section_index"],
                "metadata": {
                    "start_idx": chunk["metadata"]["start_idx"],
                    "end_idx": chunk["metadata"]["end_idx"],
                    "body_start_idx": chunk["metadata"]["body_start_idx"],
                    "body_end_idx": chunk["metadata"]["body_end_idx"],
                    "line_idx": text.count("\n", 0, chunk["metadata"]["start_idx"]) + 1,
                },
            }
            adapted_chunks.append(adapted)
            prev_content = chunk["content"]  # Use original content for next overlap

        all_results.extend(adapted_chunks)

    logger.info(
        f"Markdown hierarchy chunking complete: {len(markdown_text)} docs → "
        f"{len(all_results)} chunks (overlap={overlap_strategy}, size={overlap_size})"
    )
    return all_results
