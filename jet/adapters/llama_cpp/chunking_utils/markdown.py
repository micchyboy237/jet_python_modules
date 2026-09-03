import uuid
from typing import Callable

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.logger import logger
from jet.models.chunkers import chunk_headers_by_hierarchy
from jet.wordnet.sentence import split_sentences

from .tokenization import _get_size_fn
from .types import MarkdownChunkResult


def _create_tokenizer_adapter(size_fn: Callable) -> Callable:
    """Adapt LlamaCPP size_fn to the interface expected by markdown_hierarchy_chunking.

    The engine expects tokenizer(text) -> list[str|int].
    Our size_fn returns list[int] for str and list[list[int]] for list[str].
    """

    def _adapter(text):
        if isinstance(text, list):
            # Engine rarely passes lists to tokenizer directly,
            # but handle gracefully if it does
            return [len(ids) for ids in size_fn(text)]
        return size_fn(text)

    return _adapter


def chunk_markdown_hierarchy(
    markdown_text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
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
        chunk_overlap=chunk_overlap,
        model=model,
        buffer=buffer,
        min_chunk_size=min_chunk_size,
        show_progress=show_progress,
    )
    return [r["content"] for r in results]


def chunk_markdown_hierarchy_with_data(
    markdown_text: str | list[str],
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    ids: list[str] | None = None,
    buffer: int = 0,
    min_chunk_size: int = 32,
    show_progress: bool = True,
) -> list[MarkdownChunkResult]:
    """Chunk markdown with full hierarchy metadata and LlamaCPP tokenization.

    Args:
        markdown_text: Single markdown string or list of markdown documents.
        chunk_size: Maximum tokens per chunk (including header tokens).
        chunk_overlap: Number of words to overlap between consecutive chunks.
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
                overlap_tokens=chunk_overlap,
            )
        except ValueError as e:
            logger.error(f"Sentence alignment failed for doc {doc_id}: {e}")
            continue

        for chunk in raw_chunks:
            num_tokens = chunk["num_tokens"]

            # Skip undersized chunks unless they're the only content
            if num_tokens < min_chunk_size and len(raw_chunks) > 1:
                logger.debug(
                    f"Skipping undersized chunk ({num_tokens} < {min_chunk_size}): "
                    f"{chunk['header'][:30]}..."
                )
                continue

            # Map engine ChunkResult → Adapter MarkdownChunkResult
            adapted: MarkdownChunkResult = {
                "id": chunk["id"],
                "doc_id": doc_id,
                "doc_index": doc_index,
                "chunk_index": chunk["chunk_index"],
                "num_tokens": num_tokens,
                "content": chunk["content"],
                "start_idx": chunk["metadata"]["start_idx"],
                "end_idx": chunk["metadata"]["end_idx"],
                "line_idx": 0,  # Computed below if needed
                "overlap_start_idx": None,
                "overlap_end_idx": None,
                # Hierarchy fields
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
            all_results.append(adapted)

    logger.info(
        f"Markdown hierarchy chunking complete: {len(markdown_text)} docs → "
        f"{len(all_results)} chunks"
    )
    return all_results
