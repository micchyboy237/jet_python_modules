import re
import uuid
from typing import Callable

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.logger import logger
from jet.wordnet.sentence import split_sentences
from tqdm import tqdm

from .tokenization import _get_size_fn
from .types import MarkdownChunkResult

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _resolve_parent(
    level: int, header_stack: list[dict]
) -> tuple[str | None, int | None, str | None]:
    """Find nearest ancestor with lower level in single pass."""
    for h in reversed(header_stack):
        if h["level"] < level:
            return h["text"], h["level"], h["header_doc_id"]
    return None, None, None


def chunk_markdown_hierarchy(
    markdown_text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    buffer: int = 0,
    min_chunk_size: int = 32,
    show_progress: bool = False,
) -> list[str]:
    """Chunk markdown text respecting header hierarchy. Returns flat list of strings.

    Each returned string includes the header line followed by body content,
    suitable for embedding or generation contexts where hierarchy matters.
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
    # Combine header + content for simple string output
    chunks: list[str] = []
    for r in results:
        if r["header"]:
            chunks.append(f"{r['header']}\n{r['content']}")
        else:
            chunks.append(r["content"])
    return chunks


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
    """Chunk markdown documents preserving header hierarchy and source indices.

    Args:
        markdown_text: Single markdown string or list of strings.
        chunk_size: Max tokens per chunk (header tokens count toward limit).
        chunk_overlap: Number of overlapping tokens between consecutive chunks
                       within the same section.
        model: Model key for tokenizer.
        ids: Optional document IDs.
        buffer: Reserved token space to avoid exceeding chunk_size.
        min_chunk_size: Minimum tokens for a chunk to be kept (unless it's the only one).
        show_progress: Show progress bar.

    Returns:
        List of MarkdownChunkResult dicts with rich metadata.
    """
    if isinstance(markdown_text, str):
        texts = [markdown_text]
        doc_indices = [0]
    else:
        texts = markdown_text
        doc_indices = list(range(len(texts)))

    if min_chunk_size > chunk_size:
        min_chunk_size = chunk_size

    size_fn = _get_size_fn(model)
    effective_chunk_size = chunk_size - buffer

    all_results: list[MarkdownChunkResult] = []

    doc_iter = tqdm(
        zip(doc_indices, texts),
        total=len(texts),
        desc="Chunking markdown hierarchy",
        disable=not show_progress,
    )

    for doc_idx, text in doc_iter:
        if not text or not text.strip():
            continue

        doc_id = ids[doc_idx] if ids and doc_idx < len(ids) else str(uuid.uuid4())
        doc_chunks = _chunk_single_markdown(
            text=text,
            doc_id=doc_id,
            doc_index=doc_idx,
            size_fn=size_fn,
            effective_chunk_size=effective_chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            model=model,
        )
        all_results.extend(doc_chunks)

    logger.info(
        f"Markdown hierarchy chunking complete: {len(texts)} docs → {len(all_results)} chunks"
    )
    return all_results


def _chunk_single_markdown(
    text: str,
    doc_id: str,
    doc_index: int,
    size_fn: Callable,
    effective_chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    model: str | LLAMACPP_KEYS,
) -> list[MarkdownChunkResult]:
    """Internal worker for chunking a single markdown document."""
    results: list[MarkdownChunkResult] = []
    header_stack: list[dict] = []
    section_counter = -1

    # Current accumulator state
    current_header = ""
    current_header_tokens = 0
    current_level = 0
    current_parent_header: str | None = None
    current_parent_level: int | None = None
    current_parent_id: str | None = None
    current_header_doc_id = ""

    content_parts: list[str] = []
    current_token_count = 0

    # Index tracking
    header_abs_start = 0
    abs_start = 0
    abs_end = 0
    body_abs_start = 0
    body_abs_end = 0

    def flush_chunk():
        nonlocal \
            content_parts, \
            current_token_count, \
            abs_start, \
            abs_end, \
            body_abs_start, \
            body_abs_end

        if not content_parts:
            return

        content_str = "\n".join(content_parts).strip()
        total_tokens = current_token_count + current_header_tokens

        # Skip undersized chunks unless it's the first/only chunk for this section
        if total_tokens < min_chunk_size and results:
            # Merge with previous chunk if same header_doc_id
            prev = results[-1]
            if prev["header_doc_id"] == current_header_doc_id:
                prev["content"] += "\n" + content_str
                prev["num_tokens"] = (
                    len(size_fn(prev["content"])) + current_header_tokens
                )
                prev["metadata"]["end_idx"] = abs_end
                prev["metadata"]["body_end_idx"] = body_abs_end
                return

        chunk: MarkdownChunkResult = {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "doc_index": doc_index,
            "parent_id": current_parent_id,
            "header_doc_id": current_header_doc_id,
            "section_index": section_counter,
            "chunk_index": sum(
                1 for r in results if r["header_doc_id"] == current_header_doc_id
            ),
            "num_tokens": total_tokens,
            "header": current_header,
            "parent_header": current_parent_header,
            "content": content_str,
            "level": current_level,
            "parent_level": current_parent_level,
            "metadata": {
                "start_idx": abs_start,
                "end_idx": abs_end,
                "body_start_idx": body_abs_start,
                "body_end_idx": body_abs_end,
            },
        }
        results.append(chunk)

        # Handle overlap for next chunk in same section
        if chunk_overlap > 0 and content_parts:
            # Get raw text of last N tokens
            overlap_text = _get_last_n_tokens_text(
                content_str, chunk_overlap, size_fn, model
            )
            if overlap_text:
                content_parts = [overlap_text]
                current_token_count = len(size_fn(overlap_text))
            else:
                content_parts = []
                current_token_count = 0
        else:
            content_parts = []
            current_token_count = 0

        # Reset body indices but keep header start
        abs_start = header_abs_start
        abs_end = header_abs_start
        body_abs_start = 0
        body_abs_end = 0

    def get_last_n_tokens_text(
        text: str, n: int, sf: Callable, m: str | LLAMACPP_KEYS
    ) -> str:
        """Extract decoded text of last n tokens."""
        if n <= 0 or not text:
            return ""
        tokens = sf(text)
        if len(tokens) <= n:
            return text
        # We need to decode just the last n tokens
        from .tokenization import _decode_tokens

        return _decode_tokens(tokens[-n:], m).strip()

    pos = 0
    lines = text.split("\n")

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        stripped = line.strip()
        header_match = HEADER_RE.match(stripped)

        if header_match:
            flush_chunk()
            section_counter += 1

            level = len(header_match.group(1))
            header_text = stripped

            # Prune stack to find parent
            header_stack = [h for h in header_stack if h["level"] < level]
            new_header_doc_id = str(uuid.uuid4())
            header_stack.append(
                {
                    "level": level,
                    "text": header_text,
                    "header_doc_id": new_header_doc_id,
                }
            )

            p_header, p_level, p_id = _resolve_parent(level, header_stack)

            current_header = header_text
            current_header_tokens = len(size_fn(header_text))
            current_level = level
            current_parent_header = p_header
            current_parent_level = p_level
            current_parent_id = p_id
            current_header_doc_id = new_header_doc_id

            header_abs_start = pos
            abs_start = pos
            abs_end = pos + len(line)
            body_abs_start = 0
            body_abs_end = 0

            pos += line_len
            continue

        if stripped:
            sentences = split_sentences(stripped)
            search_offset = 0

            for sent in sentences:
                sent_stripped = sent.strip()
                if not sent_stripped:
                    continue

                idx = line.find(sent_stripped, search_offset)
                if idx == -1:
                    # Fallback: use whole line if alignment fails
                    logger.warning(
                        f"Sentence alignment failed in doc {doc_id}, using full line"
                    )
                    sent_stripped = stripped
                    idx = 0

                abs_sent_start = pos + idx
                abs_sent_end = abs_sent_start + len(sent_stripped)
                sent_tokens = len(size_fn(sent_stripped))

                projected = current_token_count + sent_tokens + current_header_tokens

                if projected > effective_chunk_size and content_parts:
                    flush_chunk()

                if not content_parts or body_abs_start == 0:
                    abs_start = header_abs_start
                    body_abs_start = abs_sent_start

                content_parts.append(sent_stripped)
                current_token_count += sent_tokens
                abs_end = abs_sent_end
                body_abs_end = abs_sent_end
                search_offset = idx + len(sent_stripped)

        pos += line_len

    flush_chunk()
    return results
