"""
docling_chunker.py
Utilities for chunking DoclingDocument objects with token-aware overlap.
"""

import logging
from typing import List

from jet.adapters.llama_cpp.chunk_strategies._common import detect_text_overlap
from jet.adapters.llama_cpp.chunking_utils import _get_size_fn
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def chunk_docling_document(
    doc,
    model: str = "qwen3.5-uncensored:2b",
    max_tokens: int = 450,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Chunk a DoclingDocument by iterating through its structural elements.

    This approach preserves atomic elements like tables and code blocks
    while grouping smaller text items into coherent paragraphs with token-aware overlap.

    Args:
        doc: The DoclingDocument to chunk.
        model: Model key for tokenizer resolution.
        max_tokens: Maximum number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.

    Returns:
        A list of LangChain Documents.
    """
    chunks = []
    current_text_parts = []
    current_token_count = 0
    previous_chunk_text = ""
    size_fn = _get_size_fn(model)

    for item, level in doc.iterate_items():
        elem_type = type(item).__name__
        content = ""

        if hasattr(item, "text"):
            content = item.text
        elif hasattr(item, "orig"):
            content = item.orig

        if elem_type in ["TableItem", "CodeItem", "FormulaItem"]:
            if current_text_parts:
                chunk_text = "\n".join(current_text_parts)
                chunks.append(Document(page_content=chunk_text))
                previous_chunk_text = chunk_text
                current_text_parts = []
                current_token_count = 0

            if elem_type == "TableItem":
                content = item.export_to_html(doc)
                chunks.append(
                    Document(page_content=content, metadata={"type": elem_type})
                )
                previous_chunk_text = content
                continue

        if content:
            item_tokens = len(size_fn(content))

            if current_token_count + item_tokens > max_tokens and current_text_parts:
                chunk_text = "\n".join(current_text_parts)
                overlap_text = ""

                if chunk_overlap > 0 and previous_chunk_text:
                    overlap_candidate, _ = detect_text_overlap(
                        previous_chunk_text, chunk_text, size_fn
                    )
                    if overlap_candidate:
                        overlap_text = overlap_candidate

                if overlap_text:
                    current_text_parts = [overlap_text, content]
                    current_token_count = len(size_fn(overlap_text)) + item_tokens
                else:
                    current_text_parts = [content]
                    current_token_count = item_tokens

                chunks.append(Document(page_content=chunk_text))
                previous_chunk_text = chunk_text
            else:
                current_text_parts.append(content)
                current_token_count += item_tokens

    if current_text_parts:
        chunk_text = "\n".join(current_text_parts)
        chunks.append(Document(page_content=chunk_text))

    logger.info(
        f"Generated {len(chunks)} chunks from DoclingDocument (max_tokens={max_tokens}, overlap={chunk_overlap})."
    )
    return chunks
