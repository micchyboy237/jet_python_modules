"""Per-file-type chunking with token-aware sizing via jet.adapters."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from llama_index.core.node_parser import (
    CodeSplitter,
    JSONNodeParser,
    MarkdownNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import Document

# Extension → parser mapping
_MARKDOWN_EXTS = {".md", ".mdx"}
_CODE_EXTS = {".py"}
_JSON_EXTS = {".json", ".yaml", ".yml"}
_NOTEBOOK_EXTS = {".ipynb"}


def _get_token_budget(embed_model: str, base_url: Optional[str] = None) -> int:
    """Derive chunk token budget from the live embedding model's context window."""
    try:
        sizes = get_model_ctx_embd_size(embed_model, base_url=base_url)
        n_ctx = sizes.get("ctx", 0) or sizes.get("ctx_train", 0)
        if n_ctx > 0:
            # 75% of context window leaves room for query + metadata tokens
            return max(256, int(n_ctx * 0.75))
    except Exception as e:
        print(
            f"[WARN] Could not query embed model context ({e}), using default",
            file=sys.stderr,
        )
    return 1024  # Safe fallback


def _build_parsers(embed_model: str, base_url: Optional[str] = None) -> dict:
    """Build parser instances sized to the live embedding model."""
    token_budget = _get_token_budget(embed_model, base_url)
    overlap = max(32, int(token_budget * 0.15))

    # Estimate lines-per-chunk for code: ~13 tokens/line average for Python
    code_chunk_lines = max(20, token_budget // 13)
    code_overlap_lines = max(5, code_chunk_lines // 4)

    return {
        "markdown": MarkdownNodeParser(),
        "code": CodeSplitter(
            language="python",
            chunk_lines=code_chunk_lines,
            chunk_lines_overlap=code_overlap_lines,
            max_chars=token_budget * 4,  # ~4 chars/token upper bound
        ),
        "json": JSONNodeParser(),
        "fallback": SentenceSplitter(
            chunk_size=token_budget,
            chunk_overlap=overlap,
        ),
    }


def _select_parser(ext: str, parsers: dict):
    """Select the appropriate parser for a file extension."""
    ext = ext.lower()
    if ext in _MARKDOWN_EXTS or ext in _NOTEBOOK_EXTS:
        return parsers["markdown"]
    if ext in _CODE_EXTS:
        return parsers["code"]
    if ext in _JSON_EXTS:
        return parsers["json"]
    return parsers["fallback"]


def chunk_documents(
    documents: List[Document],
    embed_model: str,
    embed_base_url: Optional[str] = None,
) -> list:
    """
    Apply per-file-type chunking to a list of Documents.

    Uses jet.adapters.llama_cpp.model_utils to derive token-aware chunk sizes
    from the live embedding model's context window, and jet.adapters.llama_cpp
    .token_utils for accurate token counting when needed.
    """
    parsers = _build_parsers(embed_model, embed_base_url)
    all_nodes = []

    for doc in documents:
        ext = Path(doc.metadata.get("file_path", "")).suffix.lower()
        parser = _select_parser(ext, parsers)

        try:
            nodes = parser.get_nodes_from_documents([doc])
        except Exception as e:
            # Fallback to sentence splitter if specialized parser fails
            print(
                f"[WARN] Parser failed for {doc.metadata.get('file_id', '?')} "
                f"({ext}): {e}. Falling back to SentenceSplitter.",
                file=sys.stderr,
            )
            nodes = parsers["fallback"].get_nodes_from_documents([doc])

        all_nodes.extend(nodes)

    print(f"[INFO] Chunked {len(documents)} documents into {len(all_nodes)} nodes")
    return all_nodes
