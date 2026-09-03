"""Document loading, chunking, and vector index construction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike

from .chunking import chunk_documents
from .config import SearchConfig
from .embedding import LlamaCppEmbedding

# ---------------------------------------------------------------------------
# File filtering
# ---------------------------------------------------------------------------

# Structured/binary formats that are valid UTF-8 but fail or produce
# meaningless embeddings when treated as plain-text documents.
_EXCLUDED_EXTENSIONS = {
    # Structured data
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".arrow",
    # Databases / serialized objects
    ".sqlite",
    ".db",
    ".pickle",
    ".pkl",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    # Office / PDF
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    # Archives / compressed
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    # Compiled / binary artifacts
    ".whl",
    ".egg",
    ".so",
    ".dylib",
    ".dll",
    ".pyc",
    ".pyo",
    ".class",
    ".o",
    ".a",
    # Fonts
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    # Media
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
}


def _is_text_file(path: Path, sample_bytes: int = 8192) -> bool:
    """Return True if file is a supported text document (not binary or structured data)."""
    # Fast-path: reject known non-text extensions before reading any bytes
    if path.suffix.lower() in _EXCLUDED_EXTENSIONS:
        return False

    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_bytes)
        # Null bytes indicate binary content
        if b"\x00" in chunk:
            return False
        # Must decode as valid UTF-8
        chunk.decode("utf-8", errors="strict")
        return True
    except (UnicodeDecodeError, OSError):
        return False


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_documents(data_dirs: List[str]) -> list:
    """Load ALL text files from directories, skipping binaries and structured data."""
    all_documents = []
    for dir_path in data_dirs:
        resolved = Path(dir_path).resolve()
        if not resolved.is_dir():
            print(
                f"[WARN] Skipping non-existent directory: {dir_path}", file=sys.stderr
            )
            continue

        text_files = [
            str(p)
            for p in sorted(resolved.rglob("*"))
            if p.is_file() and _is_text_file(p)
        ]

        if not text_files:
            print(f"[WARN] No text files found in {dir_path}", file=sys.stderr)
            continue

        reader = SimpleDirectoryReader(
            input_files=text_files,
            filename_as_id=False,
        )
        docs = reader.load_data()

        for doc in docs:
            source_file = Path(doc.metadata.get("file_path", ""))
            try:
                rel_path = source_file.relative_to(resolved)
            except ValueError:
                rel_path = source_file.name
            doc.id_ = f"{resolved.name}/{rel_path}"
            doc.metadata["source_dir"] = str(resolved)
            doc.metadata["file_id"] = doc.id_
            doc.metadata["extension"] = source_file.suffix.lower()

        all_documents.extend(docs)
        print(f"[INFO] Loaded {len(docs)} text documents from {dir_path}")

    if not all_documents:
        raise FileNotFoundError(f"No text documents found in any of: {data_dirs}")
    return all_documents


# ---------------------------------------------------------------------------
# Settings configuration
# ---------------------------------------------------------------------------


def configure_settings(cfg: SearchConfig) -> None:
    """Set global LlamaIndex Settings, reusing jet.adapters config values."""
    llm_model, llm_base = cfg.resolve_llm()
    emb_model, emb_base, emb_dims, q_prefix, d_prefix = cfg.resolve_embed()

    # --- LLM ---
    additional_kwargs = {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}}
    }

    Settings.llm = OpenAILike(
        model=llm_model,
        api_base=llm_base,
        api_key="not-needed",
        is_chat_model=True,
        timeout=120,
        additional_kwargs=additional_kwargs,
    )

    # --- Embeddings (backed by jet.adapters.llama_cpp.embed_utils) ---
    Settings.embed_model = LlamaCppEmbedding(
        model_name=emb_model,
        query_prefix=q_prefix,
        text_prefix=d_prefix,
        batch_size=64,
        max_workers=6,
        show_progress=True,
    )

    # Chunk size/overlap are handled dynamically by chunk_documents() via
    # token budget from the live embedding model's context window.
    # These globals serve as fallbacks only.
    Settings.chunk_size = cfg.chunk_size
    Settings.chunk_overlap = cfg.chunk_overlap


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def build_index(cfg: SearchConfig) -> VectorStoreIndex:
    """Configure settings, load docs, chunk per-file-type, and build index."""
    configure_settings(cfg)
    documents = load_documents(cfg.data_dirs)

    emb_model, emb_base, _, _, _ = cfg.resolve_embed()
    nodes = chunk_documents(
        documents,
        embed_model=emb_model,
        embed_base_url=emb_base,
    )

    return VectorStoreIndex(nodes=nodes, show_progress=True)
