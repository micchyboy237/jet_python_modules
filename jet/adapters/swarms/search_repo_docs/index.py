"""Document loading, chunking, and vector index construction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from .chunking import chunk_documents
from .config import SearchConfig


def configure_settings(cfg: SearchConfig) -> None:
    """Set global LlamaIndex Settings, reusing jet.adapters config values."""
    llm_model, llm_base = cfg.resolve_llm()
    emb_model, emb_base, emb_dims, q_prefix, d_prefix = cfg.resolve_embed()

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

    Settings.embed_model = OpenAIEmbedding(
        model_name=emb_model,
        api_base=emb_base,
        api_key="not-needed",
        dimensions=emb_dims,
        query_prefix=q_prefix,
        text_prefix=d_prefix,
        embed_batch_size=32,
    )

    # Chunk size/overlap now handled by chunk_documents() via token budget;
    # these globals serve as fallbacks only
    Settings.chunk_size = cfg.chunk_size
    Settings.chunk_overlap = cfg.chunk_overlap


def _is_text_file(path: Path, sample_bytes: int = 8192) -> bool:
    """Return True if file appears to be text (not binary)."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_bytes)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8", errors="strict")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def load_documents(data_dirs: List[str]) -> list:
    """Load ALL text files from directories, skipping binaries."""
    all_documents = []
    for dir_path in data_dirs:
        resolved = Path(dir_path).resolve()
        if not resolved.is_dir():
            print(
                f"[WARN] Skipping non-existent directory: {dir_path}", file=sys.stderr
            )
            continue

        text_files = [
            str(p) for p in resolved.rglob("*") if p.is_file() and _is_text_file(p)
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


def build_index(cfg: SearchConfig) -> VectorStoreIndex:
    """Configure settings, load docs, chunk per-file-type, and build index."""
    configure_settings(cfg)
    documents = load_documents(cfg.data_dirs)

    emb_model, emb_base, _, _, _ = cfg.resolve_embed()
    nodes = chunk_documents(documents, embed_model=emb_model, embed_base_url=emb_base)

    return VectorStoreIndex(nodes=nodes, show_progress=True)
