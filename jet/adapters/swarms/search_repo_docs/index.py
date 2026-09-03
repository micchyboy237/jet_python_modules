"""Document loading and vector index construction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from .config import REQUIRED_EXTENSIONS, SearchConfig


def configure_settings(cfg: SearchConfig) -> None:
    """Set global LlamaIndex Settings from SearchConfig."""
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

    Settings.chunk_size = cfg.chunk_size
    Settings.chunk_overlap = cfg.chunk_overlap


def load_documents(data_dirs: List[str]) -> list:
    """Load documents from multiple directories with collision-safe IDs."""
    all_documents = []
    for dir_path in data_dirs:
        resolved = Path(dir_path).resolve()
        if not resolved.is_dir():
            print(
                f"[WARN] Skipping non-existent directory: {dir_path}", file=sys.stderr
            )
            continue

        reader = SimpleDirectoryReader(
            input_dir=str(resolved),
            recursive=True,
            filename_as_id=False,
            required_exts=REQUIRED_EXTENSIONS,
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

        all_documents.extend(docs)
        print(f"[INFO] Loaded {len(docs)} documents from {dir_path}")

    if not all_documents:
        raise FileNotFoundError(
            f"No documents found in any of the specified directories: {data_dirs}"
        )
    return all_documents


def build_index(cfg: SearchConfig) -> VectorStoreIndex:
    """Configure settings, load docs, and build the vector index."""
    configure_settings(cfg)
    documents = load_documents(cfg.data_dirs)
    return VectorStoreIndex.from_documents(documents, show_progress=True)
