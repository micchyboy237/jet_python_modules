# rag_module_v1/corpus.py

import json
from pathlib import Path
from typing import Any

from .schemas import Chunk


def load_corpus(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row.get("doc_id", row["chunk_id"].split("#")[0]),
                    doc_title=row.get("doc_title", row.get("doc_id", "")),
                    content=row["content"],
                    metadata=row.get("metadata", {}),
                )
            )

    return chunks


def metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if expected is None:
            continue

        actual = metadata.get(key)

        if key.endswith("_gte"):
            base_key = key.removesuffix("_gte")
            if metadata.get(base_key) is None or metadata[base_key] < expected:
                return False

        elif key.endswith("_lt"):
            base_key = key.removesuffix("_lt")
            if metadata.get(base_key) is None or metadata[base_key] >= expected:
                return False

        else:
            if actual != expected:
                return False

    return True


def filter_chunks(chunks: list[Chunk], filters: dict[str, Any]) -> list[Chunk]:
    if not filters:
        return chunks

    return [c for c in chunks if metadata_matches(c.metadata, filters)]
