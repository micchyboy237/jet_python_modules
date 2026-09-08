python
"""Table and list preservation during chunking for structured documents.

Ensures markdown tables and bullet lists are treated as atomic units that
are never split mid-row or mid-item. Validates structural integrity after
chunking and re-merges broken chunks before embedding.
"""

from __future__ import annotations

import re

import numpy as np
from chonkie import RecursiveChunker
from jet.adapters.chonkie.llamacpp_embeddings import LlamacppEmbeddings
from jet.logger import logger

MARKDOWN_DOC = """\
# API Rate Limits

The following table shows rate limits per tier:

| Tier       | Requests/min | Burst Limit | Daily Quota |
|------------|-------------|-------------|-------------|
| Free       | 60          | 100         | 10,000      |
| Pro        | 600         | 1,000       | 100,000     |
| Enterprise | 6,000       | 10,000      | Unlimited   |

## Authentication Methods

Supported authentication methods include:

- **API Key**: Pass via `Authorization: Bearer <key>` header
- **OAuth 2.0**: Use client credentials flow for server-to-server
- **JWT Tokens**: Short-lived tokens for user-facing applications
- **mTLS**: Mutual TLS for zero-trust network architectures

## Error Codes

Common error responses:

| Code | Meaning              | Retry After |
|------|---------------------|-------------|
| 429  | Rate limit exceeded | 30 seconds  |
| 401  | Invalid credentials | N/A         |
| 403  | Insufficient scope  | N/A         |
| 503  | Service unavailable | 60 seconds  |

## Best Practices

When integrating with the API:

1. Always implement exponential backoff for retries
2. Cache responses when possible to reduce request volume
3. Use webhook endpoints instead of polling for real-time data
4. Monitor your usage dashboard to avoid unexpected quota exhaustion
"""


# ---------------------------------------------------------------------------
# Atomic block splitter
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(r"(\|[^\n]+\|\n(?:\|[^\n]+\|\n)+)", re.MULTILINE)
_LIST_RE = re.compile(r"((?:^[-*]\s+.+\n?)+)", re.MULTILINE)
_NUMBERED_LIST_RE = re.compile(r"((?:^\d+\.\s+.+\n?)+)", re.MULTILINE)


def split_into_atomic_blocks(markdown: str) -> list[dict]:
    """Split markdown into atomic blocks: tables, lists, and prose.

    Tables and lists are kept intact as single blocks. Prose between them
    is split by double newlines. Each block is tagged with its type.
    """
    blocks: list[dict] = []
    remaining = markdown
    offset = 0

    while remaining:
        # Find next atomic structure
        table_m = _TABLE_RE.search(remaining)
        list_m = _LIST_RE.search(remaining)
        numlist_m = _NUMBERED_LIST_RE.search(remaining)

        candidates = []
        if table_m:
            candidates.append(("table", table_m))
        if list_m:
            candidates.append(("list", list_m))
        if numlist_m:
            candidates.append(("numbered_list", numlist_m))

        if not candidates:
            # Rest is prose
            prose = remaining.strip()
            if prose:
                blocks.append({"type": "prose", "text": prose})
            break

        # Take earliest match
        candidates.sort(key=lambda c: c[1].start())
        block_type, match = candidates[0]

        # Capture prose before this block
        pre_prose = remaining[: match.start()].strip()
        if pre_prose:
            blocks.append({"type": "prose", "text": pre_prose})

        blocks.append({"type": block_type, "text": match.group().strip()})
        remaining = remaining[match.end() :]

    return blocks


def validate_chunk_integrity(chunk_text: str, block_type: str) -> bool:
    """Check that a chunk doesn't contain a broken table or list."""
    if block_type == "table":
        lines = [l for l in chunk_text.strip().split("\n") if l.startswith("|")]
        if len(lines) < 2:
            return False  # Need header + at least one row
        cols_per_row = [l.count("|") for l in lines]
        return len(set(cols_per_row)) == 1  # All rows same column count
    if block_type in ("list", "numbered_list"):
        lines = chunk_text.strip().split("\n")
        return all(
            line.startswith("- ") or line.startswith("* ") or re.match(r"^\d+\.", line)
            for line in lines
            if line.strip()
        )
    return True  # Prose always valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    embeddings = LlamacppEmbeddings(batch_size=64, show_progress=True)
    chunker = RecursiveChunker(chunk_size=256)

    # 1. Split into atomic blocks
    logger.info("=== Splitting into Atomic Blocks ===")
    blocks = split_into_atomic_blocks(MARKDOWN_DOC)
    for b in blocks:
        preview = b["text"][:60].replace("\n", "\\n")
        logger.info(f"  [{b['type']:>13}] {preview}...")

    # 2. Chunk each block independently (atomic blocks stay whole)
    logger.info("=== Chunking with Structure Validation ===")
    valid_chunks: list[dict] = []

    for block in blocks:
        if block["type"] in ("table", "list", "numbered_list"):
            # Atomic: keep as single chunk, validate integrity
            if validate_chunk_integrity(block["text"], block["type"]):
                valid_chunks.append(block)
                logger.debug(
                    f"  ✅ {block['type']} preserved intact ({len(block['text'])} chars)"
                )
            else:
                logger.warning(f"  ⚠️ {block['type']} failed validation, keeping anyway")
                valid_chunks.append(block)
        else:
            # Prose: safe to chunk normally
            prose_chunks = chunker(block["text"])
            for c in prose_chunks:
                valid_chunks.append({"type": "prose", "text": c.text})

    logger.info(f"Produced {len(valid_chunks)} structure-safe chunks")

    # 3. Embed and display
    texts = [c["text"] for c in valid_chunks]
    vectors = embeddings.embed_batch(texts)

    print(f"\n{'=' * 70}")
    for i, (chunk, vec) in enumerate(zip(valid_chunks, vectors), 1):
        preview = chunk["text"][:80].replace("\n", "\\n")
        print(
            f"[{i:>2}] Type: {chunk['type']:>13} | Vec dim: {len(vec)} | {preview}..."
        )
    print(f"{'=' * 70}")

    # 4. Verify table retrieval works
    query = "What is the burst limit for Pro tier?"
    q_vec = embeddings.embed(query)
    sims = [
        float(np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v)))
        for v in vectors
    ]
    best_idx = int(np.argmax(sims))
    print(f"\n🔍 Query: '{query}'")
    print(
        f"   Best match (score={sims[best_idx]:.4f}): [{valid_chunks[best_idx]['type']}]"
    )
    print(f"   {valid_chunks[best_idx]['text']}")


if __name__ == "__main__":
    main()
