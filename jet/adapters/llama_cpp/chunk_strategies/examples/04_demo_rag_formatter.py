# jet_python_modules/jet/adapters/llama_cpp/chunk_strategies/examples/04_demo_rag_formatter.py
"""Demo: RAG formatter with content type detection.

Compares heuristic auto-detection against explicit content_type metadata.
Shows false positive risks and demonstrates safe marker wrapping.
"""

import logging

from jet.adapters.llama_cpp.chunk_strategies import (
    detect_content_type,
    format_chunks_for_rag,
    get_chunker,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MIXED_CHUNKS = [
    "The function definition describes how to import data from external sources.",
    "```python\ndef load_data(path):\n    return json.load(open(path))\n```",
    "| Name | Age | Role |\n|------|-----|------|\n| Alice | 30 | Eng |",
    "Regular prose paragraph with no special structure or formatting at all.",
    "The import of this policy was approved by the board last quarter.",
]

EXPLICIT_TYPES = ["prose", "code", "table", "prose", "prose"]


def main() -> None:
    print("=" * 60)
    print("🔍 PART 1: Heuristic Auto-Detection (risky)")
    print("=" * 60)

    detected = [detect_content_type(c) for c in MIXED_CHUNKS]
    for i, (chunk, dtype) in enumerate(zip(MIXED_CHUNKS, detected)):
        preview = chunk[:60].replace("\n", "\\n")
        print(f"  [{i}] detected={dtype:6s} | {preview}...")

    auto_formatted = format_chunks_for_rag(MIXED_CHUNKS)  # triggers warning log
    print("\nAuto-formatted output:")
    for i, fmt in enumerate(auto_formatted):
        print(f"  [{i}] {fmt[:80]}...")

    print(f"\n{'=' * 60}")
    print("✅ PART 2: Explicit Content Types (safe)")
    print("=" * 60)

    explicit_formatted = format_chunks_for_rag(
        MIXED_CHUNKS, content_types=EXPLICIT_TYPES
    )
    for i, fmt in enumerate(explicit_formatted):
        print(f"  [{i}] {fmt[:80]}...")

    print(f"\n{'=' * 60}")
    print("🔗 PART 3: End-to-End Pipeline (chunk → format)")
    print("=" * 60)

    chunker = get_chunker("sentence", model="qwen3.5:2b")
    prose = (
        "Retrieval-augmented generation reduces hallucination. "
        "It grounds responses in verified source material. "
        "The architecture has three core components."
    )
    chunks = chunker.chunk(
        text=prose, chunk_size=32, chunk_overlap=4, min_chunk_size=8, buffer=2
    )
    formatted = format_chunks_for_rag(chunks, content_types=["prose"] * len(chunks))

    print(f"  Chunks: {len(chunks)} → Formatted: {len(formatted)}")
    for i, fmt in enumerate(formatted):
        print(f"  [{i}] {fmt}")

    print(f"\n{'=' * 60}")
    logger.info("Demo complete. Compare Part 1 vs Part 2 to see heuristic risks.")


if __name__ == "__main__":
    main()
