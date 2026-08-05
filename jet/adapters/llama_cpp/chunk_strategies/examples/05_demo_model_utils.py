# jet_python_modules/jet/adapters/llama_cpp/chunk_strategies/examples/05_demo_model_utils.py
"""Demo: Model utilities for optimal chunk sizing and safe token counting.

Validates get_optimal_chunk_size across different percentage/clamp configs
and demonstrates estimate_tokens_safe fallback behavior.
"""

import logging

from jet.adapters.llama_cpp.chunk_strategies import (
    estimate_tokens_safe,
    get_optimal_chunk_size,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

MODEL = "qwen3.5:2b"

SAMPLE_TEXTS = [
    "",
    "Short text.",
    "A" * 1000,
    "Retrieval-augmented generation combines external knowledge with language models.",
]


def main() -> None:
    print("=" * 60)
    print(f"📐 Optimal Chunk Size for '{MODEL}'")
    print("=" * 60)

    configs = [
        {"ctx_percentage": 0.05, "min_size": 64, "max_size": 512},
        {"ctx_percentage": 0.1, "min_size": 128, "max_size": 1024},
        {"ctx_percentage": 0.25, "min_size": 256, "max_size": 2048},
        {"ctx_percentage": 0.5, "min_size": 128, "max_size": 512},
    ]

    for cfg in configs:
        size = get_optimal_chunk_size(MODEL, **cfg)
        print(f"  {cfg} → {size} tokens")

    # Test invalid percentage (should warn and clamp)
    print("\n⚠️  Invalid ctx_percentage test:")
    bad_size = get_optimal_chunk_size(MODEL, ctx_percentage=1.5)
    print(f"  ctx_percentage=1.5 → {bad_size} tokens (clamped)")

    print(f"\n{'=' * 60}")
    print("🔢 Safe Token Estimation")
    print("=" * 60)

    for text in SAMPLE_TEXTS:
        count = estimate_tokens_safe(text, model=MODEL)
        preview = text[:40].replace("\n", "\\n") or "(empty)"
        print(f"  '{preview}' → {count} tokens")

    # Test with None model (char-based fallback)
    print("\n🔄 Fallback mode (model=None):")
    for text in SAMPLE_TEXTS[1:]:
        count = estimate_tokens_safe(text, model=None)
        preview = text[:40].replace("\n", "\\n")
        print(f"  '{preview}' → {count} tokens (char-fallback)")

    print(f"\n{'=' * 60}")
    logger.info("Demo complete. Review DEBUG logs for clamping/fallback details.")


if __name__ == "__main__":
    main()
