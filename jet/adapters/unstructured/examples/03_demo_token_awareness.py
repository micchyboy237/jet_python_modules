"""
03_demo_token_awareness.py
Demonstrates token estimation, auto chunk sizing, and budget enforcement.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from jet.adapters.unstructured.document_parser import (
    _HAS_LLAMA_CPP_UTILS,
    auto_chunk_size,
    chunk_rag_context,
    estimate_tokens,
)


def demo_token_estimation():
    """Compare heuristic vs model-aware token counting."""
    print("\n" + "-" * 50)
    print("TOKEN ESTIMATION")
    print("-" * 50)
    print(f"llama_cpp utils available: {_HAS_LLAMA_CPP_UTILS}")

    test_texts = [
        "Short sentence.",
        "A medium-length sentence with some technical terms like backpropagation.",
        "x = [i**2 for i in range(100)]  # list comprehension",
        "",  # Edge case: empty string
    ]

    print(f"\n{'Text':<55} {'Heuristic':>10} {'Model':>10}")
    print("-" * 77)
    for text in test_texts:
        heuristic = estimate_tokens(text, model=None)
        # When model=None and llama_cpp is available, it uses default LLM_MODEL
        model_count = (
            estimate_tokens(text, model="default") if _HAS_LLAMA_CPP_UTILS else "N/A"
        )
        display_text = text[:52] + "..." if len(text) > 55 else text
        print(f"{display_text:<55} {heuristic:>10} {str(model_count):>10}")


def demo_auto_chunk_sizing():
    """Show how auto_chunk_size derives from model context window."""
    print("\n" + "-" * 50)
    print("AUTO CHUNK SIZING")
    print("-" * 50)

    test_models = [None, "nonexistent_model_xyz", "qwen2.5:3b"]
    for model_key in test_models:
        size = auto_chunk_size(model_key)
        label = model_key if model_key else "(default)"
        print(f"  model={label:30s} → max_tokens={size}")


def demo_budget_enforcement():
    """Verify chunks never exceed the specified token budget."""
    print("\n" + "-" * 50)
    print("BUDGET ENFORCEMENT TEST")
    print("-" * 50)

    # Create elements that individually and collectively challenge the budget
    elements = [
        {
            "type": "NarrativeText",
            "text": "Word " * 200,
            "metadata": {},
            "element_id": "long-1",
        },
        {
            "type": "NarrativeText",
            "text": "Short.",
            "metadata": {},
            "element_id": "short-1",
        },
        {
            "type": "NarrativeText",
            "text": "Another moderately sized piece of text that should merge with neighbors.",
            "metadata": {},
            "element_id": "med-1",
        },
        {
            "type": "CodeSnippet",
            "text": "print('atomic')",
            "metadata": {},
            "element_id": "code-1",
        },
        {
            "type": "NarrativeText",
            "text": "Final paragraph with enough words to push past any reasonable boundary limit.",
            "metadata": {},
            "element_id": "final-1",
        },
    ]

    budgets = [30, 80, 200]
    for budget in budgets:
        chunks = chunk_rag_context(
            elements, max_tokens=budget, overlap_tokens=0, model=None
        )
        max_actual = max((c["token_count"] for c in chunks), default=0)
        violations = sum(1 for c in chunks if c["token_count"] > budget)
        status = "✅ PASS" if violations == 0 else f"❌ FAIL ({violations} violations)"
        print(
            f"\n  Budget={budget:4d} | Chunks={len(chunks):2d} | Max Actual={max_actual:4d} | {status}"
        )
        for i, c in enumerate(chunks):
            flag = " ⚠️" if c["token_count"] > budget else ""
            print(
                f"    [{i + 1}] tokens={c['token_count']:4d} strategy={c['strategy']}{flag}"
            )


def main():
    print("=" * 60)
    print("DEMO 03: Token Awareness & Budget Enforcement")
    print("=" * 60)

    demo_token_estimation()
    demo_auto_chunk_sizing()
    demo_budget_enforcement()

    print("\n" + "=" * 60)
    print("✅ Demo 03 Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
