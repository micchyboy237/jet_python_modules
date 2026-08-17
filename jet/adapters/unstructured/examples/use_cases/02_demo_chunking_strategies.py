"""
02_demo_chunking_strategies.py
Demonstrates chunking strategy selection based on document structure classification.
Uses synthetic elements to avoid file I/O dependencies.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from jet.adapters.unstructured.document_parser import (
    chunk_rag_context,
    classify_structure,
)


def make_elem(elem_type: str, text: str, **meta_overrides) -> Dict[str, Any]:
    """Helper to create a minimal unstructured-compatible element dict."""
    meta = {"filename": "synthetic.pdf", "page_number": 1}
    meta.update(meta_overrides)
    return {
        "type": elem_type,
        "text": text,
        "metadata": meta,
        "element_id": f"e-{id(text) % 10000}",
    }


def demo_structured():
    """Document with clear section headers → 'structured' strategy."""
    elements = [
        make_elem("Title", "Annual Report 2025"),
        make_elem("NarrativeText", "This report covers fiscal year 2025 performance."),
        make_elem("Header", "Financial Summary"),
        make_elem("NarrativeText", "Revenue increased by 15% compared to prior year."),
        make_elem(
            "NarrativeText",
            "Operating margins improved due to cost optimization initiatives.",
        ),
        make_elem("Header", "Risk Factors"),
        make_elem(
            "NarrativeText", "Market volatility remains a primary concern for Q3."
        ),
        make_elem(
            "Table", "| Metric | Value |\n|--------|-------|\n| Revenue | $1.2B |"
        ),
    ]
    return elements


def demo_flat_narrative():
    """Pure narrative text without headers → 'flat_narrative' strategy."""
    elements = [
        make_elem("NarrativeText", "The quick brown fox jumps over the lazy dog."),
        make_elem("NarrativeText", "Pack my box with five dozen liquor jugs."),
        make_elem("NarrativeText", "How vexingly quick daft zebras jump."),
        make_elem("NarrativeText", "The five boxing wizards jump quickly."),
        make_elem("NarrativeText", "Sphinx of black quartz, judge my vow."),
    ]
    return elements


def demo_atomic_flat():
    """Code/formula heavy content without sections → 'atomic_flat' strategy."""
    elements = [
        make_elem("NarrativeText", "Below are key formulas used in the model."),
        make_elem("Formula", "E = mc^2"),
        make_elem("NarrativeText", "The energy-mass equivalence is fundamental."),
        make_elem(
            "CodeSnippet",
            "def compute_loss(y, y_hat):\n    return ((y - y_hat)**2).mean()",
        ),
        make_elem("NarrativeText", "Mean squared error provides gradient stability."),
    ]
    return elements


def run_demo(name: str, elements: List[Dict[str, Any]], max_tokens: int = 80):
    """Run chunking on a synthetic element set and display results."""
    structure = classify_structure(elements)
    print(f"\n{'=' * 60}")
    print(f"STRATEGY DEMO: {name}")
    print(
        f"Classified as: '{structure}' | Elements: {len(elements)} | Max Tokens: {max_tokens}"
    )
    print(f"{'=' * 60}")

    chunks = chunk_rag_context(
        elements, max_tokens=max_tokens, overlap_tokens=10, model=None
    )

    print(f"Produced {len(chunks)} chunks:\n")
    for i, chunk in enumerate(chunks):
        text_preview = chunk["text"].replace("\n", "\\n")[:90]
        print(
            f"  [{i + 1}] strategy={chunk['strategy']:20s} tokens={chunk['token_count']:3d} | {text_preview}..."
        )


def main():
    print("=" * 60)
    print("DEMO 02: Chunking Strategy Selection")
    print("=" * 60)
    print("\nThis demo uses synthetic elements to show how classify_structure()")
    print("routes documents to different chunking strategies.")

    run_demo("Structured Document (Headers + Narrative + Table)", demo_structured())
    run_demo("Flat Narrative (No Headers)", demo_flat_narrative())
    run_demo("Atomic Flat (Code + Formula + Narrative)", demo_atomic_flat())

    print("\n" + "=" * 60)
    print("✅ Demo 02 Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
