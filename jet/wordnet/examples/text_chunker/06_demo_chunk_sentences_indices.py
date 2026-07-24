"""
Demo: chunk_sentences_with_indices & chunk_sentences_with_indices_optimized
- Returns (chunks, doc_indices) tuple
- Model/word-based, overlap, batch index tracking
- Optimized variant + edge cases
"""

from collections import Counter

from jet.wordnet.examples.text_chunker.demo_utils import apply_mocks, print_section

tc = apply_mocks()


def demo_basic():
    print_section("1. Basic + Doc Indices")
    text = "S1. S2. S3. S4."
    chunks, indices = tc.chunk_sentences_with_indices(text, chunk_size=2, model=None)
    for i, (c, idx) in enumerate(zip(chunks, indices)):
        print(f"  Chunk {i} (doc {idx}): {c}")


def demo_batch():
    print_section("2. Batch Index Tracking")
    texts = ["Doc0: S1. Doc0: S2. Doc0: S3.", "Doc1: S1. Doc1: S2. Doc1: S3. Doc1: S4."]
    chunks, indices = tc.chunk_sentences_with_indices(texts, chunk_size=2, model=None)
    for i, (c, idx) in enumerate(zip(chunks, indices)):
        print(f"  Chunk {i} → doc {idx}: {c}")


def demo_overlap():
    print_section("3. Overlap + Indices")
    text = "A. B. C. D. E. F."
    chunks, indices = tc.chunk_sentences_with_indices(
        text, chunk_size=3, chunk_overlap=1, model=None
    )
    for i, (c, idx) in enumerate(zip(chunks, indices)):
        print(f"  Chunk {i} (doc {idx}): {c}")


def demo_model():
    print_section("4. Model-Based + Indices")
    text = "Short. A longer sentence with more tokens here. Another long. Final."
    chunks, indices = tc.chunk_sentences_with_indices(
        text, chunk_size=6, chunk_overlap=1, model="llama-3.2:3b"
    )
    for i, (c, idx) in enumerate(zip(chunks, indices)):
        print(f"  Chunk {i} (doc {idx}): {c}")


def demo_optimized():
    print_section("5. Optimized Variant")
    texts = ["DocA: S1. DocA: S2. DocA: S3.", "DocB: S1. DocB: S2."]
    chunks, indices = tc.chunk_sentences_with_indices_optimized(
        texts, chunk_size=2, model="llama-3.2:3b"
    )
    dist = Counter(indices)
    print(f"  {len(chunks)} chunks, doc distribution: {dict(dist)}")
    for i, (c, idx) in enumerate(zip(chunks, indices)):
        print(f"  Chunk {i} → doc {idx}: {c}")


def demo_edge():
    print_section("6. Edge Case: Single-Sentence Docs")
    texts = ["Only one.", "Just this.", "Solo."]
    chunks, indices = tc.chunk_sentences_with_indices(texts, chunk_size=5, model=None)
    for i, (c, idx) in enumerate(zip(chunks, indices)):
        print(f"  Chunk {i} (doc {idx}): {c}")


def demo_progress():
    print_section("7. Progress Bar")
    texts = [f"Doc {i}: S1. Doc {i}: S2. Doc {i}: S3." for i in range(5)]
    print("  (Progress bar below)")
    chunks, indices = tc.chunk_sentences_with_indices_optimized(
        texts, chunk_size=2, model=None, show_progress=True
    )
    dist = Counter(indices)
    print(f"  Total: {len(chunks)} | Doc distribution: {dict(dist)}")


if __name__ == "__main__":
    for fn in [
        demo_basic,
        demo_batch,
        demo_overlap,
        demo_model,
        demo_optimized,
        demo_edge,
        demo_progress,
    ]:
        fn()
    print_section("Done — chunk_sentences_with_indices")
