"""
Demo: chunk_sentences & chunk_sentences_optimized
- Sentence-count / token-based chunking, overlap, separators preserved
- Optimized variant with caching + progress bar
"""

import logging

from jet.wordnet.examples.text_chunker.demo_utils import apply_mocks, print_section

# Suppress noisy debug logs during demo
logging.getLogger("jet").setLevel(logging.WARNING)

tc = apply_mocks()

# Pre-warm NLTK sentence tokenizer (downloads punkt if needed, only once)
import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def demo_basic():
    print_section("1. By Sentence Count")
    text = "First sentence is here. Second one follows. Third sentence now. Fourth is next. Fifth final one."
    for i, c in enumerate(tc.chunk_sentences(text, chunk_size=2, model=None)):
        print(f"  Chunk {i}: {c}")


def demo_overlap():
    print_section("2. With Overlap")
    text = (
        "Alpha is the first sentence. "
        "Beta comes in as second. "
        "Charlie is the third one. "
        "Delta follows as fourth. "
        "Echo is the fifth sentence. "
        "Foxtrot comes in sixth. "
        "Golf is number seven. "
        "Hotel is the eighth sentence."
    )
    chunks = tc.chunk_sentences(text, chunk_size=3, chunk_overlap=1, model=None)
    print(f"  {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {c}")


def demo_model():
    print_section("3. Token-Based (with model)")
    text = (
        "Short. A much longer sentence with more tokens. Another long one. Tiny. End."
    )
    for i, c in enumerate(tc.chunk_sentences(text, chunk_size=6, model="llama-3.2:3b")):
        print(f"  Chunk {i}: {c}")


def demo_model_overlap():
    print_section("4. Token-Based + Overlap")
    text = "Short. Longer sentence here with words. Another. Final."
    for i, c in enumerate(
        tc.chunk_sentences(text, chunk_size=5, chunk_overlap=1, model="llama-3.2:3b")
    ):
        print(f"  Chunk {i}: {c}")


def demo_optimized():
    print_section("5. Optimized (with caching)")
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    for i, c in enumerate(
        tc.chunk_sentences_optimized(
            text, chunk_size=2, chunk_overlap=1, model="llama-3.2:3b"
        )
    ):
        print(f"  Chunk {i}: {c}")


def demo_short():
    print_section("6. Short Text (fits one chunk)")
    text = "Just two sentences. That's all."
    chunks = tc.chunk_sentences(text, chunk_size=10, model=None)
    print(f"  {len(chunks)} chunk: {chunks[0]}")


def demo_batch():
    print_section("7. Batch + Progress Bar")
    texts = [
        f"Doc {i} first sentence. Second sentence here. Third one too."
        for i in range(5)
    ]
    print("  (Progress bar below)")
    chunks = tc.chunk_sentences_optimized(
        texts, chunk_size=3, model=None, show_progress=True
    )
    print(f"  Total: {len(chunks)}")


if __name__ == "__main__":
    for fn in [
        demo_basic,
        demo_overlap,
        demo_model,
        demo_model_overlap,
        demo_optimized,
        demo_short,
        demo_batch,
    ]:
        print(f"\n  >>> Running {fn.__name__}...", flush=True)
        fn()
    print_section("Done — chunk_sentences")
