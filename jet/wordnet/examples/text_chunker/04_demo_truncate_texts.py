"""
Demo: truncate_texts & truncate_texts_fast
- Token-based truncation, auto max_tokens, strict_sentences
- Batch, fast variant with progress bar
"""

from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.wordnet.examples.text_chunker.demo_utils import apply_mocks, print_section

tc = apply_mocks()

DEFAULT_MODEL = EMBED_MODEL


def demo_basic():
    print_section("1. Basic Truncation")
    text = "This is a test document with many words that should be truncated."
    result = tc.truncate_texts(text, model=DEFAULT_MODEL, max_tokens=5)
    print(f"  Original: {text}")
    print(f"  Truncated (5t): {result[0]}")


def demo_auto():
    print_section("2. Auto Max Tokens (from model context)")
    text = "word " * 10
    result = tc.truncate_texts(text, model=DEFAULT_MODEL, max_tokens=None)
    print(f"  Model ctx: 2048 | Text: {len(text.split())} words → fits, returned as-is")
    print(f"  {result[0][:80]}...")


def demo_strict():
    print_section("3. Strict Sentences")
    text = "First sentence. Second sentence goes beyond limit. Third cut off."
    result = tc.truncate_texts(
        text, model=DEFAULT_MODEL, max_tokens=6, strict_sentences=True
    )
    print(f"  {result[0]}")


def demo_non_strict():
    print_section("4. Non-Strict (Raw Token Cut)")
    text = "Sentence one. Sentence two longer gets cut mid-way maybe."
    result = tc.truncate_texts(
        text, model=DEFAULT_MODEL, max_tokens=5, strict_sentences=False
    )
    print(f"  {result[0]}")


def demo_batch():
    print_section("5. Batch")
    texts = ["Short.", "Medium length text here.", "Tiny."]
    for i, r in enumerate(tc.truncate_texts(texts, model=DEFAULT_MODEL, max_tokens=4)):
        print(f"  Doc {i}: {texts[i][:30]}... → {r}")


def demo_fast():
    print_section("6. Fast + Progress Bar")
    texts = [f"Doc {i} text to truncate. Another sentence." for i in range(5)]
    print("  (Progress bar below)")
    results = tc.truncate_texts_fast(
        texts, model=DEFAULT_MODEL, max_tokens=6, show_progress=True
    )
    print(f"  {len(results)} results")


def demo_fast_strict():
    print_section("7. Fast + Strict Sentences")
    text = "A here. B is longer might get dropped. C too."
    results = tc.truncate_texts_fast(
        text, model=DEFAULT_MODEL, max_tokens=6, strict_sentences=True
    )
    print(f"  {results[0] if results else '(empty)'}")


if __name__ == "__main__":
    for fn in [
        demo_basic,
        demo_auto,
        demo_strict,
        demo_non_strict,
        demo_batch,
        demo_fast,
        demo_fast_strict,
    ]:
        fn()
    print_section("Done — truncate_texts")
