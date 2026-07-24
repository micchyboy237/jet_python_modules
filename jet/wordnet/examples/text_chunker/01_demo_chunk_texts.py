"""
Demo: chunk_texts
- Token-based (model) / word-based (no model)
- chunk_overlap, min_chunk_size, buffer, strict_sentences
- Batch + progress bar
"""

from jet.wordnet.examples.text_chunker.demo_utils import apply_mocks, print_section

tc = apply_mocks()


def demo_basic():
    print_section("1. Word-Based (no model)")
    text = "This is a simple test. It has multiple sentences. We chunk it properly."
    for i, c in enumerate(tc.chunk_texts(text, chunk_size=5, model=None)):
        print(f"  Chunk {i}: [{len(c.split())} words] {c}")


def demo_model():
    print_section("2. Token-Based (with model)")
    text = "Hello world. This is a test document with several sentences. " * 2
    for i, c in enumerate(tc.chunk_texts(text, chunk_size=6, model="llama-3.2:3b")):
        print(f"  Chunk {i}: {c}")


def demo_overlap():
    print_section("3. Overlap")
    text = "S1. S2. S3. S4. S5. S6. S7. S8."
    chunks = tc.chunk_texts(text, chunk_size=4, chunk_overlap=2, model=None)
    print(f"  {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {c}")


def demo_min_size():
    print_section("4. Min Chunk Size")
    text = "Short. Another short. This is a much longer sentence kept. Tiny. Small."
    for i, c in enumerate(
        tc.chunk_texts(text, chunk_size=10, min_chunk_size=5, model=None)
    ):
        print(f"  Chunk {i}: [{len(c.split())} words] {c}")


def demo_buffer():
    print_section("5. Buffer Reservation")
    text = "A " + "word " * 50
    no_buf = len(tc.chunk_texts(text, chunk_size=10, buffer=0, model=None))
    with_buf = len(tc.chunk_texts(text, chunk_size=10, buffer=3, model=None))
    print(f"  Without buffer: {no_buf} | With buffer(3): {with_buf} (more chunks)")


def demo_strict():
    print_section("6. Strict Sentences")
    text = "Short. Another. " + "word " * 30 + "Final sentence."
    for i, c in enumerate(
        tc.chunk_texts(text, chunk_size=10, strict_sentences=True, model=None)
    ):
        print(f"  Chunk {i}: [{len(c.split())} words] {c[:80]}...")


def demo_batch():
    print_section("7. Batch + Progress Bar")
    texts = [f"Document {i} text. Another sentence." for i in range(5)]
    print("  (Progress bar below)")
    chunks = tc.chunk_texts(texts, chunk_size=10, model=None, show_progress=True)
    print(f"  Total chunks: {len(chunks)}")


if __name__ == "__main__":
    for fn in [
        demo_basic,
        demo_model,
        demo_overlap,
        demo_min_size,
        demo_buffer,
        demo_strict,
        demo_batch,
    ]:
        fn()
    print_section("Done — chunk_texts")
