"""
Demo: chunk_texts_sliding_window & chunk_texts_sliding_window_fast
- Fixed-size sliding window, step_size, model/word-based
- Min chunk merge, custom IDs, error handling, fast variant
"""

from jet.wordnet.examples.text_chunker.demo_utils import apply_mocks, print_section

tc = apply_mocks()


def demo_basic():
    print_section("1. Word-Based")
    text = "A B C D E F G H I J K L M N O P"
    for c in tc.chunk_texts_sliding_window(text, chunk_size=5, step_size=3, model=None):
        print(
            f"  #{c['chunk_index']} [{c['num_tokens']}t] start={c['start_idx']} | {c['content']}"
        )


def demo_model():
    print_section("2. Model-Based")
    text = "Hello world this is a test document with enough words to slide."
    for c in tc.chunk_texts_sliding_window(
        text, chunk_size=5, step_size=2, model="llama-3.2:3b"
    ):
        print(
            f"  #{c['chunk_index']} id={c['id'][:8]}... {c['num_tokens']}t | {c['content'][:50]}"
        )


def demo_small_step():
    print_section("3. Small Step (Heavy Overlap)")
    text = "One two three four five six seven eight nine ten."
    chunks = tc.chunk_texts_sliding_window(text, chunk_size=4, step_size=1, model=None)
    print(f"  {len(chunks)} chunks (step=1)")
    for c in chunks:
        print(f"  #{c['chunk_index']}: {c['content']}")


def demo_merge():
    print_section("4. Trailing Min Chunk Merge")
    text = "A " + "word " * 15 + "tiny end."
    for c in tc.chunk_texts_sliding_window(
        text, chunk_size=6, step_size=4, min_chunk_size=3, model=None
    ):
        print(f"  #{c['chunk_index']} {c['num_tokens']}t | {c['content']}")


def demo_custom_ids():
    print_section("5. Custom Doc IDs")
    texts = ["Doc Alpha content.", "Doc Beta content."]
    for c in tc.chunk_texts_sliding_window(
        texts, chunk_size=4, step_size=2, model=None, ids=["a-1", "b-2"]
    ):
        print(f"  doc={c['doc_id']} #{c['chunk_index']} | {c['content']}")


def demo_fast():
    print_section("6. Fast Variant")
    text = "Fast version test with words to slide through quickly."
    for c in tc.chunk_texts_sliding_window_fast(
        text, chunk_size=4, step_size=2, model=None
    ):
        print(
            f"  #{c['chunk_index']} start={c['start_idx']} end={c['end_idx']} | {c['content']}"
        )


def demo_error():
    print_section("7. Error Handling")
    try:
        tc.chunk_texts_sliding_window("test", chunk_size=5, step_size=10, model=None)
    except ValueError as e:
        print(f"  ✓ Caught: {e}")


if __name__ == "__main__":
    for fn in [
        demo_basic,
        demo_model,
        demo_small_step,
        demo_merge,
        demo_custom_ids,
        demo_fast,
        demo_error,
    ]:
        fn()
    print_section("Done — sliding window")
