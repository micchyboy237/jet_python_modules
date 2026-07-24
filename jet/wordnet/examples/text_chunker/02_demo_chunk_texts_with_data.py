"""
Demo: chunk_texts_with_data
- ChunkResult dicts (id, doc_id, chunk_index, num_tokens, overlap indices)
- Custom document IDs, model-based with overlap, min chunk merging
"""

from jet.wordnet.examples.text_chunker.demo_utils import apply_mocks, print_section

tc = apply_mocks()


def demo_basic():
    print_section("1. ChunkResult Output")
    text = "First sentence. Second sentence. Third. Fourth."
    chunks = tc.chunk_texts_with_data(text, chunk_size=4, model=None)
    print(f"  {len(chunks)} chunks")
    for c in chunks[:3]:
        print(
            f"  #{c['chunk_index']} id={c['id'][:8]}... tokens={c['num_tokens']} "
            f"doc={c['doc_id'][:8]}... | {c['content'][:50]}"
        )


def demo_custom_ids():
    print_section("2. Custom Doc IDs")
    texts = ["Doc A text one. Doc A text two.", "Doc B text one. Doc B text two."]
    chunks = tc.chunk_texts_with_data(
        texts, chunk_size=4, model=None, ids=["alpha-001", "beta-002"]
    )
    for c in chunks:
        print(
            f"  doc={c['doc_id']} idx={c['doc_index']} #{c['chunk_index']} | {c['content'][:50]}"
        )


def demo_model_overlap():
    print_section("3. Model + Overlap Indices")
    text = "Hello world this is a test. " * 3
    chunks = tc.chunk_texts_with_data(
        text, chunk_size=6, chunk_overlap=2, model="llama-3.2:3b"
    )
    for c in chunks:
        ov = (
            f"overlap=[{c['overlap_start_idx']}:{c['overlap_end_idx']}]"
            if c["overlap_start_idx"] is not None
            else ""
        )
        print(
            f"  #{c['chunk_index']} tokens={c['num_tokens']} start={c['start_idx']} end={c['end_idx']} {ov}"
        )
        print(f"    {c['content'][:70]}")


def demo_min_merge():
    print_section("4. Min Chunk Merging")
    text = "A " + "word " * 20 + "final tiny."
    chunks = tc.chunk_texts_with_data(text, chunk_size=8, min_chunk_size=4, model=None)
    for c in chunks:
        print(f"  #{c['chunk_index']} {c['num_tokens']} words | {c['content'][:60]}")


def demo_strict():
    print_section("5. Strict Sentences + Metadata")
    text = "Short. Another. " + "word " * 30 + "End."
    for c in tc.chunk_texts_with_data(
        text, chunk_size=10, strict_sentences=True, model=None
    ):
        print(
            f"  #{c['chunk_index']} {c['num_tokens']}w doc={c['doc_index']} | {c['content'][:80]}"
        )


def demo_progress():
    print_section("6. Progress Bar")
    texts = [f"Doc {i} text. Another." for i in range(5)]
    print("  (Progress bar below)")
    chunks = tc.chunk_texts_with_data(
        texts, chunk_size=6, model=None, show_progress=True
    )
    print(f"  Total: {len(chunks)}")


if __name__ == "__main__":
    for fn in [
        demo_basic,
        demo_custom_ids,
        demo_model_overlap,
        demo_min_merge,
        demo_strict,
        demo_progress,
    ]:
        fn()
    print_section("Done — chunk_texts_with_data")
