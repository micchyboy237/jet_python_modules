# demo_chunk_texts_with_data.py
"""Demo showcasing chunk_texts_with_data from chunking_utils with metadata-rich results.

Note: In strict_sentences mode, overlap metadata (overlap_start_idx/overlap_end_idx)
is always None because overlap is handled at the sentence level, not token indices.
Token-level overlap tracking only applies to non-strict mode.
"""

from collections import defaultdict

from jet.adapters.llama_cpp.chunking_utils import (
    _tokenize_for_size,
    chunk_texts_with_data,
)
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.logger import logger

# Sample texts for demonstration
SHORT_TEXT = "This is a short document. It should be a single chunk."

MEDIUM_TEXT = (
    "Artificial intelligence has transformed modern technology. "
    "Machine learning algorithms process vast amounts of data. "
    "These systems can recognize patterns and make predictions. "
    "The applications range from healthcare to finance. "
    "Deep learning has achieved remarkable results in image recognition."
)

LONG_TEXT = (
    "Renewable energy is crucial for combating climate change. "
    "Solar panels convert sunlight directly into electricity. "
    "Wind turbines harness the power of moving air. "
    "Hydroelectric dams use flowing water to generate power. "
    "Geothermal energy taps into the Earth's internal heat. "
    "Biomass energy comes from organic materials. "
    "These technologies are becoming more efficient each year. "
    "The cost of renewable energy has dropped significantly. "
    "Many countries are investing heavily in green infrastructure."
)

MULTI_DOC_TEXTS = [
    "First document about AI. It has multiple sentences. This is for testing metadata.",
    "Second document about technology. AI is changing the world. Machine learning is powerful and transformative.",
    "",  # Empty document - should be filtered out
    "Third document. Single sentence here.",
]


def demo_basic_with_metadata():
    """Demonstrate basic chunking with rich metadata."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Chunking with Metadata")
    print("=" * 80)

    print(f"\nModel: {LLM_MODEL}")
    print(f"Original text ({len(LONG_TEXT)} chars, multiple sentences):")
    print("-" * 40)
    print(LONG_TEXT)
    print("-" * 40)

    # Chunk with metadata - use small chunk_size to ensure multiple chunks
    chunks = chunk_texts_with_data(
        texts=LONG_TEXT,
        chunk_size=30,
        chunk_overlap=5,
        model=LLM_MODEL,
        strict_sentences=True,
        min_chunk_size=10,
        show_progress=False,
    )

    print(f"\nGenerated {len(chunks)} chunks with metadata:")

    # Show all chunks if few, otherwise first 3 and last
    if len(chunks) <= 5:
        display_chunks = chunks
    else:
        display_chunks = chunks[:3] + [chunks[-1]]

    for chunk in display_chunks:
        print(f"\n  ┌─ Chunk {chunk['chunk_index']}")
        print(f"  │  ID: {chunk['id'][:8]}...")
        print(f"  │  Doc ID: {chunk['doc_id'][:8]}...")
        print(f"  │  Doc Index: {chunk['doc_index']}")
        print(f"  │  Tokens: {chunk['num_tokens']}")
        print(f"  │  Content: {chunk['content'][:80]}...")
        print(f"  │  Range: [{chunk['start_idx']}:{chunk['end_idx']}]")
        print(
            f"  │  Overlap: start={chunk['overlap_start_idx']}, "
            f"end={chunk['overlap_end_idx']}"
        )
        print(f"  └─")

    if len(chunks) > len(display_chunks):
        print(f"\n  ... and {len(chunks) - len(display_chunks)} more chunks")

    # Verify we actually got multiple chunks
    assert len(chunks) > 1, (
        f"Expected multiple chunks with chunk_size=30, got {len(chunks)}. "
        f"Text may be shorter than expected."
    )

    # Verify metadata integrity
    for chunk in chunks:
        assert chunk["id"], "Should have unique ID"
        assert chunk["doc_id"], "Should have document ID"
        assert chunk["num_tokens"] > 0, (
            f"Should have positive token count, got {chunk['num_tokens']}"
        )
        assert chunk["content"], "Should have content"
        assert chunk["start_idx"] >= 0, (
            f"Should have valid start index, got {chunk['start_idx']}"
        )
        assert chunk["end_idx"] > chunk["start_idx"], (
            f"End ({chunk['end_idx']}) should be after start ({chunk['start_idx']})"
        )
        # In strict mode, overlap indices are None (sentence-level overlap)
        # This is expected behavior
        if chunk["chunk_index"] > 0:
            logger.debug(
                f"Chunk {chunk['chunk_index']} overlap_indices=None "
                f"(expected in strict mode - overlap handled at sentence level)"
            )

    # Check chunk ordering
    for i in range(1, len(chunks)):
        assert chunks[i]["chunk_index"] > chunks[i - 1]["chunk_index"], (
            f"Chunks should be in order: {chunks[i]['chunk_index']} "
            f"after {chunks[i - 1]['chunk_index']}"
        )

    # Verify unique IDs
    chunk_ids = [c["id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "All chunk IDs should be unique"

    logger.info(f"✅ Basic metadata demo passed ({len(chunks)} chunks)")


def demo_multi_document():
    """Demonstrate chunking multiple documents with document IDs."""
    print("\n" + "=" * 80)
    print("DEMO 2: Multi-Document with Custom IDs")
    print("=" * 80)

    custom_ids = ["doc_001", "doc_002", "doc_003", "doc_004"]

    print(f"\nModel: {LLM_MODEL}")
    print(f"Processing {len(MULTI_DOC_TEXTS)} documents with custom IDs:")
    for i, (doc_id, text) in enumerate(zip(custom_ids, MULTI_DOC_TEXTS)):
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"  [{i}] {doc_id}: '{preview}'")

    # Use small min_chunk_size to keep chunks from shorter documents
    chunks = chunk_texts_with_data(
        texts=MULTI_DOC_TEXTS,
        chunk_size=30,
        chunk_overlap=5,
        model=LLM_MODEL,
        ids=custom_ids,
        strict_sentences=True,
        min_chunk_size=5,  # Lower to keep smaller documents
        show_progress=True,
    )

    print(f"\nGenerated {len(chunks)} chunks across {len(MULTI_DOC_TEXTS)} documents:")

    # Group chunks by document
    doc_chunks = defaultdict(list)
    for chunk in chunks:
        doc_chunks[chunk["doc_id"]].append(chunk)

    for doc_id in custom_ids:
        if doc_id in doc_chunks:
            doc_chunk_list = doc_chunks[doc_id]
            doc_index = doc_chunk_list[0]["doc_index"]
            print(f"\n  Document '{doc_id}' (index {doc_index}):")
            for chunk in doc_chunk_list:
                print(
                    f"    Chunk {chunk['chunk_index']}: "
                    f"{chunk['num_tokens']} tokens - "
                    f'"{chunk["content"][:60]}..."'
                )
        else:
            print(f"\n  Document '{doc_id}': [empty - no chunks generated]")

    # Verify document grouping
    assert len(doc_chunks) > 0, "Should have at least one document with chunks"

    # Empty document (doc_003 at index 2) should not produce chunks
    empty_doc_id = custom_ids[2]
    assert empty_doc_id not in doc_chunks, (
        f"Empty document '{empty_doc_id}' should not produce chunks"
    )

    # Each chunk should reference the correct document
    for chunk in chunks:
        assert chunk["doc_id"] in custom_ids, (
            f"Chunk references unknown doc_id: {chunk['doc_id']}"
        )
        doc_idx = custom_ids.index(chunk["doc_id"])
        assert chunk["doc_index"] == doc_idx, (
            f"doc_index mismatch: {chunk['doc_index']} vs {doc_idx}"
        )

    logger.info(f"✅ Multi-document demo passed ({len(doc_chunks)} non-empty docs)")


def demo_overlap_non_strict():
    """Demonstrate overlap metadata in non-strict mode (where it's actually tracked)."""
    print("\n" + "=" * 80)
    print("DEMO 3: Overlap Metadata Tracking (Non-Strict Mode)")
    print("=" * 80)

    text = (
        "Artificial intelligence has transformed modern technology. "
        "Machine learning algorithms process vast amounts of data. "
        "These systems can recognize patterns and make predictions. "
        "The applications range from healthcare to finance. "
        "Deep learning has achieved remarkable results in image recognition. "
        "Natural language processing enables machines to understand text. "
        "Computer vision allows systems to interpret visual information. "
        "Robotics combines AI with physical machines for automation."
    )

    print(f"\nModel: {LLM_MODEL}")
    print(f"Original text ({len(text)} chars, 8 sentences):")
    print("-" * 40)
    print(text)
    print("-" * 40)

    # Non-strict mode with small chunks and significant overlap
    chunks = chunk_texts_with_data(
        texts=text,
        chunk_size=25,
        chunk_overlap=10,
        model=LLM_MODEL,
        strict_sentences=False,  # Non-strict mode tracks overlap indices
        min_chunk_size=5,
        show_progress=False,
    )

    print(f"\nGenerated {len(chunks)} chunks with overlap=10 (non-strict mode):")
    for chunk in chunks:
        overlap_info = ""
        if chunk["overlap_start_idx"] is not None:
            overlap_size = chunk["overlap_end_idx"] - chunk["overlap_start_idx"]
            overlap_info = (
                f" [overlap: {chunk['overlap_start_idx']}-"
                f"{chunk['overlap_end_idx']} ({overlap_size} tokens)]"
            )
        else:
            overlap_info = " [no overlap - last chunk]"

        print(
            f"\n  Chunk {chunk['chunk_index']} "
            f"({chunk['num_tokens']} tokens){overlap_info}:"
        )
        print(f"  Content: {chunk['content'][:100]}...")
        print(f"  Range: [{chunk['start_idx']}:{chunk['end_idx']}]")

    # Verify multiple chunks were created
    assert len(chunks) > 1, (
        f"Should create multiple chunks with chunk_size=25, got {len(chunks)}"
    )

    # In non-strict mode, all chunks except possibly the last should have overlap
    overlap_chunks = [c for c in chunks if c["overlap_start_idx"] is not None]
    non_overlap_chunks = [c for c in chunks if c["overlap_start_idx"] is None]

    print(f"\nOverlap statistics:")
    print(f"  Chunks with overlap: {len(overlap_chunks)}")
    print(f"  Chunks without overlap: {len(non_overlap_chunks)}")

    # At least some chunks should have overlap metadata
    if len(chunks) > 1:
        assert len(overlap_chunks) > 0, (
            "With overlap=10, non-last chunks should have overlap metadata"
        )

    # Validate overlap ranges
    for chunk in chunks:
        if chunk["overlap_start_idx"] is not None:
            assert chunk["overlap_start_idx"] >= chunk["start_idx"], (
                f"Overlap start ({chunk['overlap_start_idx']}) should be >= "
                f"chunk start ({chunk['start_idx']})"
            )
            assert chunk["overlap_end_idx"] <= chunk["end_idx"], (
                f"Overlap end ({chunk['overlap_end_idx']}) should be <= "
                f"chunk end ({chunk['end_idx']})"
            )
            assert chunk["overlap_start_idx"] < chunk["overlap_end_idx"], (
                f"Overlap range should be positive: "
                f"{chunk['overlap_start_idx']}-{chunk['overlap_end_idx']}"
            )

    logger.info("✅ Overlap metadata demo passed")


def demo_strict_vs_non_strict():
    """Demonstrate metadata differences between strict and non-strict modes."""
    print("\n" + "=" * 80)
    print("DEMO 4: Strict vs Non-Strict Metadata Comparison")
    print("=" * 80)

    text = LONG_TEXT
    print(f"\nModel: {LLM_MODEL}")
    print(f"Original text ({len(text)} chars, 9 sentences):")
    print("-" * 40)
    print(text[:200] + "...")
    print("-" * 40)

    chunk_size = 30
    overlap = 5

    # Strict mode
    strict_chunks = chunk_texts_with_data(
        texts=text,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        model=LLM_MODEL,
        strict_sentences=True,
        min_chunk_size=10,
        show_progress=False,
    )

    # Non-strict mode
    non_strict_chunks = chunk_texts_with_data(
        texts=text,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        model=LLM_MODEL,
        strict_sentences=False,
        min_chunk_size=10,
        show_progress=False,
    )

    print(f"\n{'=' * 60}")
    print(f"Strict Mode (sentence boundaries preserved):")
    print(f"{'=' * 60}")
    print(f"  Chunks: {len(strict_chunks)}")
    for chunk in strict_chunks:
        ends_with_punct = any(chunk["content"].rstrip().endswith(p) for p in ".!?")
        has_overlap = chunk["overlap_start_idx"] is not None
        print(
            f"  Chunk {chunk['chunk_index']}: "
            f"{chunk['num_tokens']} tokens, "
            f"ends_with_punct={ends_with_punct}, "
            f"has_overlap_indices={has_overlap}"
        )
        if not ends_with_punct:
            print(f"    ⚠️  Content: ...{chunk['content'][-50:]}")

    print(f"\n{'=' * 60}")
    print(f"Non-Strict Mode (token-level truncation):")
    print(f"{'=' * 60}")
    print(f"  Chunks: {len(non_strict_chunks)}")
    for chunk in non_strict_chunks:
        ends_with_punct = any(chunk["content"].rstrip().endswith(p) for p in ".!?")
        has_overlap = chunk["overlap_start_idx"] is not None
        print(
            f"  Chunk {chunk['chunk_index']}: "
            f"{chunk['num_tokens']} tokens, "
            f"ends_with_punct={ends_with_punct}, "
            f"has_overlap_indices={has_overlap}"
        )

    # Verify strict mode properties
    for chunk in strict_chunks:
        # Strict chunks should end with punctuation
        ends_with_punct = any(chunk["content"].rstrip().endswith(p) for p in ".!?")
        assert ends_with_punct, (
            f"Strict chunk should end with punctuation: '{chunk['content'][-50:]}'"
        )
        # Overlap indices should be None in strict mode
        assert chunk["overlap_start_idx"] is None, (
            "Strict mode should not have overlap indices (handled at sentence level)"
        )

    # Verify non-strict mode properties
    for chunk in non_strict_chunks:
        assert chunk["num_tokens"] > 0, "Should have tokens"
        assert chunk["start_idx"] >= 0, "Should have valid start"
        assert chunk["content"], "Should have content"

    # Modes should produce different results
    if len(strict_chunks) != len(non_strict_chunks):
        print(
            f"\n✅ Modes produce different chunk counts "
            f"({len(strict_chunks)} vs {len(non_strict_chunks)})"
        )
    else:
        # Same count but check content
        same_content = all(
            s["content"] == n["content"]
            for s, n in zip(strict_chunks, non_strict_chunks)
        )
        if not same_content:
            print(f"\n✅ Modes produce different content (same chunk count)")
        else:
            print(
                f"\n⚠️  Modes produced identical results (text may fit in single chunk)"
            )

    logger.info("✅ Strict vs non-strict comparison passed")


def demo_token_counting():
    """Demonstrate accurate token counting in metadata."""
    print("\n" + "=" * 80)
    print("DEMO 5: Token Count Verification")
    print("=" * 80)

    text = "Short sentence about AI. Another sentence about machine learning. Final sentence here."

    print(f"\nModel: {LLM_MODEL}")
    print(f"Original text: {text}")

    chunks = chunk_texts_with_data(
        texts=text,
        chunk_size=100,  # Large enough to fit everything
        chunk_overlap=0,
        model=LLM_MODEL,
        strict_sentences=True,
        show_progress=False,
    )

    print(f"\nChunks generated: {len(chunks)}")
    all_match = True
    for chunk in chunks:
        # Manually verify token count
        manual_tokens = len(_tokenize_for_size(chunk["content"], model=LLM_MODEL))
        match = chunk["num_tokens"] == manual_tokens

        print(f"\n  Chunk {chunk['chunk_index']}:")
        print(f"    Content: {chunk['content']}")
        print(f"    Reported tokens: {chunk['num_tokens']}")
        print(f"    Manual count: {manual_tokens}")
        print(f"    Match: {match}")

        if not match:
            all_match = False
            logger.warning(
                f"Token count mismatch: {chunk['num_tokens']} vs {manual_tokens}"
            )

        assert match, (
            f"Token count mismatch for chunk {chunk['chunk_index']}: "
            f"{chunk['num_tokens']} vs {manual_tokens}"
        )

    if all_match:
        print(f"\n✅ All token counts verified correctly")
    logger.info("✅ Token counting demo passed")


def demo_chunk_ordering():
    """Demonstrate chunk ordering and indices."""
    print("\n" + "=" * 80)
    print("DEMO 6: Chunk Ordering and Index Validation")
    print("=" * 80)

    print(f"\nModel: {LLM_MODEL}")

    chunks = chunk_texts_with_data(
        texts=LONG_TEXT,
        chunk_size=30,
        chunk_overlap=5,
        model=LLM_MODEL,
        strict_sentences=True,
        min_chunk_size=10,
        show_progress=False,
    )

    print(f"Total chunks: {len(chunks)}")

    # Validate ordering
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i}:")
        print(f"    Chunk Index: {chunk['chunk_index']}")
        print(f"    Content start: {chunk['start_idx']}")
        print(f"    Content end: {chunk['end_idx']}")
        print(f"    Tokens: {chunk['num_tokens']}")
        print(f"    Content preview: {chunk['content'][:60]}...")

        # Check chunk_index is sequential
        assert chunk["chunk_index"] == i, (
            f"Chunk index should be sequential: expected {i}, "
            f"got {chunk['chunk_index']}"
        )

        # Verify content is not empty
        assert chunk["content"].strip(), "Chunk content should not be empty"

    # Verify chunk progression
    if len(chunks) > 1:
        # Check that chunks don't have identical content
        for i in range(len(chunks) - 1):
            assert chunks[i]["content"] != chunks[i + 1]["content"], (
                f"Consecutive chunks {i} and {i + 1} should have different content"
            )

        # Verify total tokens across chunks
        total_chunk_tokens = sum(c["num_tokens"] for c in chunks)
        original_tokens = len(_tokenize_for_size(LONG_TEXT, model=LLM_MODEL))
        print(f"\n  Total chunk tokens: {total_chunk_tokens}")
        print(f"  Original text tokens: {original_tokens}")
        print(f"  Coverage ratio: {total_chunk_tokens / original_tokens:.1%}")

    logger.info("✅ Chunk ordering demo passed")


def main():
    """Run all demonstrations."""
    print("=" * 80)
    print("CHUNK_TEXTS_WITH_DATA DEMONSTRATION")
    print("=" * 80)
    print(f"\nUsing model: {LLM_MODEL}")
    print("\nNote: In strict_sentences mode, overlap_start_idx/overlap_end_idx")
    print("are always None because overlap is handled at the sentence level,")
    print("not token indices. Token-level overlap tracking is only in non-strict mode.")

    demo_basic_with_metadata()
    demo_multi_document()
    demo_overlap_non_strict()
    demo_strict_vs_non_strict()
    demo_token_counting()
    demo_chunk_ordering()

    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETED SUCCESSFULLY ✅")
    print("=" * 80)


if __name__ == "__main__":
    main()
