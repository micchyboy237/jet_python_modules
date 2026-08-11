# demo_chunk_texts.py
"""Demo showcasing chunk_texts from chunking_utils with various configurations."""

from jet.adapters.llama_cpp.chunking_utils import chunk_texts
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.logger import logger

# Sample texts for demonstration
SHORT_TEXT = "This is a short sentence. It should remain as one chunk."

MEDIUM_TEXT = (
    "Artificial intelligence has revolutionized technology. "
    "Machine learning models process natural language effectively. "
    "These systems understand context and generate responses. "
    "The applications span across multiple industries. "
    "Healthcare uses AI for diagnosis and treatment planning."
)

LONG_TEXT = (
    "Climate change is one of the most pressing issues of our time. "
    "Rising global temperatures are causing unprecedented weather patterns. "
    "Scientists have been warning about these changes for decades. "
    "The evidence is clear and overwhelming. "
    "We must take immediate action to reduce carbon emissions. "
    "Renewable energy sources offer a sustainable alternative. "
    "Solar and wind power have become increasingly affordable. "
    "Many countries are transitioning to green energy. "
    "This shift creates new jobs and economic opportunities. "
    "The technology continues to improve rapidly."
)

BATCH_TEXTS = [
    SHORT_TEXT,
    MEDIUM_TEXT,
    LONG_TEXT,
    "",  # Empty string edge case
    "Single sentence without period",  # Edge case
]


def demo_basic_chunking():
    """Demonstrate basic chunking with default parameters."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Chunking (Strict Sentences)")
    print("=" * 80)

    print(f"\nModel: {LLM_MODEL}")
    print(f"Original text ({len(LONG_TEXT)} chars, 10 sentences):")
    print("-" * 40)
    print(LONG_TEXT)
    print("-" * 40)

    # Chunk with small token limit to see multiple chunks
    chunks = chunk_texts(
        texts=LONG_TEXT,
        chunk_size=30,  # Small chunks to demonstrate
        chunk_overlap=5,
        model=LLM_MODEL,
        strict_sentences=True,
        min_chunk_size=10,
        show_progress=False,
    )

    print(f"\nGenerated {len(chunks)} chunks (chunk_size=30, overlap=5):")
    for i, chunk in enumerate(chunks):
        print(f"\n  Chunk {i + 1} ({len(chunk)} chars):")
        print(f"  {'─' * 40}")
        print(f"  {chunk}")

    assert len(chunks) > 1, "Should create multiple chunks"
    logger.info("✅ Basic chunking demo passed")


def demo_non_strict_chunking():
    """Demonstrate chunking without respecting sentence boundaries."""
    print("\n" + "=" * 80)
    print("DEMO 2: Non-Strict Chunking (Token-Level)")
    print("=" * 80)

    print(f"\nModel: {LLM_MODEL}")
    print(f"Original text ({len(LONG_TEXT)} chars):")
    print("-" * 40)
    print(LONG_TEXT[:200] + "...")
    print("-" * 40)

    # Compare strict vs non-strict
    strict_chunks = chunk_texts(
        texts=LONG_TEXT,
        chunk_size=40,
        chunk_overlap=10,
        model=LLM_MODEL,
        strict_sentences=True,
        show_progress=False,
    )

    non_strict_chunks = chunk_texts(
        texts=LONG_TEXT,
        chunk_size=40,
        chunk_overlap=10,
        model=LLM_MODEL,
        strict_sentences=False,
        show_progress=False,
    )

    print(f"\nStrict mode: {len(strict_chunks)} chunks")
    print(f"  First chunk: {strict_chunks[0][:100]}...")
    print(f"  Last chunk: {strict_chunks[-1][:100]}...")

    print(f"\nNon-strict mode: {len(non_strict_chunks)} chunks")
    print(f"  First chunk: {non_strict_chunks[0][:100]}...")
    print(f"  Last chunk: {non_strict_chunks[-1][:100]}...")

    # Strict mode preserves sentence boundaries (ends with punctuation)
    for chunk in strict_chunks:
        assert any(chunk.rstrip().endswith(p) for p in ".!?"), (
            f"Strict chunk should end with punctuation: {chunk[-50:]}"
        )

    logger.info("✅ Non-strict chunking demo passed")


def demo_chunk_overlap():
    """Demonstrate chunk overlap behavior."""
    print("\n" + "=" * 80)
    print("DEMO 3: Chunk Overlap Comparison")
    print("=" * 80)

    text = MEDIUM_TEXT
    print(f"\nModel: {LLM_MODEL}")
    print(f"Original text ({len(text)} chars, 5 sentences):")
    print("-" * 40)
    print(text)
    print("-" * 40)

    # No overlap
    no_overlap = chunk_texts(
        texts=text,
        chunk_size=30,
        chunk_overlap=0,
        model=LLM_MODEL,
        strict_sentences=True,
        show_progress=False,
    )

    # With overlap
    with_overlap = chunk_texts(
        texts=text,
        chunk_size=30,
        chunk_overlap=15,
        model=LLM_MODEL,
        strict_sentences=True,
        show_progress=False,
    )

    print(f"\nNo overlap ({len(no_overlap)} chunks):")
    for i, chunk in enumerate(no_overlap):
        print(f"  Chunk {i + 1}: {chunk[:80]}...")

    print(f"\nWith overlap=15 ({len(with_overlap)} chunks):")
    for i, chunk in enumerate(with_overlap):
        print(f"  Chunk {i + 1}: {chunk[:80]}...")

    # Overlap should create more chunks (or at least not fewer)
    assert len(with_overlap) >= len(no_overlap), "Overlap should not reduce chunk count"
    logger.info("✅ Chunk overlap demo passed")


def demo_batch_processing():
    """Demonstrate batch processing with progress bar."""
    print("\n" + "=" * 80)
    print("DEMO 4: Batch Processing")
    print("=" * 80)

    print(f"\nModel: {LLM_MODEL}")
    print(f"Processing {len(BATCH_TEXTS)} texts...")

    chunks = chunk_texts(
        texts=BATCH_TEXTS,
        chunk_size=50,
        chunk_overlap=10,
        model=LLM_MODEL,
        strict_sentences=True,
        show_progress=True,  # Show progress bar for batch
    )

    print(f"\nTotal chunks generated: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
        print(f"  Chunk {i + 1} ({len(chunk)} chars): {preview}")

    # Should process all texts (empty text filtered out)
    assert len(chunks) > 0, "Should generate at least some chunks"
    logger.info("✅ Batch processing demo passed")


def demo_min_chunk_size():
    """Demonstrate min_chunk_size filtering."""
    print("\n" + "=" * 80)
    print("DEMO 5: Minimum Chunk Size Filtering")
    print("=" * 80)

    text = "Short. This is a slightly longer sentence that should be kept. Another one. Small."

    print(f"\nModel: {LLM_MODEL}")
    print(f"Original text: {text}")

    # High min_chunk_size (filters out small chunks)
    high_min = chunk_texts(
        texts=text,
        chunk_size=20,
        chunk_overlap=0,
        model=LLM_MODEL,
        strict_sentences=True,
        min_chunk_size=15,  # High threshold
        show_progress=False,
    )

    # Low min_chunk_size (keeps more chunks)
    low_min = chunk_texts(
        texts=text,
        chunk_size=20,
        chunk_overlap=0,
        model=LLM_MODEL,
        strict_sentences=True,
        min_chunk_size=5,  # Low threshold
        show_progress=False,
    )

    print(f"\nHigh min_chunk_size=15: {len(high_min)} chunks")
    for chunk in high_min:
        print(f"  - {chunk}")

    print(f"\nLow min_chunk_size=5: {len(low_min)} chunks")
    for chunk in low_min:
        print(f"  - {chunk}")

    # Higher threshold should produce fewer chunks
    assert len(high_min) <= len(low_min), (
        "Higher min_chunk_size should filter more chunks"
    )
    logger.info("✅ Min chunk size demo passed")


def demo_single_vs_list_input():
    """Demonstrate single string vs list input."""
    print("\n" + "=" * 80)
    print("DEMO 6: Single String vs List Input")
    print("=" * 80)

    print(f"\nModel: {LLM_MODEL}")

    # Single string input
    single_result = chunk_texts(
        texts=MEDIUM_TEXT,
        chunk_size=30,
        chunk_overlap=5,
        model=LLM_MODEL,
        show_progress=False,
    )

    # List input (single item)
    list_result = chunk_texts(
        texts=[MEDIUM_TEXT],
        chunk_size=30,
        chunk_overlap=5,
        model=LLM_MODEL,
        show_progress=False,
    )

    print(f"Single string input → {len(single_result)} chunks")
    print(f"List with 1 item → {len(list_result)} chunks")
    print(f"Results identical: {single_result == list_result}")

    assert single_result == list_result, (
        "Single and list input should produce same chunks"
    )
    logger.info("✅ Single vs list input demo passed")


def main():
    """Run all demonstrations."""
    print("=" * 80)
    print("CHUNK_TEXTS DEMONSTRATION")
    print("=" * 80)
    print(f"\nUsing model: {LLM_MODEL}")

    demo_basic_chunking()
    demo_non_strict_chunking()
    demo_chunk_overlap()
    demo_batch_processing()
    demo_min_chunk_size()
    demo_single_vs_list_input()

    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETED SUCCESSFULLY ✅")
    print("=" * 80)


if __name__ == "__main__":
    main()
