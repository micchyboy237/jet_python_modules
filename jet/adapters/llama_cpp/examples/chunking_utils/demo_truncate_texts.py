# demo_truncate_texts.py
"""Demo showcasing truncate_texts from chunking_utils with both single string and list inputs."""

from jet.adapters.llama_cpp.chunking_utils import truncate_texts
from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.logger import logger

# Sample texts for demonstration
SHORT_TEXT = "This is a short sentence. It should not be truncated."

LONG_TEXT = (
    "Artificial intelligence has revolutionized the way we process information. "
    "Machine learning models can now understand natural language with remarkable accuracy. "
    "These systems are trained on vast amounts of text data. "
    "They learn patterns and relationships between words and concepts. "
    "The applications range from chatbots to content generation. "
    "However, these models have context window limitations. "
    "This means they can only process a certain number of tokens at once. "
    "Truncation becomes necessary when dealing with lengthy documents. "
    "Smart truncation preserves sentence boundaries for better coherence. "
    "This is especially important for downstream tasks like summarization. "
    "The quality of truncation directly impacts the quality of results. "
    "Proper token counting ensures we maximize the use of available context. "
    "Each model has its own tokenizer with specific vocabulary. "
    "Understanding tokenization is key to effective text processing."
)

MULTI_TEXT = (
    "Climate change is one of the most pressing issues of our time. "
    "Rising global temperatures are causing unprecedented weather patterns. "
    "Scientists have been warning about these changes for decades."
)

BATCH_TEXTS = [
    LONG_TEXT,
    SHORT_TEXT,
    MULTI_TEXT,
    "",  # Empty string edge case
    "Single sentence without period",  # Edge case
]


def demo_single_string():
    """Demonstrate truncate_texts with a single string input."""
    print("\n" + "=" * 80)
    print("DEMO 1: Single String Input")
    print("=" * 80)

    print(f"\nModel: {LLM_MODEL}")
    print(f"\nOriginal text length: {len(LONG_TEXT)} chars")

    # Truncate with small token limit to see the effect
    result = truncate_texts(
        texts=LONG_TEXT,
        model=LLM_MODEL,
        max_tokens=30,  # Small limit to demonstrate truncation
        strict_sentences=True,
        show_progress=False,  # No progress bar for single input
    )

    print(f"\nType of result: {type(result).__name__}")
    print(f"Is string: {isinstance(result, str)}")
    print(f"Truncated text length: {len(result)} chars")
    print(f"\nTruncated text:")
    print("-" * 40)
    print(result)
    print("-" * 40)

    # Verify it's a string
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) < len(LONG_TEXT), "Should be truncated"
    logger.info("✅ Single string demo passed type and truncation checks")


def demo_string_list():
    """Demonstrate truncate_texts with a list of strings."""
    print("\n" + "=" * 80)
    print("DEMO 2: List of Strings Input")
    print("=" * 80)

    print(f"\nModel: {LLM_MODEL}")
    print(f"Number of input texts: {len(BATCH_TEXTS)}")

    # Show input texts summary
    print("\nInput texts summary:")
    for i, text in enumerate(BATCH_TEXTS):
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"  [{i}] len={len(text)}: {preview}")

    # Truncate with moderate token limit
    results = truncate_texts(
        texts=BATCH_TEXTS,
        model=LLM_MODEL,
        max_tokens=40,  # Moderate limit
        strict_sentences=True,
        show_progress=True,  # Show progress bar for batch
    )

    print(f"\nType of result: {type(results).__name__}")
    print(f"Is list: {isinstance(results, list)}")
    print(f"Number of results: {len(results)}")

    # Show results
    print("\nTruncated results:")
    for i, result in enumerate(results):
        preview = result[:100] + "..." if len(result) > 100 else result
        print(f"  [{i}] len={len(result)}: {preview}")

    # Verify it's a list and empty strings are filtered
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) < len(BATCH_TEXTS), "Should filter empty/invalid texts"
    assert all(r for r in results), "Should not contain empty strings"
    logger.info("✅ List demo passed type and filtering checks")


def demo_strict_vs_non_strict():
    """Compare strict_sentences=True vs strict_sentences=False."""
    print("\n" + "=" * 80)
    print("DEMO 3: Strict vs Non-Strict Sentence Mode")
    print("=" * 80)

    # Make a longer text that WILL be truncated to show the difference
    text = (
        "First complete sentence with important context. "
        "Second sentence that should ideally be kept whole. "
        "Third sentence with critical information inside. "
        "Fourth sentence that demonstrates the difference between modes. "
        "Fifth sentence that will definitely be cut off in one mode."
    )

    print(f"\nModel: {LLM_MODEL}")
    print(f"Original text: {text}")

    # Strict mode (preserves sentence boundaries)
    strict_result = truncate_texts(
        texts=text,
        model=LLM_MODEL,
        max_tokens=30,  # Small enough to force truncation
        strict_sentences=True,
        show_progress=False,
    )

    # Non-strict mode (token-level truncation)
    non_strict_result = truncate_texts(
        texts=text,
        model=LLM_MODEL,
        max_tokens=30,
        strict_sentences=False,
        show_progress=False,
    )

    print(f"\nStrict (preserves sentences):")
    print(f"  Length: {len(strict_result)} chars")
    print(f"  Ends with period: {strict_result.rstrip().endswith('.')}")
    print(f"  Text: {strict_result}")

    print(f"\nNon-strict (token-level):")
    print(f"  Length: {len(non_strict_result)} chars")
    print(f"  Ends with period: {non_strict_result.rstrip().endswith('.')}")
    print(f"  Text: {non_strict_result}")

    # Check differences
    if strict_result != non_strict_result:
        print(f"\n✅ Results differ - truncation mode matters!")
        # Strict mode should preserve complete sentences
        assert strict_result.rstrip().endswith("."), (
            "Strict mode should end with complete sentence"
        )
    else:
        print(
            f"\n⚠️  Results are identical - text might be too short for max_tokens={30}"
        )

    logger.info("✅ Strict vs non-strict demo passed")


def demo_edge_cases():
    """Demonstrate edge case handling."""
    print("\n" + "=" * 80)
    print("DEMO 4: Edge Cases")
    print("=" * 80)

    print(f"\nModel: {LLM_MODEL}")

    # Case 1: Text shorter than max_tokens (should remain unchanged)
    print("\n1. Short text (no truncation needed):")
    result = truncate_texts(
        texts=SHORT_TEXT,
        model=LLM_MODEL,
        max_tokens=1000,
        strict_sentences=True,
    )
    print(f"   Input: {SHORT_TEXT}")
    print(f"   Output: {result}")
    assert result == SHORT_TEXT, "Short text should remain unchanged"

    # Case 2: Empty string
    print("\n2. Empty string:")
    result = truncate_texts(
        texts="",
        model=LLM_MODEL,
        max_tokens=100,
        strict_sentences=True,
    )
    print(f"   Output: '{result}' (type: {type(result).__name__})")
    assert isinstance(result, str), "Should return string for string input"

    # Case 3: Text with no sentence boundaries
    print("\n3. Text without sentence boundaries:")
    no_period_text = "This is a very long text without any sentence boundaries that just keeps going and going without proper punctuation to see how the system handles such cases"
    result = truncate_texts(
        texts=no_period_text,
        model=LLM_MODEL,
        max_tokens=10,
        strict_sentences=True,
    )
    print(f"   Input length: {len(no_period_text)} chars")
    print(f"   Output length: {len(result)} chars")
    print(f"   Output: {result}")

    # Case 4: Very small max_tokens (should still return something)
    print("\n4. Very small max_tokens:")
    result = truncate_texts(
        texts=LONG_TEXT,
        model=LLM_MODEL,
        max_tokens=5,
        strict_sentences=True,
    )
    print(f"   Output: {result}")
    assert len(result) > 0, "Should return at least some text"

    logger.info("✅ Edge cases demo passed")


def main():
    """Run all demonstrations."""
    print("=" * 80)
    print("TRUNCATE_TEXTS DEMONSTRATION")
    print("=" * 80)
    print(f"\nUsing model: {LLM_MODEL}")

    demo_single_string()
    demo_string_list()
    demo_strict_vs_non_strict()
    demo_edge_cases()

    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETED SUCCESSFULLY ✅")
    print("=" * 80)


if __name__ == "__main__":
    main()
