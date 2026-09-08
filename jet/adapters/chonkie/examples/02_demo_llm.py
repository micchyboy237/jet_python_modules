"""Demo: LlamacppGenie with text generation and structured JSON output.

Prerequisites:
  - Windows llama.cpp server running with an LLM model loaded
  - LLAMA_CPP_LLM_URL env var set (or default localhost:8080)
  - pip install chonkie pydantic rich
"""

from __future__ import annotations

import asyncio

from jet.adapters.chonkie.llamacpp_genie import LlamacppGenie
from jet.logger import logger
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic schemas for structured output demos
# ---------------------------------------------------------------------------


class TopicSummary(BaseModel):
    """Structured summary of a technical topic."""

    topic: str = Field(description="The main topic being summarized")
    key_points: list[str] = Field(
        description="3-5 bullet-point takeaways", min_length=1
    )
    difficulty: str = Field(description="One of: beginner, intermediate, advanced")
    one_liner: str = Field(description="A single-sentence explanation")


class SentimentResult(BaseModel):
    """Sentiment analysis result."""

    text_snippet: str = Field(description="First 50 chars of analyzed text")
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0, le=1)


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_text_generation(genie: LlamacppGenie) -> None:
    """Plain text generation."""
    logger.info("=== Text Generation ===")
    response = genie.generate(
        "Explain semantic chunking in RAG pipelines in exactly 3 sentences."
    )
    print(f"\n📝 Response ({len(response)} chars):\n{response}\n")


async def demo_async_text_generation(genie: LlamacppGenie) -> None:
    """Async plain text generation."""
    logger.info("=== Async Text Generation ===")
    response = await genie.agenerate(
        "What is the difference between window and cumulative semantic chunking? Answer in 2 sentences."
    )
    print(f"\n⚡ Async Response ({len(response)} chars):\n{response}\n")


def demo_structured_json(genie: LlamacppGenie) -> None:
    """Structured JSON output with Pydantic validation."""
    logger.info("=== Structured JSON Output ===")
    result = genie.generate_json(
        "Summarize the concept of 'embedding models' for a technical audience.",
        schema=TopicSummary,
    )
    print("\n📊 Structured Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print()


def demo_batch_generation(genie: LlamacppGenie) -> None:
    """Batch text generation (inherited from BaseGenie)."""
    logger.info("=== Batch Generation ===")
    prompts = [
        "Define 'token' in NLP in one sentence.",
        "Define 'embedding' in ML in one sentence.",
        "Define 'chunking' in RAG in one sentence.",
    ]
    responses = genie.generate_batch(prompts)
    print("\n📦 Batch Results:")
    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        print(f"  {i}. Q: {prompt}")
        print(f"     A: {response}")
    print()


async def demo_async_structured_json(genie: LlamacppGenie) -> None:
    """Async structured JSON output."""
    logger.info("=== Async Structured JSON ===")
    result = await genie.agenerate_json(
        "Analyze the sentiment of: 'Chonkie makes text chunking incredibly simple and fast!'",
        schema=SentimentResult,
    )
    print("\n🎭 Sentiment Analysis:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print()


def demo_error_handling(genie: LlamacppGenie) -> None:
    """Demonstrate graceful error handling for invalid structured output."""
    logger.info("=== Error Handling (expected failure) ===")

    class ImpossibleSchema(BaseModel):
        must_be_exactly_42: int = Field(description="Must always be 42", default=42)
        random_uuid: str = Field(description="A valid UUID v4")

    try:
        # This may fail validation depending on model capability
        genie.generate_json(
            "Return a random UUID and the number 42.",
            schema=ImpossibleSchema,
        )
        print("  ✅ Unexpectedly succeeded (model handled it well!)")
    except ValueError as e:
        print(f"  ⚠️ Expected validation error caught:\n{e}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("Initializing LlamacppGenie...")
    genie = LlamacppGenie(
        temperature=0.7,
        system_prompt="You are a concise technical assistant. Answer precisely.",
    )
    logger.info(f"Genie ready: {genie}")

    # Sync demos
    demo_text_generation(genie)
    demo_structured_json(genie)
    demo_batch_generation(genie)
    demo_error_handling(genie)

    # Async demos
    asyncio.run(demo_async_text_generation(genie))
    asyncio.run(demo_async_structured_json(genie))

    logger.info("✅ All demos completed successfully!")


if __name__ == "__main__":
    main()
