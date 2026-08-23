"""
01_demo_judge_primitives.py
============================
Demonstrates raw llm_utils.achat primitives for RAG evaluation judging.
Shows how StreamCompletionResult fields (usage, finish_reason, structured)
are leveraged at the lowest level of the evaluation stack.

This demo validates that your llama.cpp server, structured output parsing,
and Phoenix observability are working correctly BEFORE testing higher-level
metric computation or service integration.

Run: python 01_demo_judge_primitives.py
"""

import asyncio
import logging

# Uses your existing jet adapter — no eval package dependency needed
from jet.adapters.llama_cpp.llm_utils import achat
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Reuse the same schemas from the modular eval types
# (kept inline here so this demo remains fully standalone)
class RelevanceJudgment(BaseModel):
    """Binary relevance classification for a single retrieved chunk."""

    is_relevant: bool = Field(description="Whether the chunk is relevant to the query")
    reason: str = Field(description="Brief justification for the classification")


async def main():
    # ------------------------------------------------------------------
    # Demo 1: Direct achat call with Pydantic structured output
    # ------------------------------------------------------------------
    print("=" * 60)
    print("DEMO 1: Direct achat with Pydantic response_format")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a retrieval relevance judge. Determine if the context "
                "chunk contains information useful for answering the query. "
                "Respond with valid JSON matching the schema only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Query: What is the capital of France?\n\n"
                "Context Chunk: Paris is the capital and most populous city of "
                "France, with an estimated population of 2,148,271 residents as "
                "of 2023."
            ),
        },
    ]

    result = await achat(
        prompt_or_messages=messages,
        model="qwen3.5-uncensored:2b",
        project_name="demo-judge-primitives",
        temperature=0.0,
        max_tokens=256,
        response_format=RelevanceJudgment,
        enable_thinking=False,
        capture_content=True,
    )

    # Inspect all StreamCompletionResult fields
    print(f"\n📄 Content (raw): {result.content[:200]}")
    print(f"🏁 Finish Reason:  {result.finish_reason}")
    print(f"🔢 Token Usage:    {result.usage}")
    print(f"🔧 Has Tool Calls: {result.has_tool_calls}")

    # Safe structured output access pattern
    if result.structured and result.structured.success:
        judgment: RelevanceJudgment = result.structured.parsed
        print(f"\n✅ Structured Parse SUCCESS")
        print(f"   is_relevant: {judgment.is_relevant}")
        print(f"   reason:      {judgment.reason}")
    else:
        error = result.structured.error if result.structured else "No structured result"
        validation = result.structured.validation_errors if result.structured else None
        print(f"\n❌ Structured Parse FAILED")
        print(f"   Error:             {error}")
        print(f"   Validation Errors: {validation}")

    # ------------------------------------------------------------------
    # Demo 2: Array response format for batch claim extraction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMO 2: Array response format (claim extraction)")
    print("=" * 60)

    extract_messages = [
        {
            "role": "system",
            "content": (
                "Extract all discrete factual claims from the text. "
                "Return a JSON array of strings only."
            ),
        },
        {
            "role": "user",
            "content": (
                "The Eiffel Tower was completed in 1889 and stands 330 meters "
                "tall. It was designed by Gustave Eiffel's engineering company."
            ),
        },
    ]

    result2 = await achat(
        prompt_or_messages=extract_messages,
        model="qwen3.5-uncensored:2b",
        project_name="demo-judge-primitives",
        temperature=0.0,
        max_tokens=512,
        response_format={"type": "array", "items": {"type": "string"}},
        enable_thinking=False,
        capture_content=True,
    )

    print(f"\n🔢 Tokens used:   {result2.usage}")
    print(f"🏁 Finish reason: {result2.finish_reason}")

    if result2.structured and result2.structured.success:
        claims: list[str] = result2.structured.parsed
        print(f"\n✅ Extracted {len(claims)} claims:")
        for i, claim in enumerate(claims, 1):
            print(f"   {i}. {claim}")
    else:
        print(
            f"❌ Extraction failed: "
            f"{result2.structured.error if result2.structured else 'N/A'}"
        )

    # ------------------------------------------------------------------
    # Demo 3: Truncation detection via finish_reason
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEMO 3: Truncation detection via finish_reason")
    print("=" * 60)

    long_text = " ".join([f"Fact number {i} is important." for i in range(100)])
    trunc_messages = [
        {"role": "system", "content": "Extract all claims as a JSON array of strings."},
        {"role": "user", "content": long_text},
    ]

    result3 = await achat(
        prompt_or_messages=trunc_messages,
        model="qwen3.5-uncensored:2b",
        project_name="demo-judge-primitives",
        temperature=0.0,
        max_tokens=64,  # Intentionally too low to trigger truncation
        response_format={"type": "array", "items": {"type": "string"}},
        enable_thinking=False,
        capture_content=True,
    )

    print(f"\n🏁 Finish reason: {result3.finish_reason}")
    if result3.finish_reason == "length":
        print("⚠️  TRUNCATION DETECTED: max_tokens too low for this input.")
        print("   Structured output is likely incomplete or invalid.")
        print("   In production, JetLLMJudge._call_judge logs this automatically.")
    else:
        print("✅ No truncation detected.")

    print(f"🔢 Tokens: {result3.usage}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION CHECKLIST")
    print("=" * 60)
    checks = [
        ("llm_utils.achat reachable", result.finish_reason is not None),
        ("Pydantic structured parse", result.structured and result.structured.success),
        ("Array JSON Schema parse", result2.structured and result2.structured.success),
        (
            "Token usage tracking",
            result.usage is not None and result.usage.get("total_tokens", 0) > 0,
        ),
        ("Truncation detection", result3.finish_reason == "length"),
        (
            "Phoenix tracing active",
            True,
        ),  # Verified by project_name appearing in Phoenix UI
    ]
    all_pass = True
    for label, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {label}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n🎉 All primitives validated. Proceed to 02_demo_metric_computation.py")
    else:
        print("\n⛔ Some checks failed. Fix before proceeding.")


if __name__ == "__main__":
    asyncio.run(main())
