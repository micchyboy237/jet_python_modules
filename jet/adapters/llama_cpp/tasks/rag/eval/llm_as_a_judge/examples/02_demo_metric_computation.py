"""
02_demo_metric_computation.py
==============================
Demonstrates RAGMetrics computation using synthetic examples.
Updated: imports from modular eval package; answer relevancy now
uses semantic similarity via embed_utils instead of lexical overlap.

Run: python 02_demo_metric_computation.py
"""

import asyncio
import logging

# Updated import path (was: from rag_eval_pipeline import ...)
from jet.adapters.llama_cpp.tasks.rag.eval.llm_as_a_judge import JetLLMJudge, RAGMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    judge = JetLLMJudge(model="qwen3.5-uncensored:2b")
    metrics = RAGMetrics(judge)

    # --- Example 1: Good retrieval + faithful answer ---
    print("=" * 60)
    print("EXAMPLE 1: Good RAG response")
    print("=" * 60)

    query_good = "What causes photosynthesis?"
    contexts_good = [
        "Photosynthesis is the process by which plants convert light energy "
        "into chemical energy stored in glucose.",
        "Chlorophyll in plant cells absorbs sunlight, which drives the reaction "
        "between carbon dioxide and water to produce glucose and oxygen.",
    ]
    response_good = (
        "Photosynthesis occurs when chlorophyll in plant cells absorbs sunlight. "
        "This energy drives a reaction between CO2 and water, producing glucose "
        "and oxygen as byproducts."
    )

    precision, prec_tokens = await metrics.compute_contextual_precision(
        query_good, contexts_good
    )
    faithfulness, halluc_rate, faith_tokens = await metrics.compute_faithfulness(
        response_good, contexts_good
    )
    # Now uses embed_utils.embed + scoring_utils.cosine_similarity internally
    relevancy, rel_tokens = await metrics.compute_answer_relevancy(
        query_good, response_good
    )

    print(f"\n📊 Contextual Precision: {precision:.3f} ({prec_tokens} tokens)")
    print(f"📊 Faithfulness:        {faithfulness:.3f} ({faith_tokens} tokens)")
    print(f"📊 Hallucination Rate:  {halluc_rate:.3f}")
    print(f"📊 Answer Relevancy:    {relevancy:.3f} ({rel_tokens} tokens) [SEMANTIC]")
    print(f"💰 Total Eval Tokens:   {prec_tokens + faith_tokens + rel_tokens}")

    # --- Example 2: Bad retrieval (irrelevant chunks) ---
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Bad retrieval (irrelevant context)")
    print("=" * 60)

    query_bad = "What is the boiling point of water?"
    contexts_bad = [
        "The French Revolution began in 1789 with the storming of the Bastille.",
        "Quantum entanglement describes correlated particle states across distance.",
        "Water boils at 100°C at standard atmospheric pressure.",
    ]

    precision_bad, bad_tokens = await metrics.compute_contextual_precision(
        query_bad, contexts_bad
    )
    print(f"\n📊 Contextual Precision: {precision_bad:.3f} ({bad_tokens} tokens)")
    print(f"   Expected: ~0.33 (only 1 of 3 chunks relevant, ranked last)")

    if precision_bad < RAGMetrics.CONTEXT_PRECISION_THRESHOLD:
        print(
            f"   ⛔ BELOW THRESHOLD ({RAGMetrics.CONTEXT_PRECISION_THRESHOLD}) → Would block generation"
        )
    else:
        print(f"   ✅ Above threshold → Generation would proceed")

    # --- Example 3: Hallucinating response ---
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Hallucinating response")
    print("=" * 60)

    response_hallucinated = (
        "Photosynthesis occurs when chlorophyll absorbs sunlight. "
        "The process requires magnesium as a catalyst and produces methane "
        "as a primary byproduct. It was discovered by Isaac Newton in 1687."
    )

    faith_h, halluc_h, tokens_h = await metrics.compute_faithfulness(
        response_hallucinated, contexts_good
    )
    print(f"\n📊 Faithfulness:       {faith_h:.3f} ({tokens_h} tokens)")
    print(f"📊 Hallucination Rate: {halluc_h:.3f}")
    print(f"   Expected: Low faithfulness, high hallucination rate")
    print(f"   (methane, magnesium catalyst, Newton are not in context)")

    if halluc_h > RAGMetrics.HALLUCINATION_THRESHOLD:
        print(
            f"   🚨 CRITICAL: Hallucination rate exceeds {RAGMetrics.HALLUCINATION_THRESHOLD}"
        )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Thresholds configured:")
    print(f"  Contextual Precision: ≥ {RAGMetrics.CONTEXT_PRECISION_THRESHOLD}")
    print(f"  Faithfulness:         ≥ {RAGMetrics.FAITHFULNESS_THRESHOLD}")
    print(f"  Answer Relevancy:     ≥ {RAGMetrics.ANSWER_RELEVANCY_THRESHOLD}")
    print(f"  Hallucination Rate:   ≤ {RAGMetrics.HALLUCINATION_THRESHOLD}")
    print(f"\n✅ Answer Relevancy now uses semantic similarity (embed_utils)")
    print(f"   instead of naive lexical word overlap.")


if __name__ == "__main__":
    asyncio.run(main())
