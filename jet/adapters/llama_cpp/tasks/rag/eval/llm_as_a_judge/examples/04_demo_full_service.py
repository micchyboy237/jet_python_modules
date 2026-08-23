"""
04_demo_full_service.py
========================
End-to-end demo with REAL jet adapters (no mocks).
Uses hybrid_search for retrieval, achat for generation,
truncate_texts for context safety, and embed+cosine for relevancy.

Run: python 04_demo_full_service.py
"""

import asyncio
import logging

# Updated import path
from jet.adapters.llama_cpp.tasks.rag.eval.llm_as_a_judge import (
    RAGEvaluator,
    RAGService,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Real knowledge base documents
SAMPLE_DOCS = [
    "DNS translates domain names to IP addresses through hierarchical lookup.",
    "Recursive resolvers query root, TLD, and authoritative nameservers in sequence.",
    "DNS caching at browser, OS, and resolver levels reduces lookup latency.",
    "Photosynthesis converts light energy to chemical energy stored in glucose.",
    "Chlorophyll absorbs sunlight to drive CO2 + H2O → glucose + O2.",
    "The Industrial Revolution transformed manufacturing in the 18th century.",
    "Machine learning algorithms learn patterns from large training datasets.",
    "Python is a high-level programming language created by Guido van Rossum.",
]


async def main():
    evaluator = RAGEvaluator(model="qwen3.5-uncensored:2b")

    # RAGService now uses real adapters internally:
    # - hybrid_search() for retrieval (vector + cross-encoder reranking)
    # - achat() for generation with truncate_texts() context safety
    # - embed() + cosine_similarity() for answer relevancy
    service = RAGService(
        evaluator=evaluator,
        documents=SAMPLE_DOCS,
        generation_model="qwen3.5-uncensored:2b",
    )

    await service.start()
    logger.info("✅ RAGService started with real jet adapters")

    test_queries = [
        "How does DNS resolution work?",  # Good retrieval
        "Explain photosynthesis in plants",  # Good retrieval
        "What is quantum computing?",  # Bad retrieval (no match)
        "Tell me about DNS caching mechanisms",  # Good retrieval
        "Describe the history of the internet",  # Bad retrieval (no match)
    ]

    results_summary = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#' * 60}")
        print(f"# QUERY {i}/{len(test_queries)}: {query}")
        print(f"{'#' * 60}")

        response = await service.query(query)
        confidence = response.get("confidence", "unknown")
        answer_preview = response["answer"][:120] + "..."

        print(f"\n🎯 Confidence: {confidence}")
        print(f"💬 Answer: {answer_preview}")

        if response.get("debug", {}).get("gate_failed"):
            print(
                f"⛔ PRE-GEN GATE FAILED (precision={response['debug'].get('precision', 'N/A')})"
            )

        results_summary.append(
            {
                "query": query,
                "confidence": confidence,
                "gate_failed": response.get("debug", {}).get("gate_failed", False),
            }
        )

    # Wait for all queued production evals to complete
    print(
        f"\n⏳ Waiting for background eval worker to process {len(test_queries)} evaluations..."
    )
    await service._eval_queue.join()
    logger.info("All production evaluations completed")

    await service.stop()
    logger.info("Background eval worker stopped")

    # --- Final Summary ---
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Query':<45} {'Confidence':>12} {'Gate':>8}")
    print("-" * 68)
    for r in results_summary:
        gate_status = "FAIL" if r["gate_failed"] else "PASS"
        print(f"{r['query']:<45} {r['confidence']:>12} {gate_status:>8}")

    blocked = sum(1 for r in results_summary if r["gate_failed"])
    passed = len(results_summary) - blocked
    print(f"\n📊 Results: {passed} passed, {blocked} blocked by pre-gen gate")
    print(f"📊 Total queries processed: {len(results_summary)}")
    print(f"\n💡 Infrastructure used:")
    print(f"   Retrieval:  hybrid_search (vector + cross-encoder reranking)")
    print(f"   Generation: achat + truncate_texts (context-safe)")
    print(f"   Relevancy:  embed + cosine_similarity (semantic)")
    print(f"   Judge:      achat with Pydantic structured output")
    print(f"   Tracing:    Phoenix observability (auto)")


if __name__ == "__main__":
    asyncio.run(main())
