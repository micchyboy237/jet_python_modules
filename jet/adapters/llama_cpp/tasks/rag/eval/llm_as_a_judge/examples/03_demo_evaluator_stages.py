"""
03_demo_evaluator_stages.py
============================
Demonstrates all three RAGEvaluator stages with unified RAGEvaluationResult.
Updated: imports from modular eval package.

Run: python 03_demo_evaluator_stages.py
"""

import asyncio
import logging

# Updated import path
from jet.adapters.llama_cpp.tasks.rag.eval.llm_as_a_judge import RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    evaluator = RAGEvaluator(model="qwen3.5-uncensored:2b")

    query = "How does DNS resolution work?"
    contexts = [
        "DNS (Domain Name System) translates human-readable domain names like "
        "example.com into IP addresses that computers use to communicate.",
        "The resolution process starts with a recursive resolver querying root "
        "servers, then TLD servers (.com), then authoritative nameservers.",
        "DNS caching at multiple levels (browser, OS, resolver) reduces latency "
        "for repeated lookups.",
    ]
    response = (
        "DNS resolution translates domain names to IP addresses through a "
        "hierarchical lookup process. A recursive resolver queries root servers, "
        "then TLD servers, then authoritative nameservers to find the correct IP. "
        "Results are cached at various levels to speed up future requests."
    )
    reference = (
        "DNS resolution converts domain names to IP addresses. The process involves "
        "a recursive resolver that queries root nameservers, top-level domain servers, "
        "and finally authoritative nameservers. Caching occurs at the browser, OS, "
        "and resolver levels to improve performance."
    )

    # --- Stage 1: Pre-Generation Gate ---
    print("=" * 60)
    print("STAGE 1: Pre-Generation Gate (sync, blocks generation)")
    print("=" * 60)

    gate_result = await evaluator.evaluate_pre_generation_gate(query, contexts)
    print(f"\n📊 Contextual Precision: {gate_result.contextual_precision:.3f}")
    print(f"🚦 Gate Passed:          {gate_result.passed_gate}")
    print(f"🔢 Eval Tokens Used:     {gate_result.total_eval_tokens}")
    print(f"📋 Stage:                {gate_result.stage.value}")

    if not gate_result.passed_gate:
        print("⛔ Generation would be BLOCKED — returning fallback response")
    else:
        print("✅ Generation would PROCEED")

    # Test with bad contexts
    print("\n--- Testing with irrelevant contexts ---")
    bad_contexts = ["The moon landing happened in 1969.", "Python uses indentation."]
    gate_bad = await evaluator.evaluate_pre_generation_gate(query, bad_contexts)
    print(
        f"📊 Precision: {gate_bad.contextual_precision:.3f} | Gate: {'PASS' if gate_bad.passed_gate else 'FAIL'}"
    )

    # --- Stage 2: Production Async Evaluation ---
    print("\n" + "=" * 60)
    print("STAGE 2: Production Async Eval (reference-free safety)")
    print("=" * 60)

    prod_result = await evaluator.evaluate_production_async(query, contexts, response)
    print(f"\n📊 Faithfulness:       {prod_result.faithfulness:.3f}")
    print(f"📊 Hallucination Rate: {prod_result.hallucination_rate:.3f}")
    print(f"📊 Answer Relevancy:   {prod_result.answer_relevancy:.3f} [SEMANTIC]")
    print(f"🔢 Eval Tokens Used:   {prod_result.total_eval_tokens}")
    print(f"🚨 Critical Failure:   {prod_result.has_critical_failure}")
    print(f"📋 Stage:              {prod_result.stage.value}")

    # Test with hallucinated response
    print("\n--- Testing with hallucinated response ---")
    bad_response = (
        "DNS was invented by Tim Berners-Lee in 1989. It uses blockchain "
        "technology to verify domain ownership and encrypts all queries with AES-256."
    )
    prod_bad = await evaluator.evaluate_production_async(query, contexts, bad_response)
    print(
        f"📊 Faithfulness: {prod_bad.faithfulness:.3f} | Hallucination: {prod_bad.hallucination_rate:.3f}"
    )
    print(f"🚨 Critical Failure: {prod_bad.has_critical_failure}")

    # --- Stage 3: Offline Benchmark ---
    print("\n" + "=" * 60)
    print("STAGE 3: Offline Benchmark (full suite with ground truth)")
    print("=" * 60)

    offline_result = await evaluator.evaluate_offline(
        query, contexts, response, reference
    )
    print(f"\n📊 Contextual Precision: {offline_result.contextual_precision:.3f}")
    print(f"📊 Contextual Recall:    {offline_result.contextual_recall:.3f}")
    print(f"📊 Faithfulness:         {offline_result.faithfulness:.3f}")
    print(f"📊 Hallucination Rate:   {offline_result.hallucination_rate:.3f}")
    print(f"📊 Answer Relevancy:     {offline_result.answer_relevancy:.3f} [SEMANTIC]")
    print(f"🔢 Total Eval Tokens:    {offline_result.total_eval_tokens}")
    print(f"📋 Metadata:             {offline_result.metadata}")
    print(f"📋 Stage:                {offline_result.stage.value}")

    # --- Cross-stage comparison ---
    print("\n" + "=" * 60)
    print("CROSS-STAGE COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Pre-Gen':>10} {'Prod':>10} {'Offline':>10}")
    print("-" * 58)
    print(
        f"{'Contextual Precision':<25} {gate_result.contextual_precision or 0:>10.3f} "
        f"{'—':>10} {offline_result.contextual_precision or 0:>10.3f}"
    )
    print(
        f"{'Faithfulness':<25} {'—':>10} {prod_result.faithfulness or 0:>10.3f} "
        f"{offline_result.faithfulness or 0:>10.3f}"
    )
    print(
        f"{'Answer Relevancy':<25} {'—':>10} {prod_result.answer_relevancy or 0:>10.3f} "
        f"{offline_result.answer_relevancy or 0:>10.3f}"
    )
    print(
        f"{'Contextual Recall':<25} {'—':>10} {'—':>10} {offline_result.contextual_recall or 0:>10.3f}"
    )
    print(
        f"{'Total Tokens':<25} {gate_result.total_eval_tokens:>10} "
        f"{prod_result.total_eval_tokens:>10} {offline_result.total_eval_tokens:>10}"
    )


if __name__ == "__main__":
    asyncio.run(main())
