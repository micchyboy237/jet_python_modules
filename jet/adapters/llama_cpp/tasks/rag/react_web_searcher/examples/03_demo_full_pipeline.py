"""Demo: Full end-to-end ReAct web search pipeline."""

import asyncio
import logging

from jet.adapters.llama_cpp.tasks.rag.react_web_searcher import ReactEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


async def main():
    engine = ReactEngine(
        model="qwen3.5-uncensored:2b",
        max_iterations=10,
        enable_validation=True,
    )

    queries = [
        "What is the current population of Tokyo?",
        "Compare renewable energy adoption in Germany vs Japan in 2024",
        "Explain how CRISPR gene editing works and its latest medical applications",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'#' * 60}")
        print(f"# QUERY {i}/{len(queries)}: {query}")
        print(f"{'#' * 60}")

        result = await engine.search(query)

        print(f"\n🎯 Confidence: {result.confidence}")
        print(f"📊 Agent Steps: {len(result.steps)}")
        print(f"🔢 Total Tokens: {result.total_tokens}")

        for j, step in enumerate(result.steps, 1):
            print(f"   Step {j}: {step.action}({list(step.action_input.keys())})")

        print(f"\n💬 Answer:\n{result.answer[:500]}...")

        if result.eval_result:
            ev = result.eval_result
            status = "🚨 FAIL" if ev.get("has_critical_failure") else "✅ PASS"
            print(f"\n📋 Validation [{status}]:")
            print(f"   Faithfulness:  {ev.get('faithfulness', 'N/A')}")
            print(f"   Hallucination: {ev.get('hallucination_rate', 'N/A')}")
            print(f"   Relevancy:     {ev.get('answer_relevancy', 'N/A')}")
            print(f"   Eval Tokens:   {ev.get('total_eval_tokens', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
