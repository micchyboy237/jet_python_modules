"""Demo: Simple factual query through the ReAct pipeline."""

import asyncio
import logging

from jet.adapters.llama_cpp.tasks.rag.react_web_searcher import ReactEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


async def main():
    engine = ReactEngine(model="qwen3.5-uncensored:2b", max_iterations=5)

    print("=" * 60)
    print("SIMPLE QUERY: What is the capital of France?")
    print("=" * 60)

    result = await engine.search("What is the capital of France?")

    print(f"\n🎯 Confidence: {result.confidence}")
    print(f"📊 Steps: {len(result.steps)}")
    print(f"🔢 Tokens: {result.total_tokens}")
    print(f"\n💬 Answer:\n{result.answer}")

    if result.eval_result:
        print(f"\n📋 Validation:")
        print(f"   Faithfulness: {result.eval_result.get('faithfulness', 'N/A')}")
        print(
            f"   Hallucination: {result.eval_result.get('hallucination_rate', 'N/A')}"
        )
        print(f"   Relevancy: {result.eval_result.get('answer_relevancy', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
