"""Demo: Complex query with automatic decomposition."""

import asyncio
import logging

from jet.adapters.llama_cpp.tasks.rag.react_web_searcher import QueryAnalyzer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


async def main():
    analyzer = QueryAnalyzer(model="qwen3.5-uncensored:2b")

    queries = [
        "Compare the economic impact of Brexit vs COVID-19 on UK manufacturing",
        "What are the latest developments in quantum computing error correction?",
        "Who won the 2024 US presidential election?",
    ]

    for q in queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {q}")
        print(f"{'=' * 60}")

        analysis = await analyzer.analyze(q)
        print(f"\n📊 Complexity: {analysis.complexity.value}")
        print(f"💭 Reasoning: {analysis.reasoning}")
        print(f"🔍 Refined: {analysis.refined_query}")
        if analysis.sub_queries:
            print(f"📝 Sub-queries ({len(analysis.sub_queries)}):")
            for i, sq in enumerate(analysis.sub_queries, 1):
                print(f"   {i}. {sq}")
        else:
            print("📝 No decomposition needed")


if __name__ == "__main__":
    asyncio.run(main())
