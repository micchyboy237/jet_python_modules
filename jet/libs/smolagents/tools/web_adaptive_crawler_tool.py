import asyncio
import json
import os
from pathlib import Path

from crawl4ai import (
    AdaptiveConfig,
    AdaptiveCrawler,
    AsyncWebCrawler,
    CrawlState,
    LLMConfig,
)

# Load local llama.cpp URLs from environment variables
LLAMA_CPP_LLM_URL: str | None = os.getenv(
    "LLAMA_CPP_LLM_URL"
)  # e.g., http://127.0.0.1:8081/v1 (generative)
LLAMA_CPP_EMBED_URL: str | None = os.getenv(
    "LLAMA_CPP_EMBED_URL"
)  # e.g., http://127.0.0.1:8080/v1 (embeddings)

# Output directory: <script_dir>/generated/<script_name_without_ext>/
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_example_outputs(
    example_name: str,
    start_url: str,
    result: CrawlState,  # CrawlState returned by adaptive.digest()
    adaptive: AdaptiveCrawler,
    top_k_md: int = 10,
    top_k_json: int = 10,
) -> None:
    """Save full crawl state (built-in), ranked relevant content (JSON), and human-readable summary (Markdown)."""
    subdir = OUTPUT_DIR / example_name
    subdir.mkdir(parents=True, exist_ok=True)

    # 1. Save the complete CrawlState using the library's built-in method
    result.save(subdir / "crawl_state.json")

    # 2. Ranked relevant documents (top_k for JSON and Markdown)
    relevant = adaptive.get_relevant_content(top_k=top_k_json)

    # relevant_content.json – full content for the top documents
    with open(subdir / "relevant_content.json", "w") as f:
        json.dump(
            [
                {
                    "url": doc["url"],
                    "score": float(doc["score"])
                    if isinstance(doc["score"], (int, float))
                    else doc["score"],
                    "content": doc["content"],
                }
                for doc in relevant
            ],
            f,
            indent=4,
        )

    # summary.md – human-readable overview with truncated previews
    md_path = subdir / "summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Adaptive Crawling Results – {example_name}\n\n")
        f.write(f"**Query**: {result.query}\n")
        f.write(f"**Start URL**: {start_url}\n")
        if getattr(result, "embedding_model", None):
            f.write(f"**Embedding Model**: {result.embedding_model}\n")
        f.write("\n")

        confidence = result.metrics.get("confidence_score", 0.0)
        f.write(f"**Final Confidence Score**: {confidence:.4f}\n")
        f.write(f"**Pages Crawled**: {len(result.crawled_urls)}\n")
        f.write(f"**Documents in Knowledge Base**: {len(result.knowledge_base)}\n\n")

        f.write("## Top Relevant Documents\n\n")
        for i, doc in enumerate(relevant[:top_k_md], 1):
            score = (
                float(doc["score"])
                if isinstance(doc["score"], (int, float))
                else doc["score"]
            )
            f.write(
                f"### {i}. [{doc['url']}]({doc['url']}) – Relevance: {score:.4f}\n\n"
            )
            preview = doc["content"][:1500]
            if len(doc["content"]) > 1500:
                preview += "..."
            f.write(f"{preview}\n\n")
            f.write("---\n\n")


async def example_statistical_strategy(start_url: str, query: str) -> None:
    """Basic statistical strategy – no LLM/embedding needed, fast and offline."""
    async with AsyncWebCrawler() as crawler:
        adaptive = AdaptiveCrawler(crawler)
        result = await adaptive.digest(start_url=start_url, query=query)

        adaptive.print_stats()
        confidence = result.metrics.get("confidence_score", 0.0)
        print(f"Final confidence: {confidence:.4f}")
        print(f"Pages crawled: {len(result.crawled_urls)}")

        print("\nTop 5 relevant pages:")
        for doc in adaptive.get_relevant_content(top_k=5):
            score = doc["score"]
            print(f"{doc['url']} - Score: {score:.4f}")

        save_example_outputs(
            example_name="statistical_strategy",
            start_url=start_url,
            result=result,
            adaptive=adaptive,
            top_k_md=10,
            top_k_json=10,
        )


async def example_embedding_local_default(start_url: str, query: str) -> None:
    """Embedding strategy with default local sentence-transformers model."""
    config = AdaptiveConfig(
        strategy="embedding",
        confidence_threshold=0.8,
        max_pages=30,
        n_query_variations=10,  # Uses local model, no API calls
    )
    async with AsyncWebCrawler() as crawler:
        adaptive = AdaptiveCrawler(crawler, config=config)
        result = await adaptive.digest(start_url=start_url, query=query)

        adaptive.print_stats()
        confidence = result.metrics.get("confidence_score", 0.0)
        print(f"Final confidence: {confidence:.4f}")
        print(f"Pages crawled: {len(result.crawled_urls)}")

        print("\nTop 5 relevant pages:")
        relevant = adaptive.get_relevant_content(top_k=5)
        for doc in relevant:
            score = doc["score"]
            print(f"{doc['url']} - Score: {score:.4f}")

        save_example_outputs(
            example_name="embedding_local_default",
            start_url=start_url,
            result=result,
            adaptive=adaptive,
            top_k_md=10,
            top_k_json=10,
        )


async def example_embedding_custom_llama_cpp(start_url: str, query: str) -> None:
    """Embedding strategy with custom llama.cpp embedding server from env var."""
    if not LLAMA_CPP_EMBED_URL:
        raise ValueError("LLAMA_CPP_EMBED_URL environment variable not set")

    embed_config = LLMConfig(
        provider="openai/nomic-embed-text-v2-moe",  # Direct base URL (OpenAI-compatible)
        api_token="",  # Dummy/empty for local llama.cpp
        base_url="LLAMA_CPP_EMBED_URL",
    )

    config = AdaptiveConfig(
        strategy="embedding",
        confidence_threshold=0.85,
        max_pages=40,
        embedding_llm_config=embed_config,
        n_query_variations=10,  # Set to 0 if embed server cannot generate text
    )

    async with AsyncWebCrawler() as crawler:
        adaptive = AdaptiveCrawler(crawler, config=config)
        result = await adaptive.digest(start_url=start_url, query=query)

        adaptive.print_stats()
        confidence = result.metrics.get("confidence_score", 0.0)
        print(f"Final confidence: {confidence:.4f}")
        print(f"Pages crawled: {len(result.crawled_urls)}")

        print("\nTop 5 relevant pages:")
        for doc in adaptive.get_relevant_content(top_k=5):
            score = doc["score"]
            print(f"{doc['url']} - Score: {score:.4f}")

        save_example_outputs(
            example_name="embedding_custom_llama_cpp",
            start_url=start_url,
            result=result,
            adaptive=adaptive,
            top_k_md=10,
            top_k_json=10,
        )


# Example execution
if __name__ == "__main__":
    test_url = "https://docs.python.org/3/"
    test_query = "async context managers in Python"

    asyncio.run(example_statistical_strategy(test_url, test_query))
    asyncio.run(example_embedding_local_default(test_url, test_query))
    asyncio.run(example_embedding_custom_llama_cpp(test_url, test_query))
