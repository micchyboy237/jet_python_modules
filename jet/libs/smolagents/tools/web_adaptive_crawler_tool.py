import asyncio
import os

from crawl4ai import AdaptiveConfig, AdaptiveCrawler, AsyncWebCrawler, LLMConfig

# Load local llama.cpp URLs from environment variables
LLAMA_CPP_LLM_URL: str | None = os.getenv(
    "LLAMA_CPP_LLM_URL"
)  # e.g., http://127.0.0.1:8081/v1 (generative)
LLAMA_CPP_EMBED_URL: str | None = os.getenv(
    "LLAMA_CPP_EMBED_URL"
)  # e.g., http://127.0.0.1:8080/v1 (embeddings)


async def example_statistical_strategy(start_url: str, query: str) -> None:
    """Basic statistical strategy - no LLM/embedding needed, fast and offline."""
    async with AsyncWebCrawler() as crawler:
        adaptive = AdaptiveCrawler(crawler)
        result = await adaptive.digest(start_url=start_url, query=query)
        adaptive.print_stats()
        print(f"Final confidence: {result.metrics.get('confidence', 0):.2f}")
        print(f"Pages crawled: {len(result.crawled_urls)}")


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
        relevant = adaptive.get_relevant_content(top_k=5)
        for doc in relevant:
            print(f"{doc['url']} - Score: {doc['score']:.2%}")


async def example_embedding_custom_llama_cpp(start_url: str, query: str) -> None:
    """Embedding strategy with custom llama.cpp embedding server from env var.
    Disable query variations if your embed model/server lacks chat/completions support."""
    if not LLAMA_CPP_EMBED_URL:
        raise ValueError("LLAMA_CPP_EMBED_URL environment variable not set")

    embed_config = LLMConfig(
        provider=LLAMA_CPP_EMBED_URL,  # Direct base URL (OpenAI-compatible)
        api_token="",  # Dummy/empty for local llama.cpp
        # model="your-embed-model"      # Optional: specify if needed by server
    )

    config = AdaptiveConfig(
        strategy="embedding",
        confidence_threshold=0.85,
        max_pages=40,
        embedding_llm_config=embed_config,
        n_query_variations=10,  # Set to 0 if embed server cannot generate text
        # n_query_variations=0,         # Uncomment to disable expansion if needed
    )

    async with AsyncWebCrawler() as crawler:
        adaptive = AdaptiveCrawler(crawler, config=config)
        result = await adaptive.digest(start_url=start_url, query=query)
        adaptive.print_stats()
        print(f"Confidence reached: {result.metrics.get('confidence', 0):.2f}")
        for doc in adaptive.get_relevant_content(top_k=3):
            print(
                f"\n--- {doc['url']} (relevance {doc['score']:.2%}) ---\n{doc['content'][:500]}..."
            )


# Example execution
if __name__ == "__main__":
    test_url = "https://docs.python.org/3/"
    test_query = "async context managers in Python"

    asyncio.run(example_statistical_strategy(test_url, test_query))
    # asyncio.run(example_embedding_local_default(test_url, test_query))
    # asyncio.run(example_embedding_custom_llama_cpp(test_url, test_query))
