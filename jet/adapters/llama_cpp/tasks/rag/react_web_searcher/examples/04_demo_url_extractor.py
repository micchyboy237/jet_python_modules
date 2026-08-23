"""
Demonstrates the UrlContextExtractor for robust web content extraction.

Uses Trafilatura to strip boilerplate (ads, navbars) and extract main content.
"""

import asyncio
import logging

from jet.adapters.llama_cpp.tasks.rag.react_web_searcher.url_extractor import (
    UrlContextExtractor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    query = "Top isekai / reincarnation anime 2026"
    # Test URLs covering different scenarios
    test_urls = [
        "https://en.wikipedia.org/wiki/Isekai",
        "https://myanimelist.net/stacks/28254",
        "https://httpbin.org/status/404",  # Expected failure
    ]

    # Initialize the extractor with a specific model for token-aware truncation
    extractor = UrlContextExtractor(
        model="qwen3.5-uncensored:2b",
        max_tokens=1024,  # Limit output to ~1024 tokens for the demo
    )

    print("=" * 60)
    print("URL CONTEXT EXTRACTION DEMO")
    print("=" * 60)

    for url in test_urls:
        print(f"\n{'-' * 60}")
        print(f"🔍 Testing: {url}")
        print(f"{'-' * 60}")

        content, error = await extractor.extract(url, query=query)

        if error:
            print(f"❌ Error: {error}")
        else:
            print(f"✅ Success!")
            print(f"   Length: {len(content)} characters")
            print(f"   Max Tokens: {extractor.max_tokens}")
            print(f"   Preview:\n{content[:300]}...")

    print(f"\n{'=' * 60}")
    print("Demo complete. Check logs for detailed extraction info.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
