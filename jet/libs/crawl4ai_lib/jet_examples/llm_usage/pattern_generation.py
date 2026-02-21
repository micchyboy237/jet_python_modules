import asyncio
import json
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, RegexExtractionStrategy
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config


async def extract_with_generated_pattern():
    cache_dir = Path("./pattern_cache")
    cache_dir.mkdir(exist_ok=True)
    pattern_file = cache_dir / "price_pattern.json"

    # 1. Generate or load pattern
    if pattern_file.exists():
        pattern = json.load(pattern_file.open())
        print(f"Using cached pattern: {pattern}")
    else:
        print("Generating pattern via LLM...")

        # Configure LLM
        llm_config = get_llm_config()

        # Get sample HTML for context
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun("https://example.com/products")
            html = result.fit_html

        # Generate pattern (one-time LLM usage)
        pattern = RegexExtractionStrategy.generate_pattern(
            label="price",
            html=html,
            query="Product prices in USD format",
            llm_config=llm_config,
        )

        # Cache pattern for future use
        json.dump(pattern, pattern_file.open("w"), indent=2)

    # 2. Use pattern for extraction (no LLM calls)
    strategy = RegexExtractionStrategy(custom=pattern)
    config = CrawlerRunConfig(extraction_strategy=strategy)

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://example.com/products", config=config)

        if result.success:
            data = json.loads(result.extracted_content)
            for item in data[:10]:
                print(f"Extracted: {item['value']}")
            print(f"Total matches: {len(data)}")


asyncio.run(extract_with_generated_pattern())
