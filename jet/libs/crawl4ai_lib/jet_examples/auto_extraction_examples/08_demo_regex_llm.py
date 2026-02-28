import json
import asyncio
from pathlib import Path
from crawl4ai import AsyncWebCrawler, RegexExtractionStrategy, LLMConfig

async def generate_optimized_regex():
    """
    Use LLM ONCE to generate optimized regex patterns,
    then use them unlimited times with zero LLM calls.
    """
    cache_file = Path("./patterns/price_patterns.json")
    if cache_file.exists():
        patterns = json.load(cache_file.open())
        print("✅ Using cached regex patterns (FREE)")
    else:
        print("🔄 Generating regex patterns (ONE-TIME LLM COST)...")
        llm_config = LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token="env:OPENAI_API_KEY"
        )
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun("https://example.com/pricing")
            sample_html = result.cleaned_html
        patterns = RegexExtractionStrategy.generate_pattern(
            label="pricing_info",
            html=sample_html,
            query="Extract all pricing information including discounts and special offers",
            llm_config=llm_config
        )
        cache_file.parent.mkdir(exist_ok=True)
        json.dump(patterns, cache_file.open("w"), indent=2)
        print("✅ Patterns generated and cached")
    strategy = RegexExtractionStrategy(custom=patterns)
    return strategy

# Use generated patterns for unlimited extractions
# strategy = await generate_optimized_regex()
