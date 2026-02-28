import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import RegexExtractionStrategy

async def extract_common_patterns():
    strategy = RegexExtractionStrategy(
        pattern=(
            RegexExtractionStrategy.Email |
            RegexExtractionStrategy.PhoneUS |
            RegexExtractionStrategy.Url |
            RegexExtractionStrategy.Currency |
            RegexExtractionStrategy.DateIso
        )
    )
    config = CrawlerRunConfig(extraction_strategy=strategy)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/contact",
            config=config
        )
        if result.success:
            matches = json.loads(result.extracted_content)
            by_type = {}
            for match in matches:
                label = match['label']
                by_type.setdefault(label, []).append(match['value'])
            for pattern_type, values in by_type.items():
                print(f"{pattern_type}: {len(values)} matches")
                for value in values[:3]:
                    print(f"  {value}")

asyncio.run(extract_common_patterns())
