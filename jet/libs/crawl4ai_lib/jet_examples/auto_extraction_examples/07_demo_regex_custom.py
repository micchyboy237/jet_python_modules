import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import RegexExtractionStrategy

async def extract_custom_patterns():
    custom_patterns = {
        "product_sku": r"SKU[-:]?\s*([A-Z0-9]{4,12})",
        "discount": r"(\d{1,2})%\s*off",
        "model_number": r"Model\s*#?\s*([A-Z0-9-]+)",
        "isbn": r"ISBN[-:]?\s*(\d{10}|\d{13})",
        "stock_ticker": r"\$([A-Z]{2,5})",
        "version": r"v(\d+\.\d+(?:\.\d+)?)"
    }
    strategy = RegexExtractionStrategy(custom=custom_patterns)
    config = CrawlerRunConfig(extraction_strategy=strategy)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/products",
            config=config
        )
        if result.success:
            data = json.loads(result.extracted_content)
            for item in data:
                print(f"{item['label']}: {item['value']}")

asyncio.run(extract_custom_patterns())
