import json
import asyncio
from crawl4ai import AsyncWebCrawler, RegexExtractionStrategy, CrawlerRunConfig, JsonCssExtractionStrategy

async def multi_strategy_pipeline():
    """
    Efficient pipeline using multiple non-LLM strategies:
    1. Regex for simple patterns (fastest)
    2. Schema for structured data
    3. Only use LLM if absolutely necessary
    """
    url = "https://example.com/complex-page"
    async with AsyncWebCrawler() as crawler:
        # Strategy 1: Fast regex for contact info
        regex_strategy = RegexExtractionStrategy(
            pattern=RegexExtractionStrategy.Email | RegexExtractionStrategy.PhoneUS
        )
        regex_config = CrawlerRunConfig(extraction_strategy=regex_strategy)
        regex_result = await crawler.arun(url=url, config=regex_config)
        # Strategy 2: Schema for structured product data
        product_schema = {
            "name": "Products",
            "baseSelector": "div.product",
            "fields": [
                {"name": "name", "selector": "h3", "type": "text"},
                {"name": "price", "selector": ".price", "type": "text"}
            ]
        }
        css_strategy = JsonCssExtractionStrategy(product_schema)
        css_config = CrawlerRunConfig(extraction_strategy=css_strategy)
        css_result = await crawler.arun(url=url, config=css_config)
        results = {
            "contacts": json.loads(regex_result.extracted_content) if regex_result.success else [],
            "products": json.loads(css_result.extracted_content) if css_result.success else []
        }
        print(f"✅ Extracted {len(results['contacts'])} contacts (regex)")
        print(f"✅ Extracted {len(results['products'])} products (schema)")
        return results

asyncio.run(multi_strategy_pipeline())
