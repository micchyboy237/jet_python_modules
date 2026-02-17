import asyncio
import json

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, JsonCssExtractionStrategy
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config


async def extract_crypto_prices_xpath():
    html = """
    <div class="product-card">
        <h2 class="title">Gaming Laptop</h2>
        <div class="price">$999.99</div>
        <div class="specs">
            <ul>
                <li>16GB RAM</li>
                <li>1TB SSD</li>
            </ul>
        </div>
    </div>
    """

    # Option 1: Using OpenAI (requires API token)
    css_schema = JsonCssExtractionStrategy.generate_schema(
        html, schema_type="css", llm_config=get_llm_config()
    )

    # Use the generated schema for fast, repeated extractions
    strategy = JsonCssExtractionStrategy(css_schema)

    # 3. Place the strategy in the CrawlerRunConfig
    config = CrawlerRunConfig(extraction_strategy=strategy)

    # 4. Use raw:// scheme to pass dummy_html directly
    raw_url = f"raw://{html}"

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=raw_url, config=config)

        if not result.success:
            print("Crawl failed:", result.error_message)
            return

        data = json.loads(result.extracted_content)
        print(f"Extracted {len(data)} coin rows")
        if data:
            print("First item:", data[0])


asyncio.run(extract_crypto_prices_xpath())
