import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import JsonCssExtractionStrategy

# Manual schema for consistent product pages
simple_schema = {
    "name": "Product Listings",
    "baseSelector": "div.product-card",
    "fields": [
        {"name": "title", "selector": "h2.product-title", "type": "text"},
        {"name": "price", "selector": ".price", "type": "text"},
        {"name": "image_url", "selector": "img.product-image", "type": "attribute", "attribute": "src"},
        {"name": "product_url", "selector": "a.product-link", "type": "attribute", "attribute": "href"},
        {"name": "rating", "selector": ".rating", "type": "attribute", "attribute": "data-rating"}
    ]
}

async def extract_products():
    strategy = JsonCssExtractionStrategy(simple_schema, verbose=True)
    config = CrawlerRunConfig(extraction_strategy=strategy)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/products",
            config=config
        )
        if result.success:
            products = json.loads(result.extracted_content)
            print(f"Extracted {len(products)} products")
            for product in products[:3]:
                print(f"- {product['title']}: {product['price']}")

asyncio.run(extract_products())
