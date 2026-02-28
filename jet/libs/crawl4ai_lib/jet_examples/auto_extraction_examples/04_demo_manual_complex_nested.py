import json
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai import JsonCssExtractionStrategy

complex_schema = {
    "name": "E-commerce Product Catalog",
    "baseSelector": "div.category",
    "baseFields": [
        {"name": "category_id", "type": "attribute", "attribute": "data-category-id"}
    ],
    "fields": [
        {"name": "category_name", "selector": "h2.category-title", "type": "text"},
        {
            "name": "products",
            "selector": "div.product",
            "type": "nested_list",
            "fields": [
                {"name": "name", "selector": "h3.product-name", "type": "text"},
                {"name": "price", "selector": "span.price", "type": "text"},
                {
                    "name": "details",
                    "selector": "div.product-details",
                    "type": "nested",
                    "fields": [
                        {"name": "brand", "selector": "span.brand", "type": "text"},
                        {"name": "model", "selector": "span.model", "type": "text"}
                    ]
                },
                {
                    "name": "features",
                    "selector": "ul.features li",
                    "type": "list",
                    "fields": [{"name": "feature", "type": "text"}]
                },
                {
                    "name": "reviews",
                    "selector": "div.review",
                    "type": "nested_list",
                    "fields": [
                        {"name": "reviewer", "selector": "span.reviewer-name", "type": "text"},
                        {"name": "rating", "selector": "span.rating", "type": "attribute", "attribute": "data-rating"}
                    ]
                }
            ]
        }
    ]
}

async def extract_complex_ecommerce():
    strategy = JsonCssExtractionStrategy(complex_schema, verbose=True)
    config = CrawlerRunConfig(
        extraction_strategy=strategy,
        js_code="window.scrollTo(0, document.body.scrollHeight);",
        wait_for="css:.product:nth-child(10)"
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://example.com/complex-catalog",
            config=config
        )
        if result.success:
            data = json.loads(result.extracted_content)
            for category in data:
                print(f"Category: {category['category_name']}")
                print(f"Products: {len(category.get('products', []))}")

asyncio.run(extract_complex_ecommerce())
