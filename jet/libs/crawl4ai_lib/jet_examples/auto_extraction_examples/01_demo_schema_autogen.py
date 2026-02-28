import json
import asyncio
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig, CacheMode
from crawl4ai import JsonCssExtractionStrategy

async def smart_extraction_workflow():
    """
    Step 1: Generate schema once using LLM
    Step 2: Cache schema for unlimited reuse
    Step 3: Extract from thousands of pages with zero LLM calls
    """
    cache_dir = Path("./schema_cache")
    cache_dir.mkdir(exist_ok=True)
    schema_file = cache_dir / "product_schema.json"
    if schema_file.exists():
        schema = json.load(schema_file.open())
        print("✅ Using cached schema (FREE)")
    else:
        print("🔄 Generating schema (ONE-TIME LLM COST)...")
        llm_config = LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token="env:OPENAI_API_KEY"
        )
        async with AsyncWebCrawler() as crawler:
            sample_result = await crawler.arun(
                url="https://example.com/products",
                config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            )
            sample_html = sample_result.cleaned_html[:8000]
        schema = JsonCssExtractionStrategy.generate_schema(
            html=sample_html,
            schema_type="CSS",
            query="Extract product information including name, price, description, features",
            llm_config=llm_config
        )
        json.dump(schema, schema_file.open("w"), indent=2)
        print("✅ Schema generated and cached")
    strategy = JsonCssExtractionStrategy(schema, verbose=True)
    config = CrawlerRunConfig(
        extraction_strategy=strategy,
        cache_mode=CacheMode.BYPASS
    )
    urls = [
        "https://example.com/products",
        "https://example.com/electronics",
        "https://example.com/books"
    ]
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            result = await crawler.arun(url=url, config=config)
            if result.success:
                data = json.loads(result.extracted_content)
                print(f"✅ {url}: Extracted {len(data)} items (FREE)")

asyncio.run(smart_extraction_workflow())
