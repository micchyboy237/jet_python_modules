import asyncio
import json
import shutil
from pathlib import Path

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
)
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def smart_extraction_workflow():
    """
    Step 1: Generate schema once using LLM
    Step 2: Cache schema for unlimited reuse
    Step 3: Extract from thousands of pages with zero LLM calls
    """
    schema_file = OUTPUT_DIR / "product_schema.json"
    if schema_file.exists():
        schema = json.load(schema_file.open())
        print("✅ Using cached schema (FREE)")
    else:
        print("🔄 Generating schema (ONE-TIME LLM COST)...")
        llm_config = get_llm_config(strategy="llm")
        async with AsyncWebCrawler() as crawler:
            sample_result = await crawler.arun(
                url="https://webscraper.io/test-sites/e-commerce/allinone",
                config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
            )
            sample_html = sample_result.cleaned_html[:8000]
        schema = JsonCssExtractionStrategy.generate_schema(
            html=sample_html,
            schema_type="CSS",
            query="Extract product information including name, price, description, features",
            llm_config=llm_config,
        )
        print("---- Generated Schema ----")
        print(json.dumps(schema, indent=2, ensure_ascii=False))
        json.dump(schema, schema_file.open("w"), indent=2)
        print("✅ Schema generated and cached")

    strategy = JsonCssExtractionStrategy(schema, verbose=True)
    config = CrawlerRunConfig(extraction_strategy=strategy, cache_mode=CacheMode.BYPASS)
    urls = [
        "https://webscraper.io/test-sites/e-commerce/allinone",
        "https://webscraper.io/test-sites/e-commerce/allinone/computers",
        "https://webscraper.io/test-sites/e-commerce/allinone/phones",
    ]
    async with AsyncWebCrawler() as crawler:
        for url in urls:
            result = await crawler.arun(url=url, config=config)
            if result.success:
                data = json.loads(result.extracted_content)
                print(f"✅ {url}: Extracted {len(data)} items (FREE)")


asyncio.run(smart_extraction_workflow())
