"""
**Cost Analysis**:
- Non-LLM: ~$0.000001 per page
- LLM: ~$0.01-$0.10 per page (10,000x more expensive)
---
## 1. Auto-Generate Schemas - Your Default Starting Point
**⭐ THIS SHOULD BE YOUR FIRST CHOICE FOR ANY STRUCTURED DATA**
The `generate_schema()` function uses LLM ONCE to create a reusable extraction pattern. After generation, you extract unlimited pages with ZERO LLM calls.
### Basic Auto-Generation Workflow
"""

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
from jet.file.utils import save_file
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
    # Check for cached schema first
    cache_dir = OUTPUT_DIR / "schema_cache"
    cache_dir.mkdir(exist_ok=True)
    schema_file = cache_dir / "product_schema.json"

    if schema_file.exists():
        # Load cached schema - NO LLM CALLS
        schema = json.load(schema_file.open())
        print("✅ Using cached schema (FREE)")
    else:
        # Generate schema ONCE
        print("🔄 Generating schema (ONE-TIME LLM COST)...")
        llm_config = get_llm_config(strategy="llm")

        # Get sample HTML from target site
        async with AsyncWebCrawler() as crawler:
            sample_result = await crawler.arun(
                url="https://www.bookdepository.com/category/1000/All",
                config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
            )
            sample_html = sample_result.cleaned_html[:8000]  # truncated sample

        # AUTO-GENERATE SCHEMA (ONE LLM CALL)
        schema = JsonCssExtractionStrategy.generate_schema(
            html=sample_html,
            schema_type="CSS",  # or "XPATH"
            query="Extract product information including name, price, description, features, author, rating",
            llm_config=llm_config,
        )

        # Cache for unlimited future use
        json.dump(schema, schema_file.open("w"), indent=2)
        print("✅ Schema generated and cached")

    # Use schema for fast extraction (NO MORE LLM CALLS EVER)
    strategy = JsonCssExtractionStrategy(schema, verbose=True)
    config = CrawlerRunConfig(extraction_strategy=strategy, cache_mode=CacheMode.BYPASS)

    # Extract from multiple pages - ALL FREE after schema is created
    urls = [
        "https://www.bookdepository.com/category/1000/All",
        "https://www.bookdepository.com/category/1020/Technology-Engineering",
        "https://www.bookdepository.com/category/2633/Computing-Internet",
    ]

    async with AsyncWebCrawler() as crawler:
        for url in urls:
            result = await crawler.arun(url=url, config=config)
            if result.success:
                data = json.loads(result.extracted_content)
                print(f"✅ {url}: Extracted {len(data)} items (FREE)")
                save_file(
                    result.cleaned_html, OUTPUT_DIR / f"extracted_{Path(url).name}.html"
                )
                save_file(data, OUTPUT_DIR / f"extracted_{Path(url).name}.json")


asyncio.run(smart_extraction_workflow())
