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

# Generate a schema (one-time cost)
html = "<div class='product'><h2>Gaming Laptop</h2><span class='price'>$999.99</span></div>"

# Using OpenAI (requires API token)
schema = JsonCssExtractionStrategy.generate_schema(
    html, llm_config=get_llm_config(strategy="llm")
)

# # Or using Ollama (open source, no token needed)
# schema = JsonCssExtractionStrategy.generate_schema(
#     html,
#     llm_config = LLMConfig(provider="ollama/llama3.3", api_token=None)  # Not needed for Ollama
# )

# Use the schema for fast, repeated extractions
strategy = JsonCssExtractionStrategy(schema)

save_file(schema, OUTPUT_DIR / "schema.json")


async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="raw://" + html,
            config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                extraction_strategy=JsonCssExtractionStrategy(schema),
            ),
        )
        # The JSON output is stored in 'extracted_content'
        data = json.loads(result.extracted_content)
        print("Extracted content:")
        print(data)

        save_file(data, OUTPUT_DIR / "extracted_content.json")


if __name__ == "__main__":
    asyncio.run(main())
