import asyncio
import json

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    JsonCssExtractionStrategy,
)
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config


async def extract_main_articles(url: str):
    schema = {
        "name": "ArticleBlock",
        "baseSelector": "div.article-block",
        "fields": [
            {"name": "headline", "selector": "h2", "type": "text"},
            {"name": "summary", "selector": ".summary", "type": "text"},
            {
                "name": "metadata",
                "type": "nested",
                "fields": [
                    {"name": "author", "selector": ".author", "type": "text"},
                    {"name": "date", "selector": ".date", "type": "text"},
                ],
            },
        ],
    }

    config = CrawlerRunConfig(
        # Keep only #main-content
        css_selector="#main-content",
        # Filtering
        word_count_threshold=10,
        excluded_tags=["nav", "footer"],
        exclude_external_links=True,
        exclude_domains=["somebadsite.com"],
        exclude_external_images=True,
        # Extraction
        extraction_strategy=JsonCssExtractionStrategy(
            schema,
            llm_config=get_llm_config(),
        ),
        cache_mode=CacheMode.BYPASS,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)
        if not result.success:
            print(f"Error: {result.error_message}")
            return None
        return json.loads(result.extracted_content)


async def main():
    articles = await extract_main_articles("https://news.ycombinator.com/newest")
    if articles:
        print("Extracted Articles:", articles[:2])  # Show first 2


if __name__ == "__main__":
    asyncio.run(main())
