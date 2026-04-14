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

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def extract_main_articles(url: str):
    schema = {
        "name": "HNStory",
        "baseSelector": "tr.athing",  # Each story starts with this row
        "fields": [
            {
                "name": "id",
                "selector": "",  # Use the row itself
                "type": "attribute",
                "attribute": "id",
            },
            {"name": "rank", "selector": "td.title .rank", "type": "text"},
            {"name": "title", "selector": "span.titleline > a", "type": "text"},
            {
                "name": "link",
                "selector": "span.titleline > a",
                "type": "attribute",
                "attribute": "href",
            },
            {"name": "domain", "selector": "span.sitestr", "type": "text"},
            {
                "name": "metadata",
                "type": "nested",
                "fields": [
                    {"name": "score", "selector": "td.subtext .score", "type": "text"},
                    {
                        "name": "author",
                        "selector": "td.subtext .hnuser",
                        "type": "text",
                    },
                    {"name": "age", "selector": "td.subtext .age a", "type": "text"},
                    {
                        "name": "comments",
                        "selector": "td.subtext a:last-child",
                        "type": "text",
                    },
                ],
            },
        ],
    }

    config = CrawlerRunConfig(
        # Optional: limit to first N stories if desired (e.g. first 30)
        # css_selector="tr.athing:nth-child(-n+30)",
        word_count_threshold=5,  # HN titles are short
        excluded_tags=["nav", "footer", "script", "style"],
        exclude_external_links=False,  # We want the story links
        exclude_domains=[],  # Remove if you had bad domains
        exclude_external_images=True,
        extraction_strategy=JsonCssExtractionStrategy(
            schema,
            # llm_config=get_llm_config(),   # only needed if you use LLM fallback; remove for pure CSS
        ),
        cache_mode=CacheMode.BYPASS,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config)

        if not result.success:
            print(f"Error: {result.error_message}")
            if hasattr(result, "html") and result.html:
                print("HTML length:", len(result.html))
            return None

        try:
            articles = json.loads(result.extracted_content)
            return articles
        except json.JSONDecodeError:
            print("Failed to parse extracted_content as JSON")
            print("Raw extracted_content:", result.extracted_content[:500])
            return None


async def main():
    articles = await extract_main_articles("https://news.ycombinator.com/newest")
    if articles:
        print(f"Successfully extracted {len(articles)} articles")
        print("\nFirst 2 articles:")
        print(json.dumps(articles[:2], indent=2, ensure_ascii=False))

    save_file(articles, OUTPUT_DIR / "articles.json")


if __name__ == "__main__":
    asyncio.run(main())
