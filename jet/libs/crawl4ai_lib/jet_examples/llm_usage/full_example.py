import asyncio

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    JsonCssExtractionStrategy,
    LLMContentFilter,
)
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config


async def main():
    # 1) Browser config: headless, bigger viewport, no proxy
    browser_conf = BrowserConfig(
        headless=True,
        viewport_width=1280,
        viewport_height=720
    )

    # 2) Example extraction strategy
    schema = {
        "name": "Articles",
        "baseSelector": "div.article",
        "fields": [
            {"name": "title", "selector": "h2", "type": "text"},
            {"name": "link", "selector": "a", "type": "attribute", "attribute": "href"}
        ]
    }
    extraction = JsonCssExtractionStrategy(schema)

    # 3) Example LLM content filtering

    llm_config = get_llm_config()

    # Initialize LLM filter with specific instruction
    filter = LLMContentFilter(
        llm_config=llm_config,  # or your preferred provider
        instruction="""
        Focus on extracting the core educational content.
        Include:
        - Key concepts and explanations
        - Important code examples
        - Essential technical details
        Exclude:
        - Navigation elements
        - Sidebars
        - Footer content
        Format the output as clean markdown with proper code blocks and headers.
        """,
        chunk_token_threshold=500,  # Adjust based on your needs
        verbose=True
    )

    md_generator = DefaultMarkdownGenerator(
        content_filter=filter,
        options={"ignore_links": True}
    )

    # 4) Crawler run config: skip cache, use extraction
    run_conf = CrawlerRunConfig(
        markdown_generator=md_generator,
        extraction_strategy=extraction,
        cache_mode=CacheMode.BYPASS,
    )

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        # 4) Execute the crawl
        result = await crawler.arun(url="https://example.com/news", config=run_conf)

        if result.success:
            print("Extracted content:", result.extracted_content)
        else:
            print("Error:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())