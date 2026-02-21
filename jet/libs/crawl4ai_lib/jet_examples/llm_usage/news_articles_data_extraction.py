import asyncio
import json

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMExtractionStrategy
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config
from pydantic import BaseModel


class ArticleData(BaseModel):
    headline: str
    summary: str


async def main():
    llm_strategy = LLMExtractionStrategy(
        llm_config=get_llm_config(),
        schema=ArticleData.schema(),
        extraction_type="schema",
        instruction="Extract 'headline' and a short 'summary' from the content.",
    )

    config = CrawlerRunConfig(
        exclude_external_links=True,
        word_count_threshold=20,
        extraction_strategy=llm_strategy,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url="https://news.ycombinator.com", config=config)
        article = json.loads(result.extracted_content)
        print(article)


if __name__ == "__main__":
    asyncio.run(main())
