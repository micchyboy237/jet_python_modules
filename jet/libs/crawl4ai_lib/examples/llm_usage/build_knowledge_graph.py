import asyncio

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
)
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config
from pydantic import BaseModel


class Entity(BaseModel):
    name: str
    description: str


class Relationship(BaseModel):
    entity1: Entity
    entity2: Entity
    description: str
    relation_type: str


class KnowledgeGraph(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


async def main():
    # LLM extraction strategy
    llm_strat = LLMExtractionStrategy(
        llmConfig=get_llm_config(),
        schema=KnowledgeGraph.model_json_schema(),
        extraction_type="schema",
        instruction="Extract entities and relationships from the content. Return valid JSON.",
        chunk_token_threshold=1400,
        apply_chunking=True,
        input_format="html",
        extra_args={"temperature": 0.1, "max_tokens": 1500},
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strat, cache_mode=CacheMode.BYPASS
    )

    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        # Example page
        url = "https://www.nbcnews.com/business"
        result = await crawler.arun(url=url, config=crawl_config)

        print("--- LLM RAW RESPONSE ---")
        print(result.extracted_content)
        print("--- END LLM RAW RESPONSE ---")

        if result.success:
            with open("kb_result.json", "w", encoding="utf-8") as f:
                f.write(result.extracted_content)
            llm_strat.show_usage()
        else:
            print("Crawl failed:", result.error_message)


if __name__ == "__main__":
    asyncio.run(main())
