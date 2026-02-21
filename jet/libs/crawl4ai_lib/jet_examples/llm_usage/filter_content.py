from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import LLMContentFilter
from jet.libs.crawl4ai_lib.adaptive_config import get_llm_config


async def main():
    # Initialize LLM filter with specific instruction
    filter = LLMContentFilter(
        llm_config=get_llm_config(),
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
        chunk_token_threshold=4096,  # Adjust based on your needs
        verbose=True,
    )
    md_generator = DefaultMarkdownGenerator(
        content_filter=filter, options={"ignore_links": True}
    )
    config = CrawlerRunConfig(
        markdown_generator=md_generator,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun("https://example.com", config=config)
        print(result.markdown.fit_markdown)  # Filtered markdown content
