import asyncio
import shutil
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from jet.file.utils import save_file

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main(url):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                markdown=True,
                content_filter=PruningContentFilter(threshold=0.4),  # or BM25
            ),
        )
    # cleaned_content = result.markdown   # ← often the best input
    # or result.fit_markdown if you want even more aggressive filtering
    return result


if __name__ == "__main__":
    url = "https://missav.ws/dm13/en/kbkd-604"
    result = asyncio.run(main(url))

    save_file(result.markdown, OUTPUT_DIR / "md_content.json")
    save_file(result.fit_markdown, OUTPUT_DIR / "filtered_md_content.json")
