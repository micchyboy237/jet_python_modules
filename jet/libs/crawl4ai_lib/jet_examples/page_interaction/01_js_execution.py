import asyncio
import shutil
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from jet.file.utils import save_file

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main():
    # Single JS command
    config = CrawlerRunConfig(js_code="window.scrollTo(0, document.body.scrollHeight);")

    async with AsyncWebCrawler() as crawler:
        result_single_command = await crawler.arun(
            url="https://news.ycombinator.com",  # Example site
            config=config,
        )
        print("Crawled length:", len(result_single_command.cleaned_html))

        save_file(
            result_single_command.cleaned_html,
            OUTPUT_DIR / "result_single_command.html",
        )

    # Multiple commands
    js_commands = [
        "window.scrollTo(0, document.body.scrollHeight);",
        # 'More' link on Hacker News
        "document.querySelector('a.morelink')?.click();",
    ]
    config = CrawlerRunConfig(js_code=js_commands)

    async with AsyncWebCrawler() as crawler:
        result_multi_commands = await crawler.arun(
            url="https://news.ycombinator.com",  # Another pass
            config=config,
        )
        print("After scroll+click, length:", len(result_multi_commands.cleaned_html))

        save_file(
            result_multi_commands.cleaned_html,
            OUTPUT_DIR / "result_multi_commands.html",
        )


if __name__ == "__main__":
    asyncio.run(main())
