# File: async_webcrawler_multiple_urls_example.py
import os
import shutil
import sys
from pathlib import Path

from jet.file.utils import save_file

# append 2 parent directories to sys.path to import crawl4ai
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)

import asyncio

from crawl4ai import AsyncWebCrawler

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main():
    # Initialize the AsyncWebCrawler
    async with AsyncWebCrawler(verbose=True) as crawler:
        # List of URLs to crawl
        urls = [
            "https://nsfwph.org/tags/ass-shaking/",
            "https://nsfwph.org/tags/pinay-solid-twerk/",
            # "https://nsfwph.org/threads/1983646/",
            # "https://nsfwph.org/tags/twerk/page-2",
            # "https://nsfwph.org/threads/2022529/",
            # "https://nsfwph.org/tags/twerk/",
            # "https://nsfwph.org/threads/1983955/",
            # "https://nsfwph.org/threads/2022943/",
            # "https://nsfwph.org/tags/fat-ass/",
            # "https://nsfwph.org/tags/sexy-body/page-2",
        ]

        # Set up crawling parameters
        word_count_threshold = 100

        # Run the crawling process for multiple URLs
        results = await crawler.arun_many(
            urls=urls,
            word_count_threshold=word_count_threshold,
            bypass_cache=True,
            verbose=True,
        )

        # Process the results
        for result in results:
            if result.success:
                print(f"Successfully crawled: {result.url}")
                print(f"Title: {result.metadata.get('title', 'N/A')}")
                print(f"Word count: {len(result.markdown.split())}")
                print(
                    f"Number of links: {len(result.links.get('internal', [])) + len(result.links.get('external', []))}"
                )
                print(f"Number of images: {len(result.media.get('images', []))}")
                print("---")
            else:
                print(f"Failed to crawl: {result.url}")
                print(f"Error: {result.error_message}")
                print("---")

        save_file(results, OUTPUT_DIR / "results.json")


if __name__ == "__main__":
    asyncio.run(main())
