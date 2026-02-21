import argparse
import asyncio
import shutil
import time
from pathlib import Path
from typing import List

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from jet.file.utils import save_file
from rich.console import Console
from rich.table import Table

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


async def crawl_with_mode(url: str, stream: bool, max_depth: int = 1) -> List:
    config = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=max_depth, include_external=False
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=False,
        stream=stream,
    )

    async with AsyncWebCrawler() as crawler:
        start = time.perf_counter()
        results = []

        if stream:
            first_time = None
            async for result in await crawler.arun(url=url, config=config):
                results.append(result)
                if len(results) == 1:
                    first_time = time.perf_counter() - start
                if len(results) % 5 == 0:
                    console.print(f"  → #{len(results):3d} | {result.url}")
            total_time = time.perf_counter() - start
            return results, first_time, total_time
        else:
            results = await crawler.arun(url=url, config=config)
            total_time = time.perf_counter() - start
            return results, None, total_time


def print_summary(mode: str, count: int, first_time: float | None, total_time: float):
    table = Table(title=f"{mode} Mode Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Pages crawled", f"{count}")
    if first_time is not None:
        table.add_row("Time to first result", f"{first_time:.2f} s")
    table.add_row("Total time", f"{total_time:.2f} s")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare streaming vs non-streaming deep crawl"
    )
    parser.add_argument("url", nargs="?", default="https://docs.crawl4ai.com")
    parser.add_argument("--max-depth", type=int, default=1)
    args = parser.parse_args()

    console.rule("NON-STREAMING")
    results_non, _, time_non = asyncio.run(
        crawl_with_mode(args.url, False, args.max_depth)
    )
    print_summary("Non-stream", len(results_non), None, time_non)

    console.rule("STREAMING")
    results_stream, first_t, time_stream = asyncio.run(
        crawl_with_mode(args.url, True, args.max_depth)
    )
    print_summary("Streaming", len(results_stream), first_t, time_stream)

    save_file(
        {"non_stream": len(results_non), "stream": len(results_stream)},
        OUTPUT_DIR / "counts.json",
    )
    console.print(f"\n[green]Results saved → {OUTPUT_DIR}[/]")
