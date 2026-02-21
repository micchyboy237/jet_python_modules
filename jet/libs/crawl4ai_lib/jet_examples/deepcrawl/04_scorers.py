import argparse
import asyncio
import shutil
from pathlib import Path
from typing import List

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from jet.file.utils import save_file
from rich.console import Console
from rich.table import Table

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

console = Console()


async def crawl_with_scorer(url: str, keywords: List[str], max_depth: int = 1):
    scorer = KeywordRelevanceScorer(keywords=keywords, weight=1.0)

    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=max_depth,
            include_external=False,
            url_scorer=scorer,
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        cache_mode="BYPASS",
        verbose=True,
        stream=True,
    )

    async with AsyncWebCrawler() as crawler:
        results = []
        async for result in await crawler.arun(url=url, config=config):
            results.append(result)
            score = result.metadata.get("score", 0.0)
            console.print(
                f"[dim]{len(results):3d}[/] [cyan]{score:5.2f}[/] {result.url}"
            )
        return results


def print_score_stats(results):
    if not results:
        return
    scores = [r.metadata.get("score", 0) for r in results]
    table = Table(title="Score Statistics")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Count", str(len(scores)))
    table.add_row("Average", f"{sum(scores) / len(scores):.2f}")
    table.add_row("Max", f"{max(scores):.2f}")
    table.add_row("Min", f"{min(scores):.2f}")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Best-first crawling with keyword-based scoring"
    )
    parser.add_argument("url", nargs="?", default="https://docs.crawl4ai.com")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument(
        "--keywords",
        "-k",
        type=str,
        default=None,
        help="Comma-separated keywords for scoring (example: crawl,async,config)",
    )
    args = parser.parse_args()

    DEFAULT_KEYWORDS = [
        "crawl",
        "async",
        "configuration",
        "javascript",
        "example",
        "deep",
    ]
    active_keywords = [
        k.strip() for k in (args.keywords or "").split(",") if k.strip()
    ] or DEFAULT_KEYWORDS

    console.rule("Best-First Crawling with Keyword Scoring")
    console.print(f"Active keywords: [yellow]{', '.join(active_keywords)}[/yellow]")

    results = asyncio.run(crawl_with_scorer(args.url, active_keywords, args.max_depth))

    print_score_stats(results)
    save_file(
        [{"url": r.url, "score": r.metadata.get("score", 0)} for r in results],
        OUTPUT_DIR / "scored_pages.json",
    )
    console.print(f"\n[green]Saved → {OUTPUT_DIR}[/]")
