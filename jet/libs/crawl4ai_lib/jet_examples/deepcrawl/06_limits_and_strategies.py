import argparse
import asyncio
import shutil
from pathlib import Path
from typing import Any, Dict, List

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import (
    BestFirstCrawlingStrategy,
    BFSDeepCrawlStrategy,
    DFSDeepCrawlStrategy,
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from jet.file.utils import save_file
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

console = Console()


async def run_strategy(name: str, strategy: Any, url: str, **kwargs) -> Dict:
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=False,
        cache_mode="BYPASS",
        **kwargs.get("config_extra", {}),
    )

    start = asyncio.get_event_loop().time()
    async with AsyncWebCrawler() as crawler:
        if kwargs.get("stream", False):
            results = []
            async for r in await crawler.arun(url=url, config=config):
                results.append(r)
        else:
            results = await crawler.arun(url=url, config=config)

    duration = asyncio.get_event_loop().time() - start

    return {
        "name": name,
        "count": len(results),
        "duration": duration,
        "urls": [r.url for r in results],
        "depths": [r.metadata.get("depth", 0) for r in results],
        "scores": [
            r.metadata.get("score", None) for r in results if "score" in r.metadata
        ],
    }


def parse_keywords(value: str | None, fallback: List[str]) -> List[str]:
    """Parse comma-separated keywords from CLI argument."""
    if not value:
        return fallback
    cleaned = [k.strip() for k in value.split(",") if k.strip()]
    return cleaned if cleaned else fallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare deep crawling strategies with limits"
    )
    parser.add_argument("url", nargs="?", default="https://docs.crawl4ai.com")
    parser.add_argument(
        "--max-pages", type=int, default=50, help="Base max_pages limit"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth for deep crawling strategies (default: 3)",
    )
    parser.add_argument(
        "--keywords",
        "-k",
        type=str,
        default=None,
        help="Comma-separated keywords for scoring (e.g. crawl,async,config)",
    )
    args = parser.parse_args()

    DEFAULT_KEYWORDS = ["crawl", "deep", "strategy", "filter", "async"]
    active_keywords = parse_keywords(args.keywords, DEFAULT_KEYWORDS)

    console.rule("Keyword Relevance Scorer")
    console.print(f"Active keywords: [yellow]{', '.join(active_keywords)}[/yellow]")

    scorer = KeywordRelevanceScorer(keywords=active_keywords, weight=1.0)

    cases = [
        (
            "BFS + max_pages",
            BFSDeepCrawlStrategy(
                max_depth=args.max_depth, max_pages=args.max_pages, url_scorer=scorer
            ),
            {},  # no streaming / extra config needed
        ),
        (
            "DFS + threshold",
            DFSDeepCrawlStrategy(
                max_depth=args.max_depth,
                score_threshold=0.6,
                max_pages=args.max_pages * 2,
                url_scorer=scorer,
            ),
            {},  # no streaming / extra config needed
        ),
        (
            "Best-First + limit",
            BestFirstCrawlingStrategy(
                max_depth=args.max_depth,
                max_pages=args.max_pages + 3,
                url_scorer=scorer,
            ),
            {"stream": True},
        ),
    ]

    results = []
    for name, strat, extra in cases:
        extra = extra or {}
        console.rule(name)
        res = asyncio.run(
            run_strategy(
                name,
                strat,
                args.url,
                config_extra=extra,
                stream=extra.get("stream", False),
            )
        )
        results.append(res)

        console.print(f"[green]→ {res['count']} pages in {res['duration']:.2f}s[/]")
        if res.get("scores"):
            avg = sum(s for s in res["scores"] if s is not None) / len(
                [s for s in res["scores"] if s is not None]
            )
            console.print(f"Avg score: {avg:.2f}")

    save_file(results, OUTPUT_DIR / "comparison.json")
    console.print(
        f"\n[bold green]Comparison saved → {OUTPUT_DIR / 'comparison.json'}[/]"
    )
