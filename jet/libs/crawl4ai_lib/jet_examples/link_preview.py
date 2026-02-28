from __future__ import annotations

import argparse
import asyncio
from typing import Any, Dict, List, Optional, TypedDict

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LinkPreviewConfig

# ==============================
# Typed Structures
# ==============================


class LinkHeadData(TypedDict, total=False):
    title: str
    description: str
    meta: Dict[str, Any]


class LinkResult(TypedDict, total=False):
    href: str
    text: Optional[str]
    title: Optional[str]
    base_domain: Optional[str]
    head_data: Optional[LinkHeadData]
    intrinsic_score: Optional[float]
    contextual_score: Optional[float]
    total_score: Optional[float]


class CrawlLinksOutput(TypedDict):
    internal: List[LinkResult]
    external: List[LinkResult]


# ==============================
# Config Builder
# ==============================


def build_crawler_config(
    *,
    include_internal: bool,
    include_external: bool,
    max_links: int,
    concurrency: int,
    query: Optional[str],
    score_threshold: float,
    verbose: bool,
) -> CrawlerRunConfig:
    """
    Build a reusable Crawl4AI configuration object.
    """
    return CrawlerRunConfig(
        score_links=True,
        link_preview_config=LinkPreviewConfig(
            include_internal=include_internal,
            include_external=include_external,
            max_links=max_links,
            concurrency=concurrency,
            query=query,
            score_threshold=score_threshold,
            verbose=verbose,
        ),
    )


# ==============================
# Core Reusable Function
# ==============================


async def crawl_links_with_context(
    url: str,
    *,
    include_internal: bool = True,
    include_external: bool = False,
    max_links: int = 15,
    concurrency: int = 5,
    query: Optional[str] = None,
    score_threshold: float = 0.0,
    verbose: bool = False,
) -> CrawlLinksOutput:
    """
    Crawl a URL and return enriched link context results.
    """

    config = build_crawler_config(
        include_internal=include_internal,
        include_external=include_external,
        max_links=max_links,
        concurrency=concurrency,
        query=query,
        score_threshold=score_threshold,
        verbose=verbose,
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)

    internal_links: List[LinkResult] = [
        _normalize_link(link) for link in result.links.get("internal", [])
    ]

    external_links: List[LinkResult] = [
        _normalize_link(link) for link in result.links.get("external", [])
    ]

    return {
        "internal": internal_links,
        "external": external_links,
    }


# ==============================
# Normalization Helper
# ==============================


def _normalize_link(raw: Dict[str, Any]) -> LinkResult:
    """
    Normalize raw Crawl4AI link dict into a strictly typed structure.
    """
    return {
        "href": raw.get("href", ""),
        "text": raw.get("text"),
        "title": raw.get("title"),
        "base_domain": raw.get("base_domain"),
        "head_data": raw.get("head_data"),
        "intrinsic_score": raw.get("intrinsic_score"),
        "contextual_score": raw.get("contextual_score"),
        "total_score": raw.get("total_score"),
    }


# ==============================
# CLI Argument Parser
# ==============================


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl a URL and extract enriched link context using Crawl4AI."
    )

    parser.add_argument("url", type=str, help="Target URL to crawl")
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default=None,
        help="Keyword or query filter for link context",
    )

    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # Hide -q/--query because it's now positional
    )
    parser.add_argument(
        "-m",
        "--max-links",
        type=int,
        default=15,
        help="Maximum number of links to extract",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=5, help="Number of concurrent requests"
    )
    parser.add_argument(
        "-s",
        "--score-threshold",
        type=float,
        default=0.0,
        help="Score threshold for filtering links",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output (default: False)",
    )
    parser.add_argument(
        "-i",
        "--include-internal",
        action="store_true",
        default=True,
        help="Include internal links (default: True)",
    )
    parser.add_argument(
        "-x",
        "--include-external",
        action="store_true",
        default=False,
        help="Include external links (default: False)",
    )

    return parser.parse_args()


# ==============================
# Main Entrypoint
# ==============================


async def _main_async() -> None:
    args = get_args()

    output = await crawl_links_with_context(
        url=args.url,
        include_internal=args.include_internal,
        include_external=args.include_external,
        max_links=args.max_links,
        concurrency=args.concurrency,
        query=args.query,
        score_threshold=args.score_threshold,
        verbose=args.verbose,
    )

    for category in ("internal", "external"):
        print(f"\n=== {category.upper()} LINKS ===")

        for link in output[category]:
            print(f"URL: {link['href']}")
            print(f"Score: {link.get('total_score')}")
            print(f"Head Data: {link.get('head_data')}")
            print("-" * 50)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
