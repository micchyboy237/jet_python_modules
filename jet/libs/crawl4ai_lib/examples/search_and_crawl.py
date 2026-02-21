from urllib.parse import urlencode, urljoin

"""
Adaptive Crawler that ALWAYS starts from SearXNG search results

- Requires SEARXNG_URL environment variable (e.g. export SEARXNG_URL="http://searxng.local")
- Takes only a search query as positional argument
- Fetches top --top-seeds results via JSON
- Feeds them into AdaptiveCrawler.digest() sequentially
- Shows top --top-k most relevant pages at the end

USAGE EXAMPLES
--------------

Basic usage (no host restriction):
python search_and_crawl.py "python asyncio best practices"

Custom number of seeds (long or short form) + fewer final results:
python search_and_crawl.py "transformers in deep learning" --top-seeds 8 --top-k 4
python search_and_crawl.py "transformers in deep learning" -n 8 -k 4

Single site restriction:
python search_and_crawl.py "best rust web frameworks 2026" --site arewewebyet.org

Multiple sites (repeated flags):
python search_and_crawl.py "llm fine-tuning guide" \
    --site huggingface.co --site github.com --site towardsdatascience.com

Multiple sites (comma-separated – convenient for one-liners):
python search_and_crawl.py "python web scraping 2026" \
    --site github.com,beautiful-soup.readthedocs.io,scrapy.org -n 12

Combined example:
python search_and_crawl.py "django vs fastapi performance" -k 3 -n 10 --site reddit.com

python search_and_crawl.py "django vs fastapi performance" \
    -k 3 -n 10 --site reddit.com --site news.ycombinator.com
"""

import argparse
import asyncio
import os
from typing import Any, List, Optional

import httpx
from crawl4ai import AdaptiveCrawler, AsyncWebCrawler


async def fetch_seed_urls_from_searxng(
    searxng_base_url: str,
    query: str,
    top_n: Optional[int] = None,
    timeout: float = 12.0,
    **kwargs: Any,
) -> List[str]:
    """
    Query SearXNG instance and return top N (or all) result URLs using JSON format.
    """
    print(f"[SearXNG] Querying: {searxng_base_url}   →   {query!r}")

    params = {
        "q": query,
        "format": "json",
        "pageno": 1,
        "language": kwargs.get("language", "en"),
        "categories": ",".join(kwargs.get("categories", ["general"])),
        # Feel free to add more SearXNG parameters here as needed
        # "time_range": kwargs.get("time_range"),
        # "safesearch": kwargs.get("safesearch", 0),
    }

    query_string = urlencode(params)
    search_path = "search"
    full_url = (
        urljoin(searxng_base_url.rstrip("/") + "/", search_path) + "?" + query_string
    )

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            print(f"[DEBUG] Full SearXNG URL: {full_url}")
            resp = await client.get(full_url)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            valid_urls = [
                r["url"]
                for r in results
                if "url" in r and r.get("url", "").startswith("http")
            ]

            if top_n is not None:
                urls = valid_urls[:top_n]
            else:
                urls = valid_urls

            if not urls:
                print("[WARN] No valid URLs found in search results")

            return urls

        except httpx.HTTPStatusError as e:
            print(
                f"[ERROR] SearXNG HTTP error {e.response.status_code}: {e.response.text[:200]}"
            )
            return []
        except Exception as e:
            print(f"[ERROR] Cannot reach SearXNG: {type(e).__name__}: {e}")
            return []


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive crawl starting from SearXNG search results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "query",
        help="Search query used to find starting pages via SearXNG",
    )

    parser.add_argument(
        "--top-seeds",
        "-n",
        type=int,
        default=None,
        help="Number of top search results to use as starting points "
        "(default: all results from page 1; use -n or --top-seeds to limit)",
    )

    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of most relevant pages to display at the end (default: 5)",
    )

    # Changed to support multiple
    parser.add_argument(
        "-s",
        "--site",
        action="append",
        dest="sites",
        type=str,
        default=None,
        help="Restrict results to one or more sites/domains. "
        "Use multiple times or comma-separated. "
        "Example: -s github.com -s docs.python.org or -s example.com,docs.python.org",
    )

    args = parser.parse_args()
    return args


async def main():
    args = get_args()

    searxng_url = os.getenv("SEARXNG_URL")
    if not searxng_url:
        print("ERROR: SEARXNG_URL environment variable is not set.")
        print('Please set it, e.g.: export SEARXNG_URL="http://searxng.local"')
        print("Exiting.")
        return

    searxng_url = searxng_url.rstrip("/")

    print("=" * 70)
    print(" CONFIGURATION")
    print("=" * 70)
    print(f"  SearXNG instance : {searxng_url}")
    print(f"  Query            : {args.query}")

    # Site filtering
    sites: List[str] = []
    if args.sites:
        for h in args.sites:
            if "," in h:
                sites.extend(part.strip() for part in h.split(",") if part.strip())
            else:
                sites.append(h.strip())

    # Normalize domains (remove protocol, www., trailing slash)
    normalized_sites = []
    for h in sites:
        h = h.lower().strip()
        if h.startswith(("http://", "https://")):
            h = h.split("://", 1)[-1]
        h = h.removeprefix("www.").rstrip("/")
        if h and "." in h:  # basic sanity
            normalized_sites.append(h)

    if normalized_sites:
        site_parts = [f"site:{domain}" for domain in normalized_sites]
        site_clause = " OR ".join(site_parts)
        effective_query = f"{args.query} {site_clause}"
        print(f"  Site filter      : {' OR '.join(normalized_sites)}")
    else:
        effective_query = args.query
        print("  Site filter      : none")

    print(
        f"  Seed URLs to fetch : "
        f"{args.top_seeds if args.top_seeds is not None else 'all available (page 1)'}"
    )
    print(f"  Top relevant to show : {args.top_k}")
    print("=" * 70 + "\n")

    seed_urls = await fetch_seed_urls_from_searxng(
        searxng_url,
        effective_query,
        top_n=args.top_seeds,
        # kwargs for fetch_seed_urls_from_searxng (expand as needed)
        # language="en",
        # categories=["general"],
    )

    if not seed_urls:
        print("\n❌ No seed URLs retrieved from search. Cannot continue.")
        return

    print(f"\nFound {len(seed_urls)} starting URL(s):")
    for i, url in enumerate(seed_urls, 1):
        print(f"  {i:2d}. {url}")

    print("\nStarting adaptive crawling...\n")

    async with AsyncWebCrawler(verbose=True) as crawler:
        adaptive = AdaptiveCrawler(crawler)

        for idx, url in enumerate(seed_urls, 1):
            print(f"[{idx}/{len(seed_urls)}] Digesting → {url}")
            await adaptive.digest(start_url=url, query=args.query)

        # ────────────────────────────────────────────────
        # Results
        # ────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("CRAWL STATISTICS")
        print("=" * 70)
        adaptive.print_stats(detailed=False)

        print("\n" + "=" * 70)
        print("MOST RELEVANT PAGES")
        print("=" * 70)

        relevant_pages = adaptive.get_relevant_content(top_k=args.top_k)

        for i, page in enumerate(relevant_pages, 1):
            print(f"\n{i}. {page['url']}")
            print(f"   Relevance : {page['score']:.2%}")

            snippet = (page.get("content") or "").replace("\n", " ")[:220]
            if len(snippet) == 220:
                snippet += "…"
            print(f"   Preview   : {snippet}")

        print("\n" + "=" * 70)
        print(f"Final Confidence      : {adaptive.confidence:.2%}")
        print(
            f"Total Pages Crawled   : {len(adaptive.state.crawled_urls) if adaptive.state else 0}"
        )
        kb_size = (
            len(adaptive.state.knowledge_base)
            if adaptive.state and adaptive.state.knowledge_base
            else 0
        )
        print(f"Knowledge Base Size   : {kb_size} documents")

        print("\n" + "=" * 70)
        print("SUFFICIENCY ASSESSMENT")
        print("=" * 70)

        if adaptive.confidence >= 0.80:
            print("✓ High confidence — good for detailed answers")
        elif adaptive.confidence >= 0.60:
            print("~ Moderate confidence — suitable for overview")
        else:
            print("✗ Low confidence — consider broader query or more seeds")


if __name__ == "__main__":
    asyncio.run(main())
