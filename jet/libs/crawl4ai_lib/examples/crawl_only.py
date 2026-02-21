"""
Basic Adaptive Crawling Example

USAGE EXAMPLES
--------------

Default usage (documentation + async Python terms):
python crawl_only.py

Custom start URL and query (as positional arguments):
python crawl_only.py https://en.wikipedia.org/wiki/Web_crawler "web crawling algorithms"

Shorthand keyword arguments:
python crawl_only.py -u https://arxiv.org -q "transformers deep learning NLP"

Show 10 most relevant results instead of the default 5:
python crawl_only.py -k 10

Combine all options (shorthand and positional mix possible):
python crawl_only.py -u https://docs.python.org/3/library/asyncio.html -q "event loops scheduling" -k 3


DESCRIPTION
-----------

This example demonstrates the simplest use case of adaptive crawling:
finding information about a specific topic and knowing when to stop.
"""

import argparse
import asyncio

from crawl4ai import AdaptiveCrawler, AsyncWebCrawler


def get_args():
    parser = argparse.ArgumentParser(description="Adaptive web crawl example")
    parser.add_argument(
        "start_url",
        nargs="?",
        type=str,
        default="https://docs.python.org/3/library/asyncio.html",
        help="URL to start crawling from (positional or --start-url/-u)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        type=str,
        default="async await context managers coroutines",
        help="Query or keywords for information extraction (positional or --query/-q)",
    )
    parser.add_argument(
        "-u",
        "--start-url",
        dest="start_url_kw",
        type=str,
        help="URL to start crawling from (shorthand for --start_url)",
    )
    parser.add_argument(
        "-q",
        "--query",
        dest="query_kw",
        type=str,
        help="Query or keywords for information extraction (shorthand for positional query)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of most relevant pages to show",
    )
    args = parser.parse_args()

    # Allow keyword args to override positional if given
    if getattr(args, "start_url_kw", None):
        args.start_url = args.start_url_kw
    if getattr(args, "query_kw", None):
        args.query = args.query_kw
    return args


async def main():
    args = get_args()

    # Initialize the crawler
    async with AsyncWebCrawler(verbose=True) as crawler:
        # Create an adaptive crawler with default settings (statistical strategy)
        adaptive = AdaptiveCrawler(crawler)

        # Note: You can also use embedding strategy for semantic understanding:
        # from crawl4ai import AdaptiveConfig
        # config = AdaptiveConfig(strategy="embedding")
        # adaptive = AdaptiveCrawler(crawler, config)

        # Start adaptive crawling
        print(f"Starting adaptive crawl for '{args.query}' from {args.start_url}...")
        result = await adaptive.digest(start_url=args.start_url, query=args.query)

        # Display crawl statistics
        print("\n" + "=" * 50)
        print("CRAWL STATISTICS")
        print("=" * 50)
        adaptive.print_stats(detailed=False)

        # Get the most relevant content found
        print("\n" + "=" * 50)
        print("MOST RELEVANT PAGES")
        print("=" * 50)

        relevant_pages = adaptive.get_relevant_content(top_k=args.top_k)
        for i, page in enumerate(relevant_pages, 1):
            print(f"\n{i}. {page['url']}")
            print(f"   Relevance Score: {page['score']:.2%}")

            # Show a snippet of the content
            content = page["content"] or ""
            if content:
                snippet = content[:200].replace("\n", " ")
                if len(content) > 200:
                    snippet += "..."
                print(f"   Preview: {snippet}")

        # Show final confidence
        print(f"\n{'=' * 50}")
        print(f"Final Confidence: {adaptive.confidence:.2%}")
        print(f"Total Pages Crawled: {len(result.crawled_urls)}")
        print(
            f"Knowledge Base Size: {len(adaptive.state.knowledge_base) if adaptive.state and adaptive.state.knowledge_base is not None else 0} documents"
        )

        # Example: Check if we can answer specific questions
        print(f"\n{'=' * 50}")
        print("INFORMATION SUFFICIENCY CHECK")
        print(f"{'=' * 50}")

        if adaptive.confidence >= 0.8:
            print(
                "✓ High confidence - can answer detailed questions about async Python"
            )
        elif adaptive.confidence >= 0.6:
            print("~ Moderate confidence - can answer basic questions")
        else:
            print("✗ Low confidence - need more information")


if __name__ == "__main__":
    asyncio.run(main())
