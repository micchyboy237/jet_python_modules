import asyncio
import shutil
from pathlib import Path

from crawl4ai import AdaptiveConfig, AdaptiveCrawler, AsyncWebCrawler
from jet.file.utils import save_file
from jet.libs.crawl4ai_lib.adaptive_config import get_adaptive_config

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def crawl_url(
    url: str,
    query: str,
    *,
    top_k: int = 5,
    config: AdaptiveConfig | None = None,
    output_dir: str | Path = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    if config is None:
        config = get_adaptive_config(
            strategy="embedding",
            max_pages=10,
            save_state=True,
            state_path=str(output_dir / "state"),
            embedding_k_exp=4.0,
            n_query_variations=12,
        )
    async with AsyncWebCrawler(verbose=False) as crawler:
        adaptive = AdaptiveCrawler(crawler, config)
        result = await adaptive.digest(start_url=url, query=query)

        print("\n" + "=" * 50)
        print("CRAWL STATISTICS")
        print("=" * 50)
        adaptive.print_stats(detailed=False)

        # Get the most relevant content found
        print("\n" + "=" * 50)
        print("MOST RELEVANT PAGES")
        print("=" * 50)

        relevant_pages = adaptive.get_relevant_content(top_k=top_k)
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

        print(f"\n{'=' * 50}")
        print(f"Pages crawled: {len(result.crawled_urls)}")
        print(f"Final confidence: {adaptive.confidence:.1%}")
        print(f"Stopped reason: {result.metrics.get('stopped_reason', 'max_pages')}")

        if result.metrics.get("is_irrelevant", False):
            print("⚠️  Query detected as irrelevant!")

        save_file(result, output_dir / "result.json")
        save_file(relevant_pages, output_dir / "relevant_pages.json")

        return result, relevant_pages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive crawling with embedding strategy"
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default="https://docs.python.org/3/library/asyncio.html",
        help="The seed URL to crawl (default: https://docs.python.org/3/library/asyncio.html)",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default="async await context managers coroutines",
        help="The search query (default: 'async await context managers coroutines')",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=10,
        help="Number of top relevant pages to return (default: 10)",
    )

    args = parser.parse_args()

    asyncio.run(crawl_url(args.url, args.query, top_k=args.top_k))
