import argparse
import asyncio
import os
import sys

from agent import LiveRAGSearchAgent
from config import SafetyLimits
from providers.llm import (
    OpenAIAnswerGenerator,
    OpenAIFactExtractor,
    OpenAIInnerLinkFilter,
    OpenAISufficiencyEvaluator,
)
from providers.scraper import HttpxScraperProvider
from providers.search import SerpAPISearchProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live RAG Search with Accumulated Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python main.py --query "top 20 ongoing isekai anime with episodes and release dates"
  python main.py --query "..." --max-scrapes 50 --max-inner-links 3
        """,
    )
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument(
        "--max-top-results",
        type=int,
        default=10,
        help="Max top-level search results to process",
    )
    parser.add_argument(
        "--max-inner-links", type=int, default=5, help="Max inner links per page"
    )
    parser.add_argument(
        "--max-scrapes", type=int, default=30, help="Total scrape budget"
    )
    parser.add_argument(
        "--max-memory-facts",
        type=int,
        default=500,
        help="Max facts in accumulated memory",
    )
    parser.add_argument(
        "--scrape-timeout",
        type=float,
        default=10.0,
        help="Per-page scrape timeout (seconds)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for evaluation/extraction",
    )
    parser.add_argument(
        "--answer-model",
        type=str,
        default="gpt-4o",
        help="LLM model for final answer generation",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    search_api_key = os.environ.get("SERPAPI_KEY")

    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    if not search_api_key:
        print("ERROR: SERPAPI_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    limits = SafetyLimits(
        MAX_TOP_LEVEL_RESULTS=args.max_top_results,
        MAX_INNER_LINKS_PER_PAGE=args.max_inner_links,
        MAX_TOTAL_SCRAPES=args.max_scrapes,
        MAX_MEMORY_FACTS=args.max_memory_facts,
        SCRAPE_TIMEOUT_SEC=args.scrape_timeout,
    )

    agent = LiveRAGSearchAgent(
        query=args.query,
        search_provider=SerpAPISearchProvider(api_key=search_api_key),
        scraper_provider=HttpxScraperProvider(),
        evaluator=OpenAISufficiencyEvaluator(api_key=api_key, model=args.llm_model),
        extractor=OpenAIFactExtractor(api_key=api_key, model=args.llm_model),
        link_filter=OpenAIInnerLinkFilter(api_key=api_key, model=args.llm_model),
        generator=OpenAIAnswerGenerator(api_key=api_key, model=args.answer_model),
        limits=limits,
    )

    answer = await agent.run()
    print(answer)


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
