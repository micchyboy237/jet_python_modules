import argparse
import asyncio
import uuid

from agent import LiveRAGSearchAgent
from config import SafetyLimits
from jet.adapters.llama_cpp.config import LLM_MODEL, PHOENIX_REST_API
from jet.logger import logger
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import HTTPSpanExporter
from providers.llm import (
    LlamaCppAnswerGenerator,
    LlamaCppFactExtractor,
    LlamaCppInnerLinkFilter,
    LlamaCppSufficiencyEvaluator,
)
from providers.scraper import HttpxScraperProvider, PlaywrightScraperProvider
from providers.search import SearXNGSearchProvider

PROJECT_NAME = "live-rag-search-local"


def setup_telemetry():
    """Initialize OpenTelemetry + Phoenix tracing."""
    resource = Resource.create({"openinference.project.name": PROJECT_NAME})
    provider = TracerProvider(resource=resource)
    exporter = HTTPSpanExporter(endpoint=f"{PHOENIX_REST_API}/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info(f"📡 Telemetry initialized: {PHOENIX_REST_API}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live RAG Search with Accumulated Memory & Local LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument(
        "--model",
        "-m",
        default=LLM_MODEL,
        help=f"LLM model (default: {LLM_MODEL})",
    )
    parser.add_argument("--max-top-results", type=int, default=10)
    parser.add_argument("--max-inner-links", type=int, default=5)
    parser.add_argument("--max-scrapes", type=int, default=30)
    parser.add_argument("--max-memory-facts", type=int, default=500)
    parser.add_argument("--scrape-timeout", type=float, default=10.0)
    parser.add_argument("--searxng-url", default=None, help="SearXNG base URL override")
    parser.add_argument(
        "--use-playwright",
        action="store_true",
        help="Use Playwright for JS rendering (slower but more robust)",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    setup_telemetry()
    tracer = trace.get_tracer(__name__)
    session_id = str(uuid.uuid4())

    limits = SafetyLimits(
        MAX_TOP_LEVEL_RESULTS=args.max_top_results,
        MAX_INNER_LINKS_PER_PAGE=args.max_inner_links,
        MAX_TOTAL_SCRAPES=args.max_scrapes,
        MAX_MEMORY_FACTS=args.max_memory_facts,
        SCRAPE_TIMEOUT_SEC=args.scrape_timeout,
    )

    llm_kwargs = {"temperature": 0.1, "max_tokens": 4096}

    # Select scraper
    if args.use_playwright:
        scraper = PlaywrightScraperProvider()
        logger.info("Using Playwright Scraper (JS Rendering Enabled)")
    else:
        scraper = HttpxScraperProvider()
        logger.info("Using Httpx Scraper (Fast, Static Only)")

    agent = LiveRAGSearchAgent(
        query=args.query,
        search_provider=SearXNGSearchProvider(base_url=args.searxng_url),
        scraper_provider=scraper,
        evaluator=LlamaCppSufficiencyEvaluator(model=args.model, **llm_kwargs),
        extractor=LlamaCppFactExtractor(model=args.model, **llm_kwargs),
        link_filter=LlamaCppInnerLinkFilter(model=args.model, **llm_kwargs),
        generator=LlamaCppAnswerGenerator(model=args.model, **llm_kwargs),
        limits=limits,
    )

    with tracer.start_as_current_span(
        "live_rag.session",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            SpanAttributes.SESSION_ID: session_id,
            SpanAttributes.INPUT_VALUE: args.query,
            "live_rag.model": args.model,
            "live_rag.limits": str(limits),
        },
    ) as root_span:
        logger.info(f"🚀 Starting Live RAG Search: {args.query}")
        answer = await agent.run()
        root_span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer[:3000])

        phoenix_host = PHOENIX_REST_API.rstrip("/")
        if phoenix_host.endswith("/v1"):
            phoenix_host = phoenix_host[:-3]
        trace_id_hex = format(root_span.get_span_context().trace_id, "032x")
        trace_url = f"{phoenix_host}/redirects/traces/{trace_id_hex}"
        print(f"\n🔗 Trace: {trace_url}")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
