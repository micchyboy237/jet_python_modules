from abc import ABC, abstractmethod
from typing import List

from jet.search.searxng import async_search_searxng
from models import SearchResult
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, num_results: int = 20) -> List[SearchResult]: ...


class SearXNGSearchProvider(SearchProvider):
    """Async search provider using local/self-hosted SearXNG via jet.search."""

    def __init__(self, base_url: str | None = None, **kwargs):
        self.base_url = base_url
        self.kwargs = kwargs

    async def search(self, query: str, num_results: int = 20) -> List[SearchResult]:
        with tracer.start_as_current_span(
            "live_rag.tool.searxng_search",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                SpanAttributes.TOOL_NAME: "searxng_search",
                SpanAttributes.INPUT_VALUE: query,
                "search.num_results_requested": num_results,
            },
        ) as span:
            raw_results = await async_search_searxng(
                query=query,
                query_url=self.base_url,
                count=num_results,
                **self.kwargs,
            )

            search_results = [
                SearchResult(
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    score=float(r.get("score", 0.0)),
                    title=r.get("title", ""),
                )
                for r in raw_results
            ]

            span.set_attribute("search.result_count", len(search_results))
            span.set_attribute(
                SpanAttributes.OUTPUT_VALUE,
                f"Found {len(search_results)} results for '{query}'",
            )
            return search_results
