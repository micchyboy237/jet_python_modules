import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Type, TypedDict, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel, Field

from jet.adapters.llama_cpp.config import (
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
    LLM_MODEL,
)
from jet.adapters.llama_cpp.embed_utils import embed as llamacpp_embed
from jet.adapters.llama_cpp.llm_utils import achat
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity
from jet.logger import logger
from jet.search.playwright.playwright_extract import PlaywrightExtract
from jet.search.searxng import async_search_searxng

# Setup tracer consistent with crag_base/react_with_telemetry
tracer = trace.get_tracer(__name__)

try:
    from nltk.tokenize import sent_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")


class PlaywrightSearchInput(BaseModel):
    """Input for PlaywrightSearch"""

    query: str = Field(description="Search query to look up")
    include_domains: Optional[List[str]] = Field(
        default=[],
        description="""A list of domains to restrict search results to.
Use this when:
1. The user explicitly requests information from specific websites (e.g., 'Find climate data from nasa.gov')
2. The user mentions an organization or company without specifying the domain (e.g., 'Find information about iPhones from Apple')
In both cases, determine the appropriate domains (e.g., ['nasa.gov'] or ['apple.com']) and set this parameter.
Results will ONLY come from the specified domains - no other sources will be included.
Default is an empty list (no domain restriction).
""",
    )
    exclude_domains: Optional[List[str]] = Field(
        default=[],
        description="""A list of domains to exclude from search results.
Use this when:
1. The user explicitly requests to avoid certain websites (e.g., 'Find information about climate change but not from twitter.com')
2. The user mentions not wanting results from specific organizations without naming the domain (e.g., 'Find phone reviews but nothing from Apple')
In both cases, determine the appropriate domains to exclude (e.g., ['twitter.com'] or ['apple.com']) and set this parameter.
Results will filter out all content from the specified domains.
Default is an empty list (no domain exclusion).
""",
    )
    search_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Controls search thoroughness and result comprehensiveness.
Use 'basic' for simple queries requiring quick, straightforward answers.
Use 'advanced' for complex queries, specialized topics, rare information, or when in-depth analysis is needed.
Default is 'basic'.
""",
    )
    include_images: Optional[bool] = Field(
        default=True,
        description="""Determines if the search returns relevant images along with text results.
Set to True when the user explicitly requests visuals or when images would significantly enhance understanding (e.g., 'Show me what black holes look like,' 'Find pictures of Renaissance art').
Default is True for PlaywrightSearch to leverage visual content extraction.
""",
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description="""Limits results to content published within a specific timeframe.
ONLY set this when the user explicitly mentions a time period (e.g., 'latest AI news,' 'articles from last week').
For less popular or niche topics, use broader time ranges ('month' or 'year') to ensure sufficient relevant results.
Options: 'day' (24h), 'week' (7d), 'month' (30d), 'year' (365d).
Default is None (no time restriction).
""",
    )
    topic: Optional[Literal["general", "news", "finance"]] = Field(
        default="general",
        description="""Specifies search category for optimized results.
Use 'general' (default) for most queries, INCLUDING those with terms like 'latest,' 'newest,' or 'recent' when referring to general information.
Use 'finance' for markets, investments, economic data, or financial news.
Use 'news' ONLY for politics, sports, or major current events covered by mainstream media - NOT simply because a query asks for 'new' information.
""",
    )
    include_favicon: Optional[bool] = Field(
        default=True,
        description="""Determines whether to include favicon URLs for each search result.
When enabled, each search result will include the website's favicon URL, useful for:
- Building rich UI interfaces with visual website indicators
- Providing visual cues about the source's credibility or brand
- Creating bookmark-like displays with recognizable site icons
Default is True to enhance result presentation.
""",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="""Filters search results to include only content published on or after this date.
Format must be YYYY-MM-DD (e.g., '2024-01-15'). Default is None.""",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="""Filters search results to include only content published on or before this date.
Format must be YYYY-MM-DD (e.g., '2024-03-31'). Default is None.""",
    )
    max_content_length: Optional[int] = Field(
        default=500,
        description="Maximum length of the content field in characters. Default is 500.",
    )


class PlaywrightSearchResult(TypedDict):
    url: str
    title: str
    content: str
    raw_score: float
    score: float
    raw_content: Optional[str]
    images: List[str]
    favicon: Optional[str]


class PlaywrightSearchAPIWrapper(BaseModel):
    """Wrapper for Playwright-based search engine with full observability."""

    searxng_url: str = Field(default=SEARXNG_URL)
    max_results: Optional[int] = Field(default=5)
    include_image_descriptions: Optional[bool] = Field(default=False)
    max_content_length: Optional[int] = Field(default=500)

    def _score_chunks(self, chunks: List[str], query: str) -> List[float]:
        """Score chunks using centralized llama_cpp adapters with tracing."""
        if not chunks or not query:
            return [0.0] * len(chunks)

        with tracer.start_as_current_span(
            "playwright_search.score_chunks",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                "chunk_count": len(chunks),
            },
        ) as span:
            try:
                query_emb = llamacpp_embed(
                    query,
                    model=EMBED_MODEL,
                    prefix=EMBED_QUERY_PREFIX,
                    return_format="numpy",
                )
                chunk_embs = llamacpp_embed(
                    chunks,
                    model=EMBED_MODEL,
                    return_format="numpy",
                    show_progress=False,
                )
                scores = [float(cosine_similarity(query_emb, ce)) for ce in chunk_embs]
                clipped = [max(0.0, min(1.0, s)) for s in scores]

                span.set_attribute("score.min", min(clipped) if clipped else 0.0)
                span.set_attribute("score.max", max(clipped) if clipped else 0.0)
                return clipped
            except Exception as e:
                logger.error(f"Error scoring chunks: {e}")
                span.record_exception(e)
                return [0.0] * len(chunks)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, using NLTK if available, else regex."""
        if NLTK_AVAILABLE:
            return sent_tokenize(text)
        sentence_endings = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_relevant_content(
        self, raw_content: str, query: str, max_length: int
    ) -> str:
        """Extract the most relevant content from raw_content up to max_length."""
        if not raw_content:
            return ""

        chunks = self._split_into_sentences(raw_content)
        if not chunks:
            return ""

        max_chunk_chars = 800
        chunks = [chunk for chunk in chunks if len(chunk) <= max_chunk_chars]

        if not chunks:
            return (
                raw_content[:max_length] + "..."
                if len(raw_content) > max_length
                else raw_content
            )

        scores = self._score_chunks(chunks, query)
        scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

        content = ""
        separator = " [...] "
        selected_chunks = 0
        max_chunks = 3

        for chunk, _ in scored_chunks:
            if selected_chunks >= max_chunks:
                break
            chunk_with_separator = (
                chunk + separator if selected_chunks < max_chunks - 1 else chunk
            )
            if len(content) + len(chunk_with_separator) <= max_length:
                content += chunk_with_separator
                selected_chunks += 1
            else:
                remaining = max_length - len(content)
                if remaining > 10:
                    content += chunk[:remaining].rsplit(" ", 1)[0] + "..."
                break

        content = content.strip()
        if not content:
            content = (
                chunks[0][:max_length] + "..."
                if len(chunks[0]) > max_length
                else chunks[0]
            )
        if content.endswith(separator):
            content = content[: -len(separator)]

        return content

    async def raw_results_async(
        self,
        query: str,
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        search_depth: Optional[Literal["basic", "advanced"]],
        include_images: Optional[bool],
        time_range: Optional[Literal["day", "week", "month", "year"]],
        topic: Optional[Literal["general", "news", "finance"]],
        include_favicon: Optional[bool],
        start_date: Optional[str],
        end_date: Optional[str],
        include_answer: Optional[Union[bool, Literal["basic", "advanced"]]],
        include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]],
        include_image_descriptions: Optional[bool],
        auto_parameters: Optional[bool],
        country: Optional[str],
    ) -> Dict[str, Any]:
        """Fully observable async search pipeline."""
        session_id = trace.format_trace_id(
            trace.get_current_span().get_span_context().trace_id
        )

        with tracer.start_as_current_span(
            "playwright_search.pipeline",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                SpanAttributes.SESSION_ID: session_id,
                SpanAttributes.INPUT_VALUE: query,
                "search.depth": search_depth,
                "search.topic": topic,
            },
        ) as root_span:
            start_time = asyncio.get_event_loop().time()

            # 1. SearXNG Search with Observability
            with tracer.start_as_current_span(
                "playwright_search.searxng",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value
                },
            ) as search_span:
                time_range_map = {"day": 0, "week": 0, "month": 0, "year": 1}
                years_ago = time_range_map.get(time_range, 1) if time_range else 1

                min_date = None
                if start_date:
                    try:
                        min_date = datetime.strptime(start_date, "%Y-%m-%d")
                    except ValueError:
                        raise ToolException(
                            "Invalid start_date format. Use YYYY-MM-DD."
                        )

                if end_date and min_date:
                    try:
                        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                        if end_date_dt < min_date:
                            raise ToolException("end_date cannot be before start_date.")
                    except ValueError:
                        raise ToolException("Invalid end_date format. Use YYYY-MM-DD.")

                topic_map = {
                    "general": ["general"],
                    "news": ["news"],
                    "finance": ["business"],
                }
                categories = topic_map.get(topic, ["general"])

                count = (
                    self.max_results
                    if search_depth == "basic"
                    else self.max_results * 2
                )

                search_results = await async_search_searxng(
                    query=query,
                    query_url=self.searxng_url,
                    count=count,
                    include_sites=include_domains,
                    exclude_sites=exclude_domains,
                    min_date=min_date,
                    categories=categories,
                    years_ago=years_ago,
                )
                search_span.set_attribute("search.result_count", len(search_results))

            if not search_results:
                root_span.set_attribute(SpanAttributes.OUTPUT_VALUE, "No results found")
                return {
                    "query": query,
                    "results": [],
                    "images": [] if include_images else None,
                    "response_time": asyncio.get_event_loop().time() - start_time,
                }

            # 2. Playwright Extraction
            urls = [result["url"] for result in search_results]
            extractor = PlaywrightExtract()
            extract_format = (
                "markdown" if include_raw_content in (True, "markdown") else "text"
            )

            with tracer.start_as_current_span(
                "playwright_search.extract",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value
                },
            ) as extract_span:
                extract_results = await extractor._arun(
                    urls=urls,
                    extract_depth=search_depth,
                    include_images=include_images,
                    include_favicon=include_favicon,
                    format=extract_format,
                )
                extract_span.set_attribute("extract.url_count", len(urls))

            # 3. Scoring & Assembly
            results = []
            search_texts = [result["content"] for result in search_results]
            embed_scores = self._score_chunks(search_texts, query)

            for search_result, extract_result, embed_score in zip(
                search_results, extract_results["results"], embed_scores
            ):
                if "error" in extract_result:
                    continue

                content = (
                    self._extract_relevant_content(
                        extract_result["raw_content"], query, self.max_content_length
                    )
                    if extract_result.get("raw_content")
                    else search_result["content"]
                )

                result_item = {
                    "url": search_result["url"],
                    "title": search_result["title"],
                    "content": content,
                    "raw_score": search_result["score"],
                    "score": embed_score,
                    "raw_content": extract_result["raw_content"]
                    if include_raw_content
                    else None,
                }
                if include_images:
                    result_item["images"] = extract_result.get("images", [])
                if include_favicon:
                    result_item["favicon"] = extract_result.get("favicon")
                results.append(result_item)

            results.sort(key=lambda x: x["score"], reverse=True)

            # 4. Streaming LLM Answer (if requested)
            answer = None
            if include_answer and results:
                with tracer.start_as_current_span(
                    "playwright_search.llm_summary",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                        SpanAttributes.LLM_MODEL_NAME: LLM_MODEL,
                    },
                ) as llm_span:
                    context = "\n\n".join(
                        [
                            f"[{r['title']}]({r['url']})\n{r['content']}"
                            for r in results[:3]
                        ]
                    )
                    prompt = (
                        f"Based on these search results, answer: '{query}'\n\n"
                        f"Results:\n{context}"
                    )

                    # Use achat with streaming and enable_thinking=False
                    res = await achat(
                        prompt_or_messages=prompt,
                        model=LLM_MODEL,
                        enable_thinking=False,
                        extra_body_params={
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                        project_name="playwright-search-obs",
                        capture_content=True,
                    )

                    answer = res.content

                    if res.usage:
                        llm_span.set_attribute(
                            SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                            res.usage.get("prompt_tokens", 0),
                        )
                        llm_span.set_attribute(
                            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                            res.usage.get("completion_tokens", 0),
                        )
                        llm_span.set_attribute(
                            SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                            res.usage.get("total_tokens", 0),
                        )
                    llm_span.set_attribute(
                        SpanAttributes.OUTPUT_VALUE, (answer or "")[:2000]
                    )

            response_time = asyncio.get_event_loop().time() - start_time
            root_span.set_attribute("response_time_sec", round(response_time, 3))
            root_span.set_attribute(
                SpanAttributes.OUTPUT_VALUE,
                json.dumps({"result_count": len(results)})[:1000],
            )

            images = []
            if include_images and include_image_descriptions:
                for result in results:
                    if "images" in result:
                        images.extend(result["images"])

            return {
                "query": query,
                "follow_up_questions": None,
                "answer": answer,
                "images": images if include_images else None,
                "results": results[: self.max_results],
                "response_time": response_time,
            }

    def raw_results(
        self,
        query: str,
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]],
        search_depth: Optional[Literal["basic", "advanced"]],
        include_images: Optional[bool],
        time_range: Optional[Literal["day", "week", "month", "year"]],
        topic: Optional[Literal["general", "news", "finance"]],
        include_favicon: Optional[bool],
        start_date: Optional[str],
        end_date: Optional[str],
        include_answer: Optional[Union[bool, Literal["basic", "advanced"]]],
        include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]],
        include_image_descriptions: Optional[bool],
        auto_parameters: Optional[bool],
        country: Optional[str],
    ) -> Dict[str, Any]:
        return asyncio.run(
            self.raw_results_async(
                query,
                include_domains,
                exclude_domains,
                search_depth,
                include_images,
                time_range,
                topic,
                include_favicon,
                start_date,
                end_date,
                include_answer,
                include_raw_content,
                include_image_descriptions,
                auto_parameters,
                country,
            )
        )


class PlaywrightSearch(BaseTool):
    """Tool that searches the web using Playwright and SearXNG."""

    name: str = "playwright_search"
    description: str = (
        "A search engine using Playwright and SearXNG for comprehensive, accurate results. "
        "Supports advanced search depths, domain management, time range filters, and image search."
    )
    args_schema: Type[BaseModel] = PlaywrightSearchInput
    handle_tool_error: bool = True

    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    search_depth: Optional[Literal["basic", "advanced"]] = None
    include_images: bool = False
    time_range: Optional[Literal["day", "week", "month", "year"]] = None
    topic: Optional[Literal["general", "news", "finance"]] = None
    include_favicon: bool = False
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_results: Optional[int] = None
    include_answer: Optional[Union[bool, Literal["basic", "advanced"]]] = None
    include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]] = "markdown"
    include_image_descriptions: bool = False
    auto_parameters: Optional[bool] = None
    country: Optional[str] = None

    api_wrapper: PlaywrightSearchAPIWrapper = Field(
        default_factory=PlaywrightSearchAPIWrapper
    )

    def _run(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: bool = False,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        include_favicon: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            raw_results = self.api_wrapper.raw_results(
                query=query,
                include_domains=self.include_domains
                if self.include_domains
                else include_domains,
                exclude_domains=self.exclude_domains
                if self.exclude_domains
                else exclude_domains,
                search_depth=self.search_depth if self.search_depth else search_depth,
                include_images=self.include_images
                if self.include_images
                else include_images,
                time_range=self.time_range if self.time_range else time_range,
                topic=self.topic if self.topic else topic,
                include_favicon=self.include_favicon
                if self.include_favicon
                else include_favicon,
                start_date=start_date,
                end_date=end_date,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_image_descriptions=self.include_image_descriptions,
                auto_parameters=self.auto_parameters,
                country=self.country,
            )
            if not raw_results.get("results", []):
                search_params = {
                    "time_range": time_range,
                    "include_domains": include_domains,
                    "search_depth": search_depth,
                    "exclude_domains": exclude_domains,
                    "topic": topic,
                }
                suggestions = self._generate_suggestions(search_params)
                error_message = (
                    f"No search results found for '{query}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: bool = False,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        include_favicon: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                query=query,
                include_domains=self.include_domains
                if self.include_domains
                else include_domains,
                exclude_domains=self.exclude_domains
                if self.exclude_domains
                else exclude_domains,
                search_depth=self.search_depth if self.search_depth else search_depth,
                include_images=self.include_images
                if self.include_images
                else include_images,
                time_range=self.time_range if self.time_range else time_range,
                topic=self.topic if self.topic else topic,
                include_favicon=self.include_favicon
                if self.include_favicon
                else include_favicon,
                start_date=start_date,
                end_date=end_date,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                include_image_descriptions=self.include_image_descriptions,
                auto_parameters=self.auto_parameters,
                country=self.country,
            )
            if not raw_results.get("results", []):
                search_params = {
                    "time_range": time_range,
                    "include_domains": include_domains,
                    "search_depth": search_depth,
                    "exclude_domains": exclude_domains,
                    "topic": topic,
                }
                suggestions = self._generate_suggestions(search_params)
                error_message = (
                    f"No search results found for '{query}'. "
                    f"Suggestions: {', '.join(suggestions)}. "
                )
                raise ToolException(error_message)
            return raw_results
        except ToolException:
            raise
        except Exception as e:
            return {"error": str(e)}

    def _generate_suggestions(self, params: Dict[str, Any]) -> List[str]:
        """Generate helpful suggestions based on the failed search parameters."""
        suggestions = []
        if params.get("time_range"):
            suggestions.append("Remove time_range argument")
        if params.get("include_domains"):
            suggestions.append("Remove include_domains argument")
        if params.get("exclude_domains"):
            suggestions.append("Remove exclude_domains argument")
        if params.get("search_depth") == "basic":
            suggestions.append("Try 'advanced' search_depth")
        if params.get("topic") != "general":
            suggestions.append("Try 'general' topic")
        return suggestions


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Playwright Search with Observability")
    parser.add_argument(
        "query",
        nargs="?",
        default="Latest AI developments 2026",
        help="Search query",
    )
    parser.add_argument(
        "--depth",
        choices=["basic", "advanced"],
        default="basic",
        help="Search depth",
    )
    parser.add_argument(
        "--max-results", type=int, default=5, help="Maximum number of results"
    )
    parser.add_argument(
        "--include-answer",
        action="store_true",
        help="Generate streaming LLM summary",
    )
    parser.add_argument(
        "--searxng-url",
        default=SEARXNG_URL,
        help="SearXNG instance URL",
    )
    parser.add_argument(
        "--topic",
        choices=["general", "news", "finance"],
        default="general",
        help="Search topic category",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    wrapper = PlaywrightSearchAPIWrapper(
        searxng_url=args.searxng_url,
        max_results=args.max_results,
    )

    print(f"🔍 Searching: '{args.query}' (depth={args.depth})")
    if args.include_answer:
        print("💬 Generating streaming answer...\n")

    result = asyncio.run(
        wrapper.raw_results_async(
            query=args.query,
            search_depth=args.depth,
            topic=args.topic,
            include_answer=args.include_answer,
            include_images=True,
            include_favicon=True,
            include_raw_content="markdown",
            include_domains=None,
            exclude_domains=None,
            time_range=None,
            start_date=None,
            end_date=None,
            include_image_descriptions=False,
            auto_parameters=None,
            country=None,
        )
    )

    if not args.include_answer:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(
            f"\n✅ Done in {result['response_time']:.2f}s | "
            f"{len(result['results'])} results"
        )
