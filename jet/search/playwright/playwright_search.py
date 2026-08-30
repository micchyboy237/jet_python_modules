import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Type, TypedDict, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from jet.adapters.llama_cpp.config import EMBED_MODEL, EMBED_QUERY_PREFIX
from jet.adapters.llama_cpp.embed_utils import embed as llamacpp_embed
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity
from jet.logger import logger
from jet.search.playwright.playwright_extract import PlaywrightExtract
from jet.search.searxng import search_searxng
from jet.transformers.formatters import format_json

try:
    from nltk.tokenize import sent_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


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
    """Wrapper for Playwright-based search engine."""

    searxng_url: str = Field(default="http://jethros-macbook-air.local:8888")
    max_results: Optional[int] = Field(default=5)
    include_image_descriptions: Optional[bool] = Field(default=False)
    max_content_length: Optional[int] = Field(default=500)

    def _score_chunks(self, chunks: List[str], query: str) -> List[float]:
        """Score chunks using centralized llama_cpp adapters.

        Uses embed_utils for batched/deduped embedding and scoring_utils
        for standardized cosine similarity calculation.
        """
        if not chunks or not query:
            return [0.0] * len(chunks)

        try:
            logger.debug(f"Scoring {len(chunks)} chunks for query: '{query[:50]}...'")

            # Use centralized embed utility (handles batching, dedup, remote optimization)
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

            # Use standardized scoring utility instead of manual numpy linalg
            scores = [float(cosine_similarity(query_emb, ce)) for ce in chunk_embs]

            # Clip to 0-1 range for consistency
            clipped_scores = [max(0.0, min(1.0, s)) for s in scores]

            logger.debug(
                f"Chunk scores computed: min={min(clipped_scores):.3f}, max={max(clipped_scores):.3f}"
            )
            return clipped_scores

        except Exception as e:
            logger.error(f"Error scoring chunks via llama_cpp adapter: {e}")
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

        max_chunk_chars = 800  # ~200 tokens
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
        start_time = asyncio.get_event_loop().time()
        time_range_map = {"day": 0, "week": 0, "month": 0, "year": 1}
        years_ago = time_range_map.get(time_range, 1) if time_range else 1
        min_date = None

        if start_date:
            try:
                min_date = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ToolException("Invalid start_date format. Use YYYY-MM-DD.")

        if end_date and min_date:
            try:
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                if end_date_dt < min_date:
                    raise ToolException("end_date cannot be before start_date.")
            except ValueError:
                raise ToolException("Invalid end_date format. Use YYYY-MM-DD.")

        topic_map = {"general": ["general"], "news": ["news"], "finance": ["business"]}
        categories = topic_map.get(topic, ["general"])

        logger.debug(
            f"[search_searxng] args:\n{format_json({'query': query, 'count': self.max_results if search_depth == 'basic' else self.max_results * 2})}"
        )

        search_results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_searxng(
                query_url=self.searxng_url,
                query=query,
                count=self.max_results
                if search_depth == "basic"
                else self.max_results * 2,
                include_sites=include_domains,
                exclude_sites=exclude_domains,
                min_date=min_date,
                categories=categories,
                years_ago=years_ago,
            ),
        )

        logger.success(f"[search_searxng] results ({len(search_results)})")

        if not search_results:
            return {
                "query": query,
                "results": [],
                "images": [] if include_images else None,
                "response_time": asyncio.get_event_loop().time() - start_time,
            }

        urls = [result["url"] for result in search_results]
        extractor = PlaywrightExtract()
        extract_format = (
            "markdown" if include_raw_content in (True, "markdown") else "text"
        )

        extract_results = await extractor._arun(
            urls=urls,
            extract_depth=search_depth,
            include_images=include_images,
            include_favicon=include_favicon,
            format=extract_format,
        )

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
                result_item["images"] = extract_result["images"]
            if include_favicon:
                result_item["favicon"] = extract_result["favicon"]
            results.append(result_item)

        results.sort(key=lambda x: x["score"], reverse=True)

        images = []
        if include_images and include_image_descriptions:
            for result in results:
                if "images" in result:
                    images.extend(result["images"])

        answer = None
        if include_answer:
            answer_content = " ".join([r["content"] for r in results[:3]])
            answer = (
                answer_content[:200] + "..."
                if len(answer_content) > 200
                else answer_content
            )

        return {
            "query": query,
            "follow_up_questions": None,
            "answer": answer,
            "images": images if include_images else None,
            "results": results[: self.max_results],
            "response_time": asyncio.get_event_loop().time() - start_time,
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
