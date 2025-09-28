from typing import Any, Dict, List, Literal, Optional, Type, TypedDict, Union
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
from jet.search.playwright.playwright_extract import PlaywrightExtract
from jet.search.searxng import search_searxng

class PlaywrightSearchInput(BaseModel):
    """Input for PlaywrightSearch"""
    query: str = Field(description="Search query to look up")
    include_domains: Optional[List[str]] = Field(
        default=[],
        description="""A list of domains to restrict search results to.
        Use this parameter when:
        1. The user explicitly requests information from specific websites (e.g., "Find climate data from nasa.gov")
        2. The user mentions an organization or company without specifying the domain (e.g., "Find information about iPhones from Apple")
        In both cases, you should determine the appropriate domains (e.g., ["nasa.gov"] or ["apple.com"]) and set this parameter.
        Results will ONLY come from the specified domains - no other sources will be included.
        Default is None (no domain restriction).
        """,
    )
    exclude_domains: Optional[List[str]] = Field(
        default=[],
        description="""A list of domains to exclude from search results.
        Use this parameter when:
        1. The user explicitly requests to avoid certain websites (e.g., "Find information about climate change but not from twitter.com")
        2. The user mentions not wanting results from specific organizations without naming the domain (e.g., "Find phone reviews but nothing from Apple")
        In both cases, you should determine the appropriate domains to exclude (e.g., ["twitter.com"] or ["apple.com"]) and set this parameter.
        Results will filter out all content from the specified domains.
        Default is None (no domain exclusion).
        """,
    )
    search_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Controls search thoroughness and result comprehensiveness.
        Use "basic" for simple queries requiring quick, straightforward answers.
        Use "advanced" for complex queries, specialized topics,
        rare information, or when in-depth analysis is needed.
        """,
    )
    include_images: Optional[bool] = Field(
        default=False,
        description="""Determines if the search returns relevant images along with text results.
        Set to True when the user explicitly requests visuals or when images would
        significantly enhance understanding (e.g., "Show me what black holes look like,"
        "Find pictures of Renaissance art").
        Leave as False (default) for most informational queries where text is sufficient.
        """,
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description="""Limits results to content published within a specific timeframe.
        ONLY set this when the user explicitly mentions a time period
        (e.g., "latest AI news," "articles from last week").
        For less popular or niche topics, use broader time ranges
        ("month" or "year") to ensure sufficient relevant results.
        Options: "day" (24h), "week" (7d), "month" (30d), "year" (365d).
        Default is None.
        """,
    )
    topic: Optional[Literal["general", "news", "finance"]] = Field(
        default="general",
        description="""Specifies search category for optimized results.
        Use "general" (default) for most queries, INCLUDING those with terms like
        "latest," "newest," or "recent" when referring to general information.
        Use "finance" for markets, investments, economic data, or financial news.
        Use "news" ONLY for politics, sports, or major current events covered by
        mainstream media - NOT simply because a query asks for "new" information.
        """,
    )
    include_favicon: Optional[bool] = Field(
        default=False,
        description="""Determines whether to include favicon URLs for each search result.
        When enabled, each search result will include the website's favicon URL,
        which can be useful for:
        - Building rich UI interfaces with visual website indicators
        - Providing visual cues about the source's credibility or brand
        - Creating bookmark-like displays with recognizable site icons
        Set to True when creating user interfaces that benefit from visual branding
        or when favicon information enhances the user experience.
        Default is False to minimize response size and API usage.
        """,
    )
    start_date: Optional[str] = Field(
        default=None,
        description="""Filters search results to include only content published on or after this date.
        Use this parameter when you need to:
        - Find recent developments or updates on a topic
        - Exclude outdated information from search results
        - Focus on content within a specific timeframe
        - Combine with end_date to create a custom date range
        Format must be YYYY-MM-DD (e.g., "2024-01-15" for January 15, 2024).
        Examples:
        - "2024-01-01" - Results from January 1, 2024 onwards
        - "2023-12-25" - Results from December 25, 2023 onwards
        When combined with end_date, creates a precise date range filter.
        Default is None (no start date restriction).
        """,
    )
    end_date: Optional[str] = Field(
        default=None,
        description="""Filters search results to include only content published on or before this date.
        Use this parameter when you need to:
        - Exclude content published after a certain date
        - Study historical information or past events
        - Research how topics were covered during specific time periods
        - Combine with start_date to create a custom date range
        Format must be YYYY-MM-DD (e.g., "2024-03-31" for March 31, 2024).
        Examples:
        - "2024-03-31" - Results up to and including March 31, 2024
        - "2023-12-31" - Results up to and including December 31, 2023
        When combined with start_date, creates a precise date range filter.
        For example: start_date="2024-01-01", end_date="2024-03-31"
        returns results from Q1 2024 only.
        Default is None (no end date restriction).
        """,
    )

class PlaywrightSearchResult(TypedDict):
    url: str
    title: str
    content: str
    score: float
    raw_content: Optional[str]
    images: List[str]
    favicon: Optional[str]

class PlaywrightSearchAPIWrapper(BaseModel):
    """Wrapper for Playwright-based search engine."""
    searxng_url: str = Field(default="http://jethros-macbook-air.local:3000")
    max_results: Optional[int] = Field(default=5)
    include_image_descriptions: Optional[bool] = Field(default=False)

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
        """Search the web using Playwright and SearXNG asynchronously."""
        start_time = asyncio.get_event_loop().time()
        
        # Map time_range to years_ago for searxng
        time_range_map = {"day": 0, "week": 0, "month": 0, "year": 1}
        years_ago = time_range_map.get(time_range, 1) if time_range else 1
        
        # Convert start_date/end_date to min_date if provided
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

        # Map topic to searxng categories
        topic_map = {
            "general": ["general"],
            "news": ["news"],
            "finance": ["business"]
        }
        categories = topic_map.get(topic, ["general"])

        # Perform search using searxng
        search_results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_searxng(
                query_url=self.searxng_url,
                query=query,
                count=self.max_results if search_depth == "basic" else self.max_results * 2,
                include_sites=include_domains,
                exclude_sites=exclude_domains,
                min_date=min_date,
                categories=categories,
                years_ago=years_ago
            )
        )

        if not search_results:
            return {
                "query": query,
                "results": [],
                "images": [],
                "response_time": asyncio.get_event_loop().time() - start_time
            }

        # Extract content using PlaywrightExtract
        urls = [result["url"] for result in search_results]
        extractor = PlaywrightExtract()
        extract_format = "markdown" if include_raw_content in (True, "markdown") else "text"
        extract_results = await extractor._arun(
            urls=urls,
            extract_depth=search_depth,
            include_images=include_images,
            include_favicon=include_favicon,
            format=extract_format
        )

        # Combine results
        results = []
        for search_result, extract_result in zip(search_results, extract_results["results"]):
            if "error" in extract_result:
                continue
            results.append({
                "url": search_result["url"],
                "title": search_result["title"],
                "content": search_result["content"],
                "score": search_result["score"],
                "raw_content": extract_result["raw_content"] if include_raw_content else None,
                "images": extract_result["images"] if include_images else [],
                "favicon": extract_result["favicon"] if include_favicon else None
            })

        # Handle images separately if include_image_descriptions
        images = []
        if include_images and include_image_descriptions:
            for result in results:
                images.extend(result["images"])

        # Generate answer if requested
        answer = None
        if include_answer:
            answer_depth = "advanced" if include_answer == "advanced" else "basic"
            answer_content = " ".join([r["content"] for r in results[:3]])
            answer = answer_content[:200] + "..." if len(answer_content) > 200 else answer_content

        return {
            "query": query,
            "follow_up_questions": None,
            "answer": answer,
            "images": images,
            "results": results[:self.max_results],
            "response_time": asyncio.get_event_loop().time() - start_time
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
        """Synchronous wrapper for async search."""
        return asyncio.run(self.raw_results_async(
            query, include_domains, exclude_domains, search_depth, include_images,
            time_range, topic, include_favicon, start_date, end_date,
            include_answer, include_raw_content, include_image_descriptions,
            auto_parameters, country
        ))

class PlaywrightSearch(BaseTool):
    """Tool that searches the web using Playwright and SearXNG."""
    name: str = "playwright_search"
    description: str = (
        "A search engine using Playwright and SearXNG for comprehensive, accurate results. "
        "Useful for answering questions about current events. "
        "Supports advanced search depths, domain management, time range filters, and image search."
    )
    args_schema: Type[BaseModel] = PlaywrightSearchInput
    handle_tool_error: bool = True
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None
    search_depth: Optional[Literal["basic", "advanced"]] = None
    include_images: Optional[bool] = None
    time_range: Optional[Literal["day", "week", "month", "year"]] = None
    topic: Optional[Literal["general", "news", "finance"]] = None
    include_favicon: Optional[bool] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_results: Optional[int] = None
    include_answer: Optional[Union[bool, Literal["basic", "advanced"]]] = None
    include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]] = None
    include_image_descriptions: Optional[bool] = None
    auto_parameters: Optional[bool] = None
    country: Optional[str] = None
    api_wrapper: PlaywrightSearchAPIWrapper = Field(default_factory=PlaywrightSearchAPIWrapper)

    def _run(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: Optional[bool] = None,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        include_favicon: Optional[bool] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            raw_results = self.api_wrapper.raw_results(
                query=query,
                include_domains=self.include_domains if self.include_domains else include_domains,
                exclude_domains=self.exclude_domains if self.exclude_domains else exclude_domains,
                search_depth=self.search_depth if self.search_depth else search_depth,
                include_images=self.include_images if self.include_images else include_images,
                time_range=self.time_range if self.time_range else time_range,
                topic=self.topic if self.topic else topic,
                include_favicon=self.include_favicon if self.include_favicon else include_favicon,
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
                    f"Try modifying your search parameters with one of these approaches."
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
        include_images: Optional[bool] = None,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        include_favicon: Optional[bool] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        try:
            raw_results = await self.api_wrapper.raw_results_async(
                query=query,
                include_domains=self.include_domains if self.include_domains else include_domains,
                exclude_domains=self.exclude_domains if self.exclude_domains else exclude_domains,
                search_depth=self.search_depth if self.search_depth else search_depth,
                include_images=self.include_images if self.include_images else include_images,
                time_range=self.time_range if self.time_range else time_range,
                topic=self.topic if self.topic else topic,
                include_favicon=self.include_favicon if self.include_favicon else include_favicon,
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
                    f"Try modifying your search parameters with one of these approaches."
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
        search_depth = params.get("search_depth")
        exclude_domains = params.get("exclude_domains")
        include_domains = params.get("include_domains")
        time_range = params.get("time_range")
        topic = params.get("topic")
        if time_range:
            suggestions.append("Remove time_range argument")
        if include_domains:
            suggestions.append("Remove include_domains argument")
        if exclude_domains:
            suggestions.append("Remove exclude_domains argument")
        if search_depth == "basic":
            suggestions.append("Try a more detailed search using 'advanced' search_depth")
        if topic != "general":
            suggestions.append("Try a general search using 'general' topic")
        return suggestions
