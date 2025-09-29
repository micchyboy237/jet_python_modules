from typing import Any, Dict, List, Literal, Optional, Type, TypedDict, Union
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import re
import numpy as np
from jet.scrapers.playwright_utils import scrape_urls
from jet.search.searxng import search_searxng
from jet.llm.utils.embeddings import get_ollama_embedding_function
from jet.logger import logger
from bs4 import BeautifulSoup
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
        """
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
        """
    )
    search_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="basic",
        description="""Controls search thoroughness and result comprehensiveness.
        Use 'basic' for simple queries requiring quick, straightforward answers.
        Use 'advanced' for complex queries, specialized topics, rare information, or when in-depth analysis is needed.
        Default is 'basic'.
        """
    )
    include_images: Optional[bool] = Field(
        default=True,
        description="""Determines if the search returns relevant images along with text results.
        Set to True when the user explicitly requests visuals or when images would significantly enhance understanding (e.g., 'Show me what black holes look like,' 'Find pictures of Renaissance art').
        Default is True for PlaywrightSearch to leverage visual content extraction.
        """
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description="""Limits results to content published within a specific timeframe.
        ONLY set this when the user explicitly mentions a time period (e.g., 'latest AI news,' 'articles from last week').
        For less popular or niche topics, use broader time ranges ('month' or 'year') to ensure sufficient relevant results.
        Options: 'day' (24h), 'week' (7d), 'month' (30d), 'year' (365d).
        Default is None (no time restriction).
        """
    )
    topic: Optional[Literal["general", "news", "finance"]] = Field(
        default="general",
        description="""Specifies search category for optimized results.
        Use 'general' (default) for most queries, INCLUDING those with terms like 'latest,' 'newest,' or 'recent' when referring to general information.
        Use 'finance' for markets, investments, economic data, or financial news.
        Use 'news' ONLY for politics, sports, or major current events covered by mainstream media - NOT simply because a query asks for 'new' information.
        """
    )
    include_favicon: Optional[bool] = Field(
        default=True,
        description="""Determines whether to include favicon URLs for each search result.
        When enabled, each search result will include the website's favicon URL, useful for:
        - Building rich UI interfaces with visual website indicators
        - Providing visual cues about the source's credibility or brand
        - Creating bookmark-like displays with recognizable site icons
        Default is True to enhance result presentation.
        """
    )
    start_date: Optional[str] = Field(
        default=None,
        description="""Filters search results to include only content published on or after this date.
        Use this when you need to:
        - Find recent developments or updates on a topic
        - Exclude outdated information from search results
        - Focus on content within a specific timeframe
        - Combine with end_date to create a custom date range
        Format must be YYYY-MM-DD (e.g., '2024-01-15' for January 15, 2024).
        Examples:
        - '2024-01-01' - Results from January 1, 2024 onwards
        - '2023-12-25' - Results from December 25, 2023 onwards
        When combined with end_date, creates a precise date range filter.
        Default is None (no start date restriction).
        """
    )
    end_date: Optional[str] = Field(
        default=None,
        description="""Filters search results to include only content published on or before this date.
        Use this when you need to:
        - Exclude content published after a certain date
        - Study historical information or past events
        - Research how topics were covered during specific time periods
        - Combine with start_date to create a custom date range
        Format must be YYYY-MM-DD (e.g., '2024-03-31' for March 31, 2024).
        Examples:
        - '2024-03-31' - Results up to and including March 31, 2024
        - '2023-12-31' - Results up to and including December 31, 2023
        When combined with start_date, creates a precise date range filter.
        For example: start_date='2024-01-01', end_date='2024-03-31' returns results from Q1 2024 only.
        Default is None (no end date restriction).
        """
    )
    max_content_length: Optional[int] = Field(
        default=500,
        description="Maximum length of the content field in characters. Default is 500."
    )
    ollama_embed_model: Optional[str] = Field(
        default="nomic-embed-text",
        description="Ollama embedding model to use for relevance scoring. Default is 'nomic-embed-text'."
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
    searxng_url: str = Field(default="http://jethros-macbook-air.local:3000")
    max_results: Optional[int] = Field(default=5)
    include_image_descriptions: Optional[bool] = Field(default=False)
    max_content_length: Optional[int] = Field(default=500)
    ollama_embed_model: str = Field(default="nomic-embed-text")
    ollama_url: str = Field(default="http://localhost:11434")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._embed_func = get_ollama_embedding_function(
            model=self.ollama_embed_model,
            url=self.ollama_url,
            return_format="numpy"
        )
        self._query_embedding_cache = {}
        logger.debug("Initialized PlaywrightSearchAPIWrapper with model %s and URL %s", 
                    self.ollama_embed_model, self.ollama_url)

    def _score_chunks(self, chunks: List[str], query: str) -> List[float]:
        """Score multiple text chunks based on semantic similarity to the query using Ollama embeddings."""
        logger.debug("Scoring %d chunks for query: %s", len(chunks), query)
        if not chunks or not query:
            logger.warning("Empty chunks or query provided, returning zero scores")
            return [0.0] * len(chunks)
        try:
            # Cache query embedding to avoid redundant computation
            if query not in self._query_embedding_cache:
                logger.debug("Generating embedding for query: %s", query)
                self._query_embedding_cache[query] = self._embed_func(query)
            query_embedding = self._query_embedding_cache[query]
            
            # Generate embeddings for all chunks in a single batch
            logger.debug("Generating embeddings for %d chunks", len(chunks))
            chunk_embeddings = self._embed_func(chunks)
            
            # Calculate cosine similarity for all chunks
            scores = []
            for chunk_embedding in chunk_embeddings:
                dot_product = np.dot(query_embedding, chunk_embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_chunk = np.linalg.norm(chunk_embedding)
                if norm_query == 0 or norm_chunk == 0:
                    scores.append(0.0)
                else:
                    similarity = dot_product / (norm_query * norm_chunk)
                    scores.append(max(0.0, min(1.0, similarity)))
            logger.debug("Computed %d similarity scores", len(scores))
            return scores
        except Exception as e:
            logger.error("Error scoring chunks: %s", str(e))
            return [0.0] * len(chunks)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, using NLTK if available, else regex."""
        logger.debug("Splitting text into sentences, NLTK available: %s", NLTK_AVAILABLE)
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
            logger.debug("Split text into %d sentences using NLTK", len(sentences))
            return sentences
        # Fallback regex for sentence splitting
        sentence_endings = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        logger.debug("Split text into %d sentences using regex", len(sentences))
        return sentences

    def _extract_relevant_content(self, raw_content: str, query: str, max_length: int) -> str:
        """Extract the most relevant content from raw_content up to max_length, with [...] separators."""
        logger.debug("Extracting relevant content for query: %s, max_length: %d", query, max_length)
        if not raw_content:
            logger.warning("No raw content provided, returning empty string")
            return ""
        chunks = self._split_into_sentences(raw_content)
        if not chunks:
            logger.warning("No chunks after splitting, returning empty string")
            return ""
        max_chunk_tokens = 200
        max_chunk_chars = max_chunk_tokens * 4
        chunks = [chunk for chunk in chunks if len(chunk) <= max_chunk_chars]
        if not chunks:
            logger.debug("No valid chunks after filtering, truncating raw content")
            return raw_content[:max_length] + "..." if len(raw_content) > max_length else raw_content
        
        # Score all chunks in a single batch
        scores = self._score_chunks(chunks, query)
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        content = ""
        separator = " [...] "
        selected_chunks = 0
        max_chunks = 3
        for chunk, _ in scored_chunks:
            if selected_chunks >= max_chunks:
                break
            chunk_with_separator = chunk + separator if selected_chunks < max_chunks - 1 else chunk
            if len(content) + len(chunk_with_separator) <= max_length:
                content += chunk_with_separator
                selected_chunks += 1
            else:
                remaining = max_length - len(content)
                if remaining > 10:
                    content += chunk[:remaining].rsplit(' ', 1)[0] + "..."
                break
        content = content.strip()
        if not content:
            logger.debug("No content selected, using first chunk")
            content = chunks[0][:max_length] + "..." if len(chunks[0]) > max_length else chunks[0]
        if content.endswith(separator):
            content = content[:-len(separator)]
        logger.debug("Extracted content length: %d characters", len(content))
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
        """Search the web using Playwright and SearXNG asynchronously."""
        logger.info("Starting async search for query: %s, search_depth: %s", query, search_depth)
        start_time = asyncio.get_event_loop().time()
        time_range_map = {"day": 0, "week": 0, "month": 0, "year": 1}
        years_ago = time_range_map.get(time_range, 1) if time_range else 1
        min_date = None
        if start_date:
            try:
                min_date = datetime.strptime(start_date, "%Y-%m-%d")
                logger.debug("Set min_date to %s", start_date)
            except ValueError:
                logger.error("Invalid start_date format: %s", start_date)
                raise ToolException("Invalid start_date format. Use YYYY-MM-DD.")
        if end_date and min_date:
            try:
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
                if end_date_dt < min_date:
                    logger.error("end_date %s is before start_date %s", end_date, start_date)
                    raise ToolException("end_date cannot be before start_date.")
            except ValueError:
                logger.error("Invalid end_date format: %s", end_date)
                raise ToolException("Invalid end_date format. Use YYYY-MM-DD.")
        topic_map = {
            "general": ["general"],
            "news": ["news"],
            "finance": ["business"]
        }
        categories = topic_map.get(topic, ["general"])
        logger.debug("Searching with categories: %s, include_domains: %s, exclude_domains: %s", 
                    categories, include_domains, exclude_domains)
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
        logger.info("Retrieved %d search results from SearXNG", len(search_results))
        if not search_results:
            logger.warning("No search results found for query: %s", query)
            return {
                "query": query,
                "results": [],
                "images": [],
                "response_time": asyncio.get_event_loop().time() - start_time
            }
        urls = [result["url"] for result in search_results]
        logger.debug("Scraping %d URLs", len(urls))
        num_parallel = 5 if search_depth == "basic" else 10
        wait_for_js = search_depth == "advanced"
        scrape_results = []
        async for result in scrape_urls(
            urls=urls,
            num_parallel=num_parallel,
            limit=self.max_results,
            show_progress=False,
            timeout=10000,
            max_retries=1,
            with_screenshot=include_images,
            headless=True,
            wait_for_js=wait_for_js,
            use_cache=True
        ):
            logger.debug(f"Scraped URL status: {result["status"]}")
            if result["status"] != "started":
                scrape_results.append(result)
        logger.info("Scraped %d URLs", len(scrape_results))
        results = []
        search_texts = [result["content"] for result in search_results]
        embed_scores = self._score_chunks(search_texts, query)
        for search_result, scrape_result, embed_score in zip(search_results, scrape_results, embed_scores):
            if scrape_result["status"] != "completed" or not scrape_result["html"]:
                logger.warning("Skipping result due to incomplete scrape: %s", search_result["url"])
                continue
            # Parse HTML to extract title, content, images, and favicon
            logger.debug("Parsing HTML for URL: %s", search_result["url"])
            soup = BeautifulSoup(scrape_result["html"], "html.parser")
            title = soup.title.string.strip() if soup.title else search_result["title"]
            content_elements = soup.find_all(["p", "div", "article"])
            raw_content = " ".join([elem.get_text(strip=True) for elem in content_elements])
            content = self._extract_relevant_content(
                raw_content,
                query,
                self.max_content_length
            ) if raw_content else search_result["content"]
            images = []
            if include_images:
                img_tags = soup.find_all("img")
                images = [img.get("src") for img in img_tags if img.get("src")]
                logger.debug("Extracted %d images from %s", len(images), search_result["url"])
            favicon = None
            if include_favicon:
                favicon_tag = soup.find("link", rel=lambda x: x and "icon" in x.lower())
                favicon = favicon_tag.get("href") if favicon_tag else None
                logger.debug("Favicon %s for %s", "found" if favicon else "not found", search_result["url"])
            # Format raw content based on include_raw_content parameter
            formatted_raw_content = None
            if include_raw_content:
                if include_raw_content == "markdown":
                    formatted_raw_content = f"# {title}\n\n{content}"
                else:
                    formatted_raw_content = raw_content
            results.append({
                "url": search_result["url"],
                "title": title,
                "content": content,
                "raw_score": search_result["score"],
                "score": embed_score,
                "raw_content": formatted_raw_content,
                "images": images,
                "favicon": favicon
            })
        logger.info("Processed %d valid results", len(results))
        results.sort(key=lambda x: x["score"], reverse=True)
        images = []
        if include_images and include_image_descriptions:
            for result in results:
                images.extend(result["images"])
        answer = None
        if include_answer:
            answer_depth = "advanced" if include_answer == "advanced" else "basic"
            answer_content = " ".join([r["content"] for r in results[:3]])
            answer = answer_content[:200] + "..." if len(answer_content) > 200 else answer_content
            logger.debug("Generated answer of length %d for query: %s", len(answer or ""), query)
        response_time = asyncio.get_event_loop().time() - start_time
        logger.info("Search completed in %.2f seconds, returning %d results", response_time, len(results))
        return {
            "query": query,
            "follow_up_questions": None,
            "answer": answer,
            "images": images,
            "results": results[:self.max_results],
            "response_time": response_time
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
        logger.info("Starting sync search for query: %s, search_depth: %s", query, search_depth)
        result = asyncio.run(self.raw_results_async(
            query, include_domains, exclude_domains, search_depth, include_images,
            time_range, topic, include_favicon, start_date, end_date,
            include_answer, include_raw_content, include_image_descriptions,
            auto_parameters, country
        ))
        logger.info("Sync search completed for query: %s", query)
        return result

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
    include_images: bool = True
    time_range: Optional[Literal["day", "week", "month", "year"]] = None
    topic: Optional[Literal["general", "news", "finance"]] = None
    include_favicon: bool = True
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_results: Optional[int] = None
    include_answer: Optional[Union[bool, Literal["basic", "advanced"]]] = None
    include_raw_content: Optional[Union[bool, Literal["markdown", "text"]]] = None
    include_image_descriptions: bool = False
    auto_parameters: Optional[bool] = None
    country: Optional[str] = None
    api_wrapper: PlaywrightSearchAPIWrapper = Field(default_factory=PlaywrightSearchAPIWrapper)

    def _run(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: bool = True,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        include_favicon: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        logger.info("Executing synchronous search tool for query: %s", query)
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
                logger.warning("No results found for query: %s", query)
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
                logger.error(error_message)
                raise ToolException(error_message)
            logger.info("Synchronous search tool completed for query: %s, %d results returned", 
                       query, len(raw_results.get("results", [])))
            return raw_results
        except ToolException:
            logger.error("ToolException occurred during synchronous search: %s", query)
            raise
        except Exception as e:
            logger.error("Unexpected error during synchronous search: %s", str(e))
            return {"error": str(e)}

    async def _arun(
        self,
        query: str,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_depth: Optional[Literal["basic", "advanced"]] = None,
        include_images: bool = True,
        time_range: Optional[Literal["day", "week", "month", "year"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        include_favicon: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Dict[str, Any]:
        logger.info("Executing asynchronous search tool for query: %s", query)
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
                logger.warning("No results found for async query: %s", query)
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
                logger.error(error_message)
                raise ToolException(error_message)
            logger.info("Asynchronous search tool completed for query: %s, %d results returned", 
                       query, len(raw_results.get("results", [])))
            return raw_results
        except ToolException:
            logger.error("ToolException occurred during asynchronous search: %s", query)
            raise
        except Exception as e:
            logger.error("Unexpected error during asynchronous search: %s", str(e))
            return {"error": str(e)}

    def _generate_suggestions(self, params: Dict[str, Any]) -> List[str]:
        """Generate helpful suggestions based on the failed search parameters."""
        logger.debug("Generating suggestions for failed search with params: %s", params)
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
        logger.debug("Generated %d suggestions", len(suggestions))
        return suggestions
