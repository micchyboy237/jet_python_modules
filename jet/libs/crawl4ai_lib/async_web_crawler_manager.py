from typing import Any, Awaitable, Callable, List

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
)
from crawl4ai.async_dispatcher import (
    CrawlerMonitor,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# ----------------------------------------------------------------------
# Reusable Generic Crawler Manager
# ----------------------------------------------------------------------


class AsyncWebCrawlerManager:
    """
    Reusable async crawler manager with optional BM25 relevance filtering.
    """

    def __init__(
        self,
        headless: bool = True,
        verbose: bool = False,
        max_session_permit: int = 8,  # Lowered for stability
        semaphore_count: int = 12,
        memory_threshold_percent: float = 75.0,
        base_delay: tuple[float, float] = (1.2, 3.5),  # Gentler delays
        delay_before_return_html: float = 0.8,
        cache_mode: CacheMode = CacheMode.BYPASS,
        monitor_max_width: int = 130,
    ):
        self.headless = headless
        self.verbose = verbose
        self.max_session_permit = max_session_permit
        self.semaphore_count = semaphore_count
        self.memory_threshold_percent = memory_threshold_percent
        self.base_delay = base_delay
        self.delay_before_return_html = delay_before_return_html
        self.cache_mode = cache_mode
        self.monitor_max_width = monitor_max_width

        # Store defaults that will be merged into run_config dict
        self._default_run_config = {
            "cache_mode": self.cache_mode,
            "stream": True,
            "delay_before_return_html": self.delay_before_return_html,
            "semaphore_count": self.semaphore_count,
        }

        self.browser_config = BrowserConfig(
            headless=headless,
            verbose=verbose,
        )

    def _create_run_config(self, markdown_generator=None) -> dict:
        """Return a dict of run configuration parameters (to be passed as run_config=...).

        The dict will be merged with any user-provided run_config in crawl_many().
        """
        cfg = self._default_run_config.copy()
        if markdown_generator is not None:
            cfg["markdown_generator"] = markdown_generator
        return cfg

    def _merge_run_config(self, base: dict, override: dict | None) -> dict:
        """Merge user-provided run_config dict on top of base defaults."""
        if not override:
            return base.copy()
        merged = base.copy()
        merged.update(override)
        return merged

    def _create_dispatcher(self, urls_total: int) -> MemoryAdaptiveDispatcher:
        monitor = CrawlerMonitor(
            urls_total=urls_total,
            refresh_rate=1.0,
            enable_ui=True,
            max_width=self.monitor_max_width,
        )

        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=self.memory_threshold_percent,
            check_interval=1.0,
            max_session_permit=self.max_session_permit,
            memory_wait_timeout=300.0,
            rate_limiter=RateLimiter(
                base_delay=self.base_delay,
                max_delay=15.0,
                max_retries=2,
            ),
            monitor=monitor,
        )

    async def crawl_many(
        self,
        urls: List[str],
        process_result: Callable[
            [Any], Awaitable[None]
        ],  # Changed: now only takes 'result'
        user_query: str | None = None,
        bm25_threshold: float = 1.0,
        run_config: dict
        | None = None,  # Changed: now a dict of kwargs for CrawlerRunConfig
    ) -> None:
        """
        Stream crawl multiple URLs.
        The process_result callback now only receives the result.
        user_query is stored internally if BM25 is used.
        """
        if not urls:
            print("⚠️ No URLs provided.")
            return

        # Create BM25 + markdown generator only when user_query is provided
        markdown_generator = None
        if user_query:
            bm25_filter = BM25ContentFilter(
                user_query=user_query,
                bm25_threshold=bm25_threshold,
            )
            markdown_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

        # Build final run_config dict
        base_config = self._create_run_config(markdown_generator)
        final_config_dict = self._merge_run_config(base_config, run_config)

        # Instantiate CrawlerRunConfig from the dict (it accepts **kwargs via __init__)
        config = CrawlerRunConfig(**final_config_dict)

        dispatcher = self._create_dispatcher(len(urls))

        print("🚀 Starting streaming multi-URL crawl with AsyncWebCrawlerManager")
        if user_query:
            print(f'   Query         : "{user_query}"')
        print(f"   URLs          : {len(urls)}")
        print(f"   Headless      : {self.headless}")
        print(
            f"   Concurrency   : {self.max_session_permit} sessions / {self.semaphore_count} semaphore"
        )
        print("-" * 90)

        # Store user_query so we can pass it to the wrapper
        self._current_user_query = user_query

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            async for result in await crawler.arun_many(
                urls=urls,
                config=config,  # Still passes the CrawlerRunConfig instance
                dispatcher=dispatcher,
            ):
                await process_result(result)  # Only pass result

        print("\n🎉 Streaming crawl completed.\n")
