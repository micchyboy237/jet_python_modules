"""Reusable async crawler manager with BM25 relevance filtering.

Wraps Crawl4AI's AsyncWebCrawler with sensible defaults, automatic
MemoryAdaptiveDispatcher creation, and streaming result processing.
"""

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
from jet.libs.crawl4ai_lib.config import (
    CRAWLER_CHECK_INTERVAL,
    CRAWLER_MAX_SESSION_PERMIT,
    CRAWLER_MEMORY_THRESHOLD_PERCENT,
    CRAWLER_MONITOR_MAX_WIDTH,
)
from jet.logger import logger


class AsyncWebCrawlerManager:
    """Reusable async crawler manager with optional BM25 relevance filtering."""

    def __init__(
        self,
        headless: bool = True,
        verbose: bool = False,
        max_session_permit: int = CRAWLER_MAX_SESSION_PERMIT,
        semaphore_count: int = 12,
        memory_threshold_percent: float = CRAWLER_MEMORY_THRESHOLD_PERCENT,
        base_delay: tuple[float, float] = (1.2, 3.5),
        delay_before_return_html: float = 1.0,
        cache_mode: CacheMode = CacheMode.BYPASS,
        monitor_max_width: int = CRAWLER_MONITOR_MAX_WIDTH,
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
        """Return a dict of run configuration parameters."""
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
        """Create MemoryAdaptiveDispatcher with correct CrawlerMonitor signature."""
        monitor = CrawlerMonitor(
            urls_total=urls_total,
            refresh_rate=CRAWLER_CHECK_INTERVAL,
            enable_ui=True,
            max_width=self.monitor_max_width,
        )

        return MemoryAdaptiveDispatcher(
            memory_threshold_percent=self.memory_threshold_percent,
            check_interval=CRAWLER_CHECK_INTERVAL,
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
        process_result: Callable[[Any], Awaitable[None]],
        user_query: str | None = None,
        bm25_threshold: float = 1.0,
        run_config: dict | None = None,
    ) -> None:
        """Stream crawl multiple URLs with memory-adaptive concurrency.

        The dispatcher is created internally via _create_dispatcher() using
        the manager's configured memory_threshold_percent and max_session_permit.
        """
        if not urls:
            logger.warning("crawl_many: no URLs provided")
            print("⚠️ No URLs provided.")
            return

        markdown_generator = None
        if user_query:
            bm25_filter = BM25ContentFilter(
                user_query=user_query,
                bm25_threshold=bm25_threshold,
            )
            markdown_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

        base_config = self._create_run_config(markdown_generator)
        final_config_dict = self._merge_run_config(base_config, run_config)
        config = CrawlerRunConfig(**final_config_dict)

        dispatcher = self._create_dispatcher(len(urls))

        logger.info(
            f"crawl_many: starting stream crawl for {len(urls)} URLs, "
            f"memory_threshold={self.memory_threshold_percent}%, "
            f"max_sessions={self.max_session_permit}"
        )
        print("🚀 Starting streaming multi-URL crawl with AsyncWebCrawlerManager")
        if user_query:
            print(f'   Query         : "{user_query}"')
        print(f"   URLs          : {len(urls)}")
        print(f"   Headless      : {self.headless}")
        print(
            f"   Concurrency   : {self.max_session_permit} sessions / "
            f"{self.semaphore_count} semaphore"
        )
        print("-" * 90)

        self._current_user_query = user_query

        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            async for result in await crawler.arun_many(
                urls=urls,
                config=config,
                dispatcher=dispatcher,
            ):
                await process_result(result)

        logger.info("crawl_many: streaming crawl completed")
        print("\n🎉 Streaming crawl completed.\n")
