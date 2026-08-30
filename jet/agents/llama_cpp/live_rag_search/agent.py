import asyncio
from typing import List

from config import DEFAULT_LIMITS, SafetyLimits
from memory import AccumulatedMemory
from models import SufficiencyStatus
from providers.llm import (
    AnswerGenerator,
    FactExtractor,
    InnerLinkFilter,
    SufficiencyEvaluator,
)
from providers.scraper import ScraperProvider
from providers.search import SearchProvider


class LiveRAGSearchAgent:
    def __init__(
        self,
        query: str,
        search_provider: SearchProvider,
        scraper_provider: ScraperProvider,
        evaluator: SufficiencyEvaluator,
        extractor: FactExtractor,
        link_filter: InnerLinkFilter,
        generator: AnswerGenerator,
        limits: SafetyLimits = DEFAULT_LIMITS,
    ):
        self.query = query
        self.search = search_provider
        self.scraper = scraper_provider
        self.evaluator = evaluator
        self.extractor = extractor
        self.link_filter = link_filter
        self.generator = generator
        self.limits = limits
        self.memory = AccumulatedMemory()
        self.scrape_count = 0
        self._lock = asyncio.Lock()

    async def run(self) -> str:
        search_results = await self.search.search(self.query, num_results=20)
        sorted_results = sorted(search_results, key=lambda r: r.score, reverse=True)

        for result in sorted_results[: self.limits.MAX_TOP_LEVEL_RESULTS]:
            # --- Snippet pre-check ---
            snippet_check = await self.evaluator.evaluate(
                self.query, self.memory.to_context_string(), result.snippet, "snippet"
            )
            if snippet_check.status == SufficiencyStatus.COMPLETE:
                return await self.generator.generate(
                    self.query, self.memory.to_context_string()
                )

            snippet_facts = await self.extractor.extract(
                result.snippet, self.query, self.memory.get_entity_ids()
            )
            self.memory.add_facts(snippet_facts.entities, self.limits.MAX_MEMORY_FACTS)

            # --- Scrape top-level page ---
            if not await self._can_scrape():
                break

            page_content = await self.scraper.scrape(
                result.url, self.limits.SCRAPE_TIMEOUT_SEC
            )
            if page_content is None:
                continue

            page_check = await self.evaluator.evaluate(
                self.query,
                self.memory.to_context_string(),
                page_content,
                "top_level_page",
            )
            if page_check.status == SufficiencyStatus.COMPLETE:
                return await self.generator.generate(
                    self.query, self.memory.to_context_string()
                )

            page_facts = await self.extractor.extract(
                page_content, self.query, self.memory.get_entity_ids()
            )
            self.memory.add_facts(page_facts.entities, self.limits.MAX_MEMORY_FACTS)

            # --- Inner links (1 level deep) ---
            inner_links = await self.link_filter.filter_links(
                page_content,
                result.url,
                self.query,
                self.limits.MAX_INNER_LINKS_PER_PAGE,
            )
            if await self._process_inner_links(inner_links):
                return await self.generator.generate(
                    self.query, self.memory.to_context_string()
                )

        # Exhausted all results
        return await self.generator.generate(
            self.query, self.memory.to_context_string(), partial=True
        )

    async def _process_inner_links(self, inner_links: List[str]) -> bool:
        for link in inner_links[: self.limits.MAX_INNER_LINKS_PER_PAGE]:
            if not await self._can_scrape():
                return False

            content = await self.scraper.scrape(link, self.limits.SCRAPE_TIMEOUT_SEC)
            if content is None:
                continue

            check = await self.evaluator.evaluate(
                self.query, self.memory.to_context_string(), content, "inner_link"
            )
            if check.status == SufficiencyStatus.COMPLETE:
                return True

            facts = await self.extractor.extract(
                content, self.query, self.memory.get_entity_ids()
            )
            self.memory.add_facts(facts.entities, self.limits.MAX_MEMORY_FACTS)

        return False

    async def _can_scrape(self) -> bool:
        async with self._lock:
            if self.scrape_count >= self.limits.MAX_TOTAL_SCRAPES:
                return False
            self.scrape_count += 1
            return True
