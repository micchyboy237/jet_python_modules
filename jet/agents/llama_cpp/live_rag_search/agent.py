# jet_python_modules/jet/agents/llama_cpp/live_rag_search/agent.py

import asyncio
from typing import List

from config import DEFAULT_LIMITS, SafetyLimits
from jet.logger import logger
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
        logger.info(f"🚀 Starting Live RAG for query: {self.query}")

        search_results = await self.search.search(self.query, num_results=20)
        sorted_results = sorted(search_results, key=lambda r: r.score, reverse=True)

        limit = min(len(sorted_results), self.limits.MAX_TOP_LEVEL_RESULTS)
        logger.info(f"🔍 Found {len(sorted_results)} results. Processing top {limit}.")

        for idx, result in enumerate(sorted_results[:limit]):
            logger.info(f"🔄 Outer Loop [{idx + 1}/{limit}]: {result.url}")

            # 1. Snippet Check
            snippet_check = await self.evaluator.evaluate(
                self.query, self.memory.to_context_string(), result.snippet, "snippet"
            )
            if snippet_check.status == SufficiencyStatus.COMPLETE:
                logger.info("✅ Sufficiency met at SNIPPET level.")
                return await self.generator.generate(
                    self.query, self.memory.to_context_string()
                )

            snippet_facts = await self.extractor.extract(
                result.snippet, self.query, self.memory.get_entity_ids()
            )
            added = self.memory.add_facts(
                snippet_facts.entities, self.limits.MAX_MEMORY_FACTS
            )
            if added:
                logger.debug(f"   + Added {added} facts from snippet.")

            # 2. Scrape Page
            if not await self._can_scrape():
                logger.warning("⛔ Scrape limit reached.")
                break

            logger.info(f"   🕷️ Scraping: {result.url}")
            page_content = await self.scraper.scrape(
                result.url, self.limits.SCRAPE_TIMEOUT_SEC
            )
            if not page_content:
                logger.warning(f"   ⚠️ Failed to scrape {result.url}")
                continue

            # 3. Page Check
            page_check = await self.evaluator.evaluate(
                self.query,
                self.memory.to_context_string(),
                page_content,
                "top_level_page",
            )
            if page_check.status == SufficiencyStatus.COMPLETE:
                logger.info("✅ Sufficiency met at PAGE level.")
                return await self.generator.generate(
                    self.query, self.memory.to_context_string()
                )

            page_facts = await self.extractor.extract(
                page_content, self.query, self.memory.get_entity_ids()
            )
            added = self.memory.add_facts(
                page_facts.entities, self.limits.MAX_MEMORY_FACTS
            )
            if added:
                logger.debug(f"   + Added {added} facts from page content.")

            # 4. Inner Links
            # Note: We need raw HTML for link extraction.
            # Since scraper returns text, we assume for this skeleton that
            # we can't extract links reliably unless we modify scraper to return HTML.
            # However, to keep the flow working, we'll skip inner links if we only have text.
            # In a production version, scraper.scrape should return (text, html).
            inner_links = []

            # If you implemented PlaywrightScraperProvider which stores _last_html:
            if hasattr(self.scraper, "_last_html"):
                all_links = await self.scraper.extract_links(
                    self.scraper._last_html, result.url
                )
                inner_links = await self.link_filter.filter_links(
                    all_links,
                    result.url,
                    self.query,
                    self.limits.MAX_INNER_LINKS_PER_PAGE,
                )

            if inner_links:
                logger.info(f"   🔗 Found {len(inner_links)} relevant inner links.")
                if await self._process_inner_links(inner_links):
                    return await self.generator.generate(
                        self.query, self.memory.to_context_string()
                    )
            else:
                logger.info("   🔗 No inner links processed.")

        logger.warning("⚠️ Exhausted results. Generating partial answer.")
        return await self.generator.generate(
            self.query, self.memory.to_context_string(), partial=True
        )

    async def _process_inner_links(self, inner_links: List[str]) -> bool:
        for i, link in enumerate(inner_links):
            if not await self._can_scrape():
                return False

            logger.info(f"      ↳ Inner Link [{i + 1}/{len(inner_links)}]: {link}")
            content = await self.scraper.scrape(link, self.limits.SCRAPE_TIMEOUT_SEC)
            if not content:
                continue

            check = await self.evaluator.evaluate(
                self.query, self.memory.to_context_string(), content, "inner_link"
            )
            if check.status == SufficiencyStatus.COMPLETE:
                logger.info("      ✅ Sufficiency met at INNER LINK level.")
                return True

            facts = await self.extractor.extract(
                content, self.query, self.memory.get_entity_ids()
            )
            added = self.memory.add_facts(facts.entities, self.limits.MAX_MEMORY_FACTS)
            if added:
                logger.debug(f"         + Added {added} facts from inner link.")

        return False

    async def _can_scrape(self) -> bool:
        async with self._lock:
            if self.scrape_count >= self.limits.MAX_TOTAL_SCRAPES:
                return False
            self.scrape_count += 1
            return True
