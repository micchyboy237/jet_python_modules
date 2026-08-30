# jet_python_modules/jet/agents/llama_cpp/live_rag_search/agent.py

import asyncio
from typing import List

from config import DEFAULT_LIMITS, SafetyLimits
from jet.logger import logger
from memory import AccumulatedMemory
from models import SufficiencyStatus
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from providers.llm import (
    AnswerGenerator,
    FactExtractor,
    InnerLinkFilter,
    SufficiencyEvaluator,
)
from providers.scraper import ScraperProvider
from providers.search import SearchProvider

tracer = trace.get_tracer(__name__)


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
        with tracer.start_as_current_span(
            "live_rag.agent.run",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                SpanAttributes.INPUT_VALUE: self.query,
                "live_rag.limits.max_scrapes": self.limits.MAX_TOTAL_SCRAPES,
                "live_rag.limits.max_memory_facts": self.limits.MAX_MEMORY_FACTS,
            },
        ) as root_span:
            logger.info(f"🚀 Starting Live RAG for query: {self.query}")

            # --- Web Search Phase ---
            search_results = await self.search.search(self.query, num_results=20)
            sorted_results = sorted(search_results, key=lambda r: r.score, reverse=True)

            limit = min(len(sorted_results), self.limits.MAX_TOP_LEVEL_RESULTS)
            logger.info(
                f"🔍 Found {len(sorted_results)} results. Processing top {limit}."
            )
            root_span.set_attribute("live_rag.search.result_count", len(sorted_results))

            # --- Outer Loop: Top-Level Search Results ---
            for idx, result in enumerate(sorted_results[:limit]):
                with tracer.start_as_current_span(
                    f"live_rag.agent.outer_loop.{idx}",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                        "live_rag.url": result.url,
                        "live_rag.score": result.score,
                        "live_rag.title": result.title,
                    },
                ) as loop_span:
                    logger.info(f"🔄 Outer Loop [{idx + 1}/{limit}]: {result.url}")

                    # === Step 1: Snippet Sufficiency Check ===
                    snippet_check = await self.evaluator.evaluate(
                        self.query,
                        self.memory.to_context_string(),
                        result.snippet,
                        "snippet",
                    )
                    loop_span.set_attribute(
                        "live_rag.snippet.status", snippet_check.status.value
                    )
                    loop_span.set_attribute(
                        "live_rag.snippet.reasoning", snippet_check.reasoning[:500]
                    )

                    if snippet_check.status == SufficiencyStatus.COMPLETE:
                        logger.info("✅ Sufficiency met at SNIPPET level.")
                        root_span.set_attribute(
                            "live_rag.stop_reason", "snippet_sufficiency"
                        )
                        return await self.generator.generate(
                            self.query, self.memory.to_context_string()
                        )

                    # Extract facts from snippet even if insufficient
                    snippet_facts = await self.extractor.extract(
                        result.snippet, self.query, self.memory.get_entity_ids()
                    )
                    added_snippet = self.memory.add_facts(
                        snippet_facts.entities, self.limits.MAX_MEMORY_FACTS
                    )
                    loop_span.set_attribute(
                        "live_rag.memory.facts_added_snippet", added_snippet
                    )
                    loop_span.set_attribute(
                        "live_rag.memory.total_facts", self.memory.total_fact_count
                    )
                    if added_snippet:
                        logger.debug(f"   + Added {added_snippet} facts from snippet.")

                    # === Step 2: Scrape Top-Level Page ===
                    if not await self._can_scrape():
                        logger.warning("⛔ Scrape budget exhausted.")
                        root_span.set_attribute(
                            "live_rag.stop_reason", "max_scrapes_reached"
                        )
                        break

                    logger.info(f"   🕷️ Scraping: {result.url}")
                    page_content = await self.scraper.scrape(
                        result.url, self.limits.SCRAPE_TIMEOUT_SEC
                    )

                    if not page_content:
                        logger.warning(f"   ⚠️ Failed to scrape {result.url}")
                        loop_span.set_attribute("live_rag.scrape.success", False)
                        continue

                    loop_span.set_attribute("live_rag.scrape.success", True)
                    loop_span.set_attribute(
                        "live_rag.scrape.content_length", len(page_content)
                    )

                    # === Step 3: Page Content Sufficiency Check ===
                    page_check = await self.evaluator.evaluate(
                        self.query,
                        self.memory.to_context_string(),
                        page_content,
                        "top_level_page",
                    )
                    loop_span.set_attribute(
                        "live_rag.page.status", page_check.status.value
                    )
                    loop_span.set_attribute(
                        "live_rag.page.reasoning", page_check.reasoning[:500]
                    )

                    if page_check.status == SufficiencyStatus.COMPLETE:
                        logger.info("✅ Sufficiency met at PAGE level.")
                        root_span.set_attribute(
                            "live_rag.stop_reason", "page_sufficiency"
                        )
                        return await self.generator.generate(
                            self.query, self.memory.to_context_string()
                        )

                    # Extract facts from page content
                    page_facts = await self.extractor.extract(
                        page_content, self.query, self.memory.get_entity_ids()
                    )
                    added_page = self.memory.add_facts(
                        page_facts.entities, self.limits.MAX_MEMORY_FACTS
                    )
                    loop_span.set_attribute(
                        "live_rag.memory.facts_added_page", added_page
                    )
                    loop_span.set_attribute(
                        "live_rag.memory.total_facts", self.memory.total_fact_count
                    )
                    if added_page:
                        logger.debug(
                            f"   + Added {added_page} facts from page content."
                        )

                    # === Step 4: Inner Link Discovery & Filtering ===
                    inner_links: List[str] = []
                    raw_html = getattr(self.scraper, "_last_html", None)

                    if raw_html:
                        all_links = await self.scraper.extract_links(
                            raw_html, result.url
                        )
                        loop_span.set_attribute(
                            "live_rag.inner_links.raw_count", len(all_links)
                        )

                        inner_links = await self.link_filter.filter_links(
                            all_links,
                            result.url,
                            self.query,
                            self.limits.MAX_INNER_LINKS_PER_PAGE,
                        )
                    else:
                        loop_span.set_attribute("live_rag.inner_links.raw_count", 0)
                        logger.debug("   ⚠️ No raw HTML available for link extraction.")

                    loop_span.set_attribute(
                        "live_rag.inner_links.filtered_count", len(inner_links)
                    )

                    if inner_links:
                        logger.info(
                            f"   🔗 Found {len(inner_links)} relevant inner links."
                        )
                        completed_via_inner = await self._process_inner_links(
                            inner_links
                        )
                        if completed_via_inner:
                            root_span.set_attribute(
                                "live_rag.stop_reason", "inner_link_sufficiency"
                            )
                            return await self.generator.generate(
                                self.query, self.memory.to_context_string()
                            )
                    else:
                        logger.info("   🔗 No relevant inner links found.")

            # --- Exhausted All Results ---
            logger.warning(
                "⚠️ Exhausted all results without full sufficiency. Generating partial answer."
            )
            root_span.set_attribute("live_rag.stop_reason", "exhausted_results")
            root_span.set_attribute(
                "live_rag.memory.final_fact_count", self.memory.total_fact_count
            )
            return await self.generator.generate(
                self.query, self.memory.to_context_string(), partial=True
            )

    async def _process_inner_links(self, inner_links: List[str]) -> bool:
        """Iterate through inner links with per-link sufficiency checks and memory tracing."""
        for i, link in enumerate(inner_links[: self.limits.MAX_INNER_LINKS_PER_PAGE]):
            with tracer.start_as_current_span(
                f"live_rag.agent.inner_loop.{i}",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    "live_rag.url": link,
                },
            ) as inner_span:
                if not await self._can_scrape():
                    inner_span.set_attribute(
                        "live_rag.stop_reason", "max_scrapes_reached"
                    )
                    return False

                logger.info(f"      ↳ Inner Link [{i + 1}/{len(inner_links)}]: {link}")

                content = await self.scraper.scrape(
                    link, self.limits.SCRAPE_TIMEOUT_SEC
                )
                if not content:
                    logger.warning(f"         ⚠️ Failed to scrape inner link: {link}")
                    inner_span.set_attribute("live_rag.scrape.success", False)
                    continue

                inner_span.set_attribute("live_rag.scrape.success", True)
                inner_span.set_attribute("live_rag.scrape.content_length", len(content))

                # Sufficiency check for inner link
                check = await self.evaluator.evaluate(
                    self.query,
                    self.memory.to_context_string(),
                    content,
                    "inner_link",
                )
                inner_span.set_attribute(
                    "live_rag.sufficiency.status", check.status.value
                )
                inner_span.set_attribute(
                    "live_rag.sufficiency.reasoning", check.reasoning[:500]
                )

                if check.status == SufficiencyStatus.COMPLETE:
                    logger.info("      ✅ Sufficiency met at INNER LINK level.")
                    return True

                # Extract and accumulate facts from inner link
                facts = await self.extractor.extract(
                    content, self.query, self.memory.get_entity_ids()
                )
                added_inner = self.memory.add_facts(
                    facts.entities, self.limits.MAX_MEMORY_FACTS
                )
                inner_span.set_attribute(
                    "live_rag.memory.facts_added_inner", added_inner
                )
                inner_span.set_attribute(
                    "live_rag.memory.total_facts", self.memory.total_fact_count
                )
                if added_inner:
                    logger.debug(
                        f"         + Added {added_inner} facts from inner link."
                    )

        return False

    async def _can_scrape(self) -> bool:
        """Thread-safe scrape budget gate."""
        async with self._lock:
            if self.scrape_count >= self.limits.MAX_TOTAL_SCRAPES:
                return False
            self.scrape_count += 1
            return True
