"""Core deep research orchestrator using existing jet_python_modules primitives."""

from __future__ import annotations

import logging
from typing import Any

from jet.adapters.llama_cpp.chunking_utils import chunk_markdown_hierarchy_with_data
from jet.adapters.llama_cpp.config import EMBED_MODEL, LLM_MODEL, RERANK_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.llm_utils import achat
from jet.adapters.llama_cpp.model_utils import (
    get_loaded_models,
    get_model_ctx_embd_size,
)
from jet.adapters.llama_cpp.rerank_utils import rerank
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity
from jet.adapters.llama_cpp.token_utils import count_chat_tokens, count_tokens
from jet.scrapers.playwright_utils import scrape_url
from jet.search.searxng import async_search_searxng
from playwright.async_api import async_playwright

from .models import (
    ExtractedLinks,
    GroundingResult,
    QueryPlan,
    SubQuery,
    SynthesizedAnswer,
)
from .priority_queue import AgentPriorityQueue, QueueItem

logger = logging.getLogger(__name__)

# Thresholds
SNIPPET_SUFFICIENCY_THRESHOLD = 0.7
LINK_SCORE_THRESHOLD = 0.3
MAX_SERP_RESULTS = 10
MAX_QUEUE_SIZE = 30
GENERATION_RESERVE_RATIO = 0.25

PLANNING_PROMPT = """\
You are a search query planner. Given the user query, determine if it needs \
decomposition into sub-queries, and rewrite it for better search results.

Rules:
- Simple factual queries: needs_decomposition=false, just rewrite for clarity.
- Multi-part/comparison/temporal queries: decompose into self-contained sub-queries.
- Each sub-query must be independently searchable (resolve pronouns/references).
- Assign unique IDs like sq_1, sq_2, etc.
- Set depends_on only when a sub-query requires another's result.

User query: {query}
"""

GROUNDING_PROMPT = """\
Verify if the following page content contains evidence answering the sub-query.

Sub-query: {sub_query}
Page content (truncated):
{content}

Be strict: only mark is_grounded=true if there is explicit, verifiable evidence.
"""

LINK_EXTRACTION_PROMPT = """\
From the page content below, extract links that might help answer these \
UNANSWERED sub-queries: {unanswered_ids}

Page content (markdown):
{content}

Only include links whose anchor text is semantically related to an unanswered sub-query. \
Return absolute URLs. If no relevant links exist, return an empty list.
"""

SYNTHESIS_SYSTEM_PROMPT = """\
Synthesize a complete answer to the original query using ONLY the verified evidence below.

Rules:
- Every factual claim MUST have a citation with exact quote and source URL.
- List any sub-queries that could not be answered in unresolved_sub_queries.
- Set confidence based on coverage: 1.0=all answered, 0.5=partial, 0.0=none.
- Never fabricate information not present in the evidence.
"""


class AgenticRAG:
    """Orchestrates adaptive retrieval with verification using jet primitives."""

    def __init__(
        self,
        llm_model: str | None = None,
        embed_model: str | None = None,
        rerank_model: str | None = None,
        max_depth: int = 2,
        snippet_threshold: float = SNIPPET_SUFFICIENCY_THRESHOLD,
    ):
        self.llm_model = llm_model or LLM_MODEL
        self.embed_model = embed_model or EMBED_MODEL
        self.rerank_model = rerank_model or RERANK_MODEL
        self.max_depth = max_depth
        self.snippet_threshold = snippet_threshold
        self.synthesis_token_budget: int = 6000

    async def initialize(self) -> None:
        """Validate models are loaded and compute dynamic token budget."""
        self._validate_models_loaded()
        self._init_token_budget()

    def _validate_models_loaded(self) -> None:
        """Fail fast if required models are not loaded on the server."""
        try:
            loaded = get_loaded_models()
            loaded_ids = {
                alias
                for m in loaded.get("data", [])
                for alias in m.get("aliases", [m["id"]])
            }
            missing = []
            for label, model_key in [
                ("LLM", self.llm_model),
                ("Embed", self.embed_model),
            ]:
                if model_key and model_key not in loaded_ids:
                    missing.append(f"{label}={model_key}")
            if missing:
                logger.warning(
                    f"[AgenticRAG] Models not found in loaded set: {missing}. "
                    f"Loaded: {sorted(loaded_ids)[:10]}... "
                    "Proceeding anyway (server may lazy-load)."
                )
        except Exception as e:
            logger.warning(f"[AgenticRAG] Could not validate loaded models: {e}")

    def _init_token_budget(self) -> None:
        """Derive synthesis token budget from actual model n_ctx."""
        try:
            ctx_info = get_model_ctx_embd_size(self.llm_model)
            n_ctx = ctx_info.get("ctx", 0)
            if n_ctx > 0:
                self.synthesis_token_budget = int(
                    n_ctx * (1 - GENERATION_RESERVE_RATIO)
                )
                logger.info(
                    f"[AgenticRAG] Token budget: {self.synthesis_token_budget} "
                    f"(n_ctx={n_ctx}, reserve={GENERATION_RESERVE_RATIO:.0%})"
                )
            else:
                logger.warning(
                    f"[AgenticRAG] n_ctx=0 for {self.llm_model}, "
                    f"using fallback budget={self.synthesis_token_budget}"
                )
        except ValueError as e:
            logger.warning(f"[AgenticRAG] Cannot get n_ctx: {e}. Using fallback.")
        except Exception as e:
            logger.warning(
                f"[AgenticRAG] Token budget init failed: {e}. Using fallback."
            )

    async def run(self, query: str) -> dict[str, Any]:
        """Execute the full deep research pipeline with managed browser context."""
        await self.initialize()
        logger.info(f"[AgenticRAG] Starting pipeline for: {query[:80]}...")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            try:
                result = await self._run_pipeline(query, context)
            finally:
                await context.close()
                await browser.close()

        return result

    async def _run_pipeline(self, query: str, context) -> dict[str, Any]:
        """Internal pipeline logic that receives an active browser context."""
        # Phase A: Planning
        plan = await self._plan_query(query)
        logger.info(
            f"[AgenticRAG] Plan: decompose={plan.needs_decomposition}, "
            f"sub_queries={len(plan.sub_queries)}"
        )

        active_queries = (
            [sq.text for sq in plan.sub_queries]
            if plan.needs_decomposition and plan.sub_queries
            else [plan.rewritten_query]
        )
        sub_query_map = (
            {sq.id: sq for sq in plan.sub_queries} if plan.needs_decomposition else {}
        )

        # Phase B+C: Retrieval loop
        evidence_store: list[dict[str, Any]] = []
        answered_sq_ids: set[str] = set()
        queue = AgentPriorityQueue(max_depth=self.max_depth)

        for q in active_queries:
            sq_id = next(
                (sid for sid, sq in sub_query_map.items() if sq.text == q),
                "root",
            )
            serp_results = await async_search_searxng(
                query=q, count=MAX_SERP_RESULTS, use_cache=True
            )
            if not serp_results:
                logger.warning(f"[AgenticRAG] No SERP results for: {q[:60]}")
                continue

            snippets = [r["content"] for r in serp_results if r.get("content")]
            if snippets:
                reranked = rerank(
                    query=q,
                    documents=snippets,
                    top_n=3,
                    model=self.rerank_model,
                    method="auto",
                )
                best_score = reranked[0]["score"] if reranked else 0.0
                if best_score >= self.snippet_threshold:
                    logger.info(
                        f"[AgenticRAG] Snippets sufficient (score={best_score:.3f}) "
                        f"for: {q[:60]}"
                    )
                    for rr in reranked:
                        evidence_store.append(
                            {
                                "text": rr["text"],
                                "source_url": serp_results[rr["index"]].get("url", ""),
                                "sub_query_id": sq_id,
                                "section_header": "SERP Snippet",
                            }
                        )
                    answered_sq_ids.add(sq_id)
                    continue

            logger.info(
                f"[AgenticRAG] Queuing {min(3, len(serp_results))} URLs for: {q[:60]}"
            )
            items = [
                QueueItem(
                    sort_key=-result.get("score", 0.0),
                    url=result["url"],
                    depth=0,
                    sub_query_id=sq_id,
                )
                for result in serp_results[:3]
            ]
            await queue.push_many(items)

        # Deep navigation loop
        while not await queue.is_empty():
            all_answered = sub_query_map and answered_sq_ids >= set(
                sub_query_map.keys()
            )
            if all_answered:
                logger.info(
                    "[AgenticRAG] All sub-queries answered, stopping navigation"
                )
                break
            if queue.size > MAX_QUEUE_SIZE:
                logger.warning(
                    f"[AgenticRAG] Queue exceeded {MAX_QUEUE_SIZE}, trimming"
                )
                while queue.size > MAX_QUEUE_SIZE:
                    await queue.pop()

            item = await queue.pop()
            if item is None:
                break

            logger.info(
                f"[AgenticRAG] Navigating (depth={item.depth}): {item.url[:80]}"
            )
            scrape_result = await scrape_url(
                context=context,
                url=item.url,
                scroll_strategy="until_stable",
                use_cache=True,
                with_screenshot=False,
            )
            if scrape_result["status"] != "completed" or not scrape_result.get("html"):
                logger.warning(f"[AgenticRAG] Scrape failed: {item.url[:80]}")
                continue

            markdown_content = self._html_to_markdown(scrape_result["html"])
            if not markdown_content.strip():
                continue

            grounding = await self._check_grounding(
                markdown_content, item.sub_query_id, sub_query_map
            )
            if not grounding.is_grounded:
                logger.debug(f"[AgenticRAG] Not grounded: {item.url[:80]}")
                continue

            logger.info(f"[AgenticRAG] ✓ Grounded: {item.url[:80]}")
            evidence_store.append(
                {
                    "text": grounding.relevant_section or markdown_content[:2000],
                    "source_url": item.url,
                    "sub_query_id": item.sub_query_id,
                    "section_header": "Verified Page Content",
                }
            )
            answered_sq_ids.add(item.sub_query_id)

            unanswered = set(sub_query_map.keys()) - answered_sq_ids
            if unanswered and item.depth < self.max_depth:
                links = await self._extract_links(markdown_content, unanswered)
                if links.links:
                    scored_items = await self._score_and_rank_links(
                        links, query, item.depth + 1
                    )
                    added = await queue.push_many(scored_items)
                    logger.info(
                        f"[AgenticRAG] Added {added}/{len(links.links)} links to queue"
                    )

        # Phase E: Synthesis
        result = await self._synthesize(
            query, evidence_store, answered_sq_ids, sub_query_map
        )
        logger.info(
            f"[AgenticRAG] Complete: confidence={result.confidence:.2f}, "
            f"citations={len(result.citations)}, "
            f"unresolved={len(result.unresolved_sub_queries)}"
        )
        return {
            "answer": result.answer,
            "citations": [c.model_dump() for c in result.citations],
            "unresolved_sub_queries": result.unresolved_sub_queries,
            "confidence": result.confidence,
            "evidence_count": len(evidence_store),
        }

    async def _plan_query(self, query: str) -> QueryPlan:
        result = await achat(
            PLANNING_PROMPT.format(query=query),
            model=self.llm_model,
            response_format=QueryPlan,
            temperature=0.1,
        )
        if result.structured and result.structured.success:
            return result.structured.parsed
        logger.warning("[AgenticRAG] Planning failed, using raw query")
        return QueryPlan(needs_decomposition=False, rewritten_query=query)

    async def _check_grounding(
        self, content: str, sq_id: str, sq_map: dict[str, SubQuery]
    ) -> GroundingResult:
        sq_text = sq_map[sq_id].text if sq_id in sq_map else "the original query"
        truncated = content[:4000]
        result = await achat(
            GROUNDING_PROMPT.format(sub_query=sq_text, content=truncated),
            model=self.llm_model,
            response_format=GroundingResult,
            temperature=0.0,
        )
        if result.structured and result.structured.success:
            return result.structured.parsed
        return GroundingResult(is_grounded=False, evidence_summary="LLM parsing failed")

    async def _extract_links(
        self, content: str, unanswered_ids: set[str]
    ) -> ExtractedLinks:
        result = await achat(
            LINK_EXTRACTION_PROMPT.format(
                unanswered_ids=", ".join(unanswered_ids),
                content=content[:4000],
            ),
            model=self.llm_model,
            response_format=ExtractedLinks,
            temperature=0.1,
        )
        if result.structured and result.structured.success:
            return result.structured.parsed
        return ExtractedLinks(links=[])

    async def _score_and_rank_links(
        self, links: ExtractedLinks, original_query: str, depth: int
    ) -> list[QueueItem]:
        if not links.links:
            return []
        anchors = [lk.anchor_text for lk in links.links]
        try:
            anchor_embs = embed(anchors, model=self.embed_model, return_format="numpy")
            query_emb = embed(
                original_query, model=self.embed_model, return_format="numpy"
            )
            scores = [float(cosine_similarity(query_emb, ae)) for ae in anchor_embs]
        except Exception as e:
            logger.warning(f"[AgenticRAG] Link embedding failed: {e}")
            return []

        items = []
        for link, score in zip(links.links, scores):
            if score >= LINK_SCORE_THRESHOLD:
                items.append(
                    QueueItem(
                        sort_key=-score,
                        url=link.url,
                        depth=depth,
                        sub_query_id=link.target_sub_query_id or "",
                    )
                )
        return sorted(items, key=lambda x: x.sort_key)

    async def _synthesize(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        answered_ids: set[str],
        sq_map: dict[str, SubQuery],
    ) -> SynthesizedAnswer:
        if not evidence:
            return SynthesizedAnswer(
                answer="I could not find sufficient evidence to answer this query.",
                unresolved_sub_queries=list(sq_map.keys()),
                confidence=0.0,
            )

        evidence_texts = [e["text"] for e in evidence]
        chunks = chunk_markdown_hierarchy_with_data(
            markdown_text=evidence_texts,
            chunk_size=512,
            overlap_strategy="sentence",
            overlap_size=1,
            show_progress=False,
        )

        selected_chunks = []
        system_msg = {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT}

        for chunk in chunks:
            prospective_user_content = self._build_synthesis_user_content(
                query, selected_chunks + [chunk], evidence, sq_map, answered_ids
            )
            prospective_messages = [
                system_msg,
                {"role": "user", "content": prospective_user_content},
            ]
            try:
                msg_tokens = count_chat_tokens(
                    prospective_messages, model=self.llm_model
                )["input_tokens"]
            except Exception:
                msg_tokens = count_tokens(
                    prospective_user_content, model=self.llm_model
                )

            if msg_tokens > self.synthesis_token_budget:
                logger.info(
                    f"[AgenticRAG] Token budget reached at {msg_tokens}/{self.synthesis_token_budget}"
                )
                break
            selected_chunks.append(chunk)

        user_content = self._build_synthesis_user_content(
            query, selected_chunks, evidence, sq_map, answered_ids
        )
        messages = [system_msg, {"role": "user", "content": user_content}]

        result = await achat(
            messages,
            model=self.llm_model,
            response_format=SynthesizedAnswer,
            temperature=0.2,
            max_tokens=int(self.synthesis_token_budget * GENERATION_RESERVE_RATIO),
        )
        if result.structured and result.structured.success:
            return result.structured.parsed

        logger.warning(
            "[AgenticRAG] Synthesis structured output failed, using fallback"
        )
        unresolved = [sq_map[sid].text for sid in set(sq_map.keys()) - answered_ids]
        return SynthesizedAnswer(
            answer=result.content or "Synthesis failed.",
            unresolved_sub_queries=unresolved,
            confidence=0.3,
        )

    @staticmethod
    def _build_synthesis_user_content(
        query: str,
        chunks: list[dict],
        evidence: list[dict[str, Any]],
        sq_map: dict[str, SubQuery],
        answered_ids: set[str],
    ) -> str:
        evidence_formatted = "\n\n---\n\n".join(
            f"[Source: {evidence[i]['source_url']}]\n{c['content']}"
            for i, c in enumerate(chunks)
            if i < len(evidence)
        )
        parts = [
            f"Original query: {query}",
            f"\nVerified evidence chunks:\n{evidence_formatted}",
        ]
        unresolved = [sq_map[sid].text for sid in set(sq_map.keys()) - answered_ids]
        if unresolved:
            parts.append(f"\nUnresolved sub-queries: {', '.join(unresolved)}")
        return "\n".join(parts)

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """Lightweight HTML→Markdown conversion. Prefers trafilatura if installed."""
        try:
            import trafilatura

            md = trafilatura.extract(html, output_format="markdown")
            return md or ""
        except ImportError:
            pass
        try:
            from markdownify import markdownify

            return markdownify(html, strip=["img", "script", "style", "nav", "footer"])
        except ImportError:
            pass
        import re

        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()
