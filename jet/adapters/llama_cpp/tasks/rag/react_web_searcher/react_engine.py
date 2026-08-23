"""ReAct loop engine using llm_utils.achat with full feature utilization.

✅ IMPROVEMENTS:
- Validation feedback loop: re-searches on critical failure instead of returning bad answer
- Source population: extracts source URLs from AgentStep metadata for citation
- Re-search cap: prevents infinite loops from persistent validation failures
"""

from __future__ import annotations

import logging
import uuid

from jet.adapters.llama_cpp.factory import get_async_llm_client
from jet.adapters.llama_cpp.llm_utils import achat
from openai import AsyncOpenAI

from .query_analyzer import QueryAnalyzer
from .tools import get_tool_definitions, get_tool_registry
from .types import AgentStep, FinalAnswer, QueryComplexity, SearchResult
from .validator import PostAnswerValidator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a thorough web research agent. Your goal is to answer the user's question \
accurately and completely using web search.

You have access to three tools:
1. searxng_search - Search the web for information
2. read_url - Read the full content of a specific web page (supports focused extraction via 'query' param)
3. synthesize - Combine your findings into a final answer

## INTENT-AWARE SEARCH STRATEGY
Before searching, classify the query intent and follow the matching strategy:

### LIST / RANKING / "TOP N" QUERIES
(e.g., "Top 10 isekai anime", "Best programming languages 2026", "Highest grossing films")
- ✅ DO: Search for CURATED LISTS or RANKINGS (e.g., "best isekai anime list 2026")
- ✅ DO: Use read_url on the best list page to extract the full structured ranking
- ❌ NEVER: Search for individual items/titles one-by-one
- ❌ NEVER: Decompose a list query into per-entity sub-queries
- If the first list is incomplete, search for ANOTHER LIST, not individual items

### COMPARISON QUERIES
(e.g., "React vs Vue", "Python vs Rust for backend")
- ✅ DO: Search for direct comparison articles or benchmarks
- ✅ DO: Use read_url on comparison pages for structured data
- Only search individual subjects if no comparison source exists

### FACTUAL / SIMPLE QUERIES
(e.g., "Capital of France", "When was X released")
- Search directly for the fact
- Use read_url if snippets lack detail or citations

### COMPLEX / MULTI-FACETED QUERIES
(e.g., "Compare renewable energy policies in EU vs US 2024")
- Decompose into 2-5 focused sub-queries
- Search each sub-query separately
- Use read_url on promising results for each sub-query

## GENERAL RULES
1. ⚠️ CRITICAL: Search snippets are summaries and often lack detail.
   You MUST use read_url on at least 1-2 promising results to verify facts
   and gather comprehensive details before synthesizing.
2. When calling read_url, ALWAYS pass the original query or sub-query
   to focus extraction on relevant sections.
3. When you have enough VERIFIED information, call synthesize to produce the final answer.
4. Always cite your sources.
5. Do NOT make up information. Only use what you find via search AND verify via read_url.
6. If you cannot find sufficient information after multiple searches and page reads, say so.
7. Call synthesize ONLY when ready to produce the final answer."""

# Maximum number of re-search attempts triggered by validation failures
_MAX_VALIDATION_RETRIES = 2


class ReactEngine:
    """Orchestrates the full ReAct search pipeline.

    Utilizes all applicable llm_utils.achat features:
    - session_id: All calls within one search share a Phoenix conversation thread
    - seed: Analyzer uses fixed seed for reproducible decomposition
    - stop: Prevents runaway generation beyond current turn
    - client: Single shared AsyncOpenAI client avoids per-call overhead
    - finish_reason: Detects truncation on final answer
    - step_tracker: Mutable list passed to tool wrappers for accurate step counting

    ✅ NEW: Validation feedback loop re-searches on critical failure.
    ✅ NEW: Source population from AgentStep metadata.
    """

    def __init__(
        self,
        model: str = "qwen3.5-uncensored:2b",
        max_iterations: int = 10,
        enable_validation: bool = True,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.enable_validation = enable_validation
        self.analyzer = QueryAnalyzer(model=model)
        self.validator = PostAnswerValidator(model=model) if enable_validation else None
        self.tool_definitions = get_tool_definitions()
        self._client: AsyncOpenAI = get_async_llm_client()

    async def search(self, query: str) -> FinalAnswer:
        """Run the full ReAct web search pipeline for a query."""
        logger.info("🚀 Starting ReAct search for: %r", query[:80])

        session_id = f"react-{uuid.uuid4().hex[:12]}"
        logger.debug("🧵 Session ID: %s", session_id)

        # Step 1: Analyze query complexity
        logger.debug("📋 Step 1: Analyzing query complexity")
        analysis = await self.analyzer.analyze(
            query, session_id=session_id, client=self._client
        )

        logger.info(
            "📋 Analysis result: complexity=%s, sub_queries=%d, refined=%r",
            analysis.complexity.value,
            len(analysis.sub_queries),
            analysis.refined_query[:60],
        )

        if analysis.complexity == QueryComplexity.COMPLEX and analysis.sub_queries:
            user_content = (
                f"Original Question: {query}\n"
                f"This is a complex question. Break it down and search for each sub-query:\n"
                + "\n".join(f"- {sq}" for sq in analysis.sub_queries)
                + f"\nRefined query: {analysis.refined_query}\n"
                f"Search each sub-query, gather findings, then synthesize a complete answer."
            )
            logger.debug(
                "📝 Complex query: %d sub-queries injected into prompt",
                len(analysis.sub_queries),
            )
        else:
            user_content = (
                f"Question: {analysis.refined_query}\n"
                f"Search for information and provide a complete, accurate answer."
            )
            logger.debug("📝 Simple query: using refined query directly")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        steps: list[AgentStep] = []
        bound_registry = get_tool_registry(step_tracker=steps)

        logger.info(
            "🔄 Starting ReAct loop (max_iterations=%d, tools=%d, session=%s)",
            self.max_iterations,
            len(bound_registry),
            session_id,
        )

        # ✅ Main ReAct loop with validation feedback
        validation_retries = 0
        answer_text = ""
        truncated = False
        total_tokens = 0
        eval_result = None
        confidence = "high"

        while True:
            result = await achat(
                prompt_or_messages=messages,
                model=self.model,
                project_name="react-web-searcher",
                temperature=0.3,
                max_tokens=4096,
                tools=self.tool_definitions,
                tool_choice="auto",
                tool_registry=bound_registry,
                max_tool_rounds=self.max_iterations,
                enable_thinking=False,
                capture_content=True,
                session_id=session_id,
                stop=["Observation:", "Thought:"],
                client=self._client,
            )

            truncated = result.finish_reason == "length"
            total_tokens = result.usage.get("total_tokens", 0) if result.usage else 0
            answer_text = result.content or ""

            logger.info(
                "✅ ReAct loop complete: %d steps, %d tokens, %d chars, truncated=%s, finish=%s",
                len(steps),
                total_tokens,
                len(answer_text),
                truncated,
                result.finish_reason,
            )

            for i, step in enumerate(steps, 1):
                logger.debug(
                    "   Step %d/%d: %s(%s) → %d chars observation",
                    i,
                    len(steps),
                    step.action,
                    list(step.action_input.keys()),
                    len(step.observation),
                )

            if truncated:
                logger.warning(
                    "⚠️ ReAct loop truncated at max_tokens=4096. "
                    "Final answer may be incomplete."
                )

            # ✅ Validation with feedback loop
            eval_result = None
            confidence = "high"

            if self.enable_validation and self.validator and answer_text:
                validation_contexts = [
                    s.observation
                    for s in steps
                    if s.action in ("searxng_search", "read_url")
                ]

                logger.debug(
                    "📋 Validation contexts: %d observations collected (%d search, %d read_url)",
                    len(validation_contexts),
                    sum(1 for s in steps if s.action == "searxng_search"),
                    sum(1 for s in steps if s.action == "read_url"),
                )

                if validation_contexts:
                    logger.debug("🔍 Running post-answer validation")
                    eval_result = await self.validator.validate(
                        query=query,
                        response=answer_text,
                        contexts=validation_contexts,
                        session_id=session_id,
                        client=self._client,
                    )

                    logger.debug(
                        "🔍 Validation complete: faith=%.3f halluc=%.3f relevancy=%.3f critical=%s",
                        eval_result.get("faithfulness", -1),
                        eval_result.get("hallucination_rate", -1),
                        eval_result.get("answer_relevancy", -1),
                        eval_result.get("has_critical_failure", False),
                    )

                    if eval_result.get("has_critical_failure"):
                        confidence = "low"
                        logger.warning(
                            "⚠️ Validation flagged critical failure: faith=%.3f halluc=%.3f",
                            eval_result.get("faithfulness", -1),
                            eval_result.get("hallucination_rate", -1),
                        )

                        # ✅ NEW: Re-search on critical failure (with cap)
                        if validation_retries < _MAX_VALIDATION_RETRIES:
                            validation_retries += 1
                            logger.info(
                                "🔄 Triggering re-search (attempt %d/%d) due to validation failure",
                                validation_retries,
                                _MAX_VALIDATION_RETRIES,
                            )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"Your previous answer failed validation:\n"
                                        f"- Faithfulness: {eval_result.get('faithfulness', 'N/A')}\n"
                                        f"- Hallucination rate: {eval_result.get('hallucination_rate', 'N/A')}\n"
                                        f"- Answer relevancy: {eval_result.get('answer_relevancy', 'N/A')}\n"
                                        f"Search for additional evidence and synthesize again. "
                                        f"Focus on grounding claims in retrieved sources."
                                    ),
                                }
                            )
                            continue  # ← Re-enter ReAct loop
                        else:
                            logger.warning(
                                "⚠️ Max validation retries (%d) exhausted, returning best-effort answer",
                                _MAX_VALIDATION_RETRIES,
                            )
                    else:
                        logger.info(
                            "✅ Validation passed: faith=%.3f halluc=%.3f relevancy=%.3f",
                            eval_result.get("faithfulness", -1),
                            eval_result.get("hallucination_rate", -1),
                            eval_result.get("answer_relevancy", -1),
                        )
                else:
                    logger.warning(
                        "⚠️ No search/read_url observations found in %d steps — skipping validation",
                        len(steps),
                    )
            elif not self.enable_validation:
                logger.debug("⏭️ Validation disabled, skipping")
            elif not answer_text:
                logger.debug("⏭️ Empty answer, skipping validation")

            # Exit the validation retry loop
            break

        # ✅ NEW: Populate sources from AgentStep metadata
        sources: list[SearchResult] = []
        seen_urls: set[str] = set()
        for step in steps:
            if step.source_url and step.source_url not in seen_urls:
                seen_urls.add(step.source_url)
                sources.append(
                    SearchResult(
                        title=step.source_title or "",
                        url=step.source_url,
                        snippet=step.observation[:200] if step.observation else "",
                        engine="web",
                        score=0.0,
                    )
                )

        final = FinalAnswer(
            answer=answer_text,
            sources=sources,
            steps=steps,
            confidence=confidence,
            total_tokens=total_tokens,
            truncated=truncated,
            eval_result=eval_result,
        )

        logger.info(
            "🏁 Search complete: confidence=%s, steps=%d, tokens=%d, truncated=%s, "
            "validated=%s, sources=%d, validation_retries=%d",
            final.confidence,
            len(final.steps),
            final.total_tokens,
            final.truncated,
            final.eval_result is not None,
            len(final.sources),
            validation_retries,
        )
        return final
