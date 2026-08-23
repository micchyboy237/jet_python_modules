"""Query classification and decomposition using llm_utils.achat."""

from __future__ import annotations

import logging

from jet.adapters.llama_cpp.llm_utils import achat
from openai import AsyncOpenAI

from .types import QueryAnalysis, QueryIntent

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """\
Analyze the following user query and determine how to best answer it via web search.

## CLASSIFICATION TASK
Determine BOTH complexity AND intent:

### Complexity
- "simple": Can be answered with 1-2 searches (single fact, list lookup, comparison)
- "complex": Requires 3+ searches across different angles, synthesis of multiple sources

### Intent (choose ONE)
- "list": Rankings, "Top N", "Best of", recommendations, curated lists
  Examples: "Top 10 isekai anime 2026", "Best Python frameworks", "Highest rated movies"
- "comparison": Direct comparisons, alternatives, pros/cons
  Examples: "React vs Vue", "Python vs Rust for backend", "iPhone alternatives"
- "factual": Single fact, definition, date, entity attribute
  Examples: "Capital of France", "When was Mushoku Tensei released", "GDP of Japan"
- "complex": Multi-faceted questions requiring decomposition
  Examples: "Compare renewable energy policies in EU vs US 2024"

## REFINED QUERY RULES
- For LIST intent: Add "list" or "ranking" keywords. PRESERVE temporal constraints
  (e.g., "2026", "latest", "recent"). Do NOT replace specific years with "all time".
- For COMPARISON intent: Ensure both subjects are named explicitly.
- For FACTUAL intent: Make the query more specific and searchable.
- For COMPLEX intent: Keep the refined query as the overarching theme.

## SUB-QUERIES
- Only populate for "complex" intent. Leave empty for list/comparison/factual.
- For list queries, do NOT create per-item sub-queries.

User query: {query}

Respond with valid JSON matching the QueryAnalysis schema only."""


class QueryAnalyzer:
    """Classifies queries and decomposes complex ones into sub-queries."""

    ANALYSIS_SEED = 42

    def __init__(self, model: str = "qwen3.5-uncensored:2b"):
        self.model = model

    async def analyze(
        self,
        query: str,
        session_id: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> QueryAnalysis:
        """Analyze a query and return classification + optional decomposition.

        Args:
            session_id: Phoenix session ID for trace correlation.
            client: Shared AsyncOpenAI client to avoid per-call overhead.
        """
        logger.info("🔍 Analyzing query: %r (session=%s)", query[:80], session_id)
        messages = [
            {"role": "user", "content": ANALYSIS_PROMPT.format(query=query)},
        ]
        result = await achat(
            prompt_or_messages=messages,
            model=self.model,
            project_name="react-query-analyzer",
            temperature=0.0,
            max_tokens=1024,
            response_format=QueryAnalysis,
            enable_thinking=False,
            capture_content=True,
            seed=self.ANALYSIS_SEED,
            session_id=session_id,
            client=client,
        )
        if not result.structured or not result.structured.success:
            error = (
                result.structured.error if result.structured else "No structured output"
            )
            logger.warning(
                "⚠️ Query analysis parse failed (%s), defaulting to simple/factual",
                error,
            )
            return QueryAnalysis(
                complexity="simple",
                intent=QueryIntent.UNKNOWN,
                reasoning=f"Parse failed: {error}",
                sub_queries=[],
                refined_query=query,
            )
        analysis: QueryAnalysis = result.structured.parsed
        logger.info(
            "✅ Query classified as %s (intent=%s): %d sub-queries, refined=%r",
            analysis.complexity.value,
            analysis.intent.value,
            len(analysis.sub_queries),
            analysis.refined_query[:60],
        )
        return analysis
