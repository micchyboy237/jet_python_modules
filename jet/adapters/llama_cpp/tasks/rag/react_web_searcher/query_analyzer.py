"""Query classification and decomposition using llm_utils.achat."""

from __future__ import annotations

import logging

from jet.adapters.llama_cpp.llm_utils import achat
from openai import AsyncOpenAI

from .types import QueryAnalysis

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT = """\
Analyze the following user query and determine how to best answer it via web search.

If the query asks a single factual question that can be answered with one search, \
classify as "simple" and provide a refined search query.

If the query is multi-faceted, comparative, temporal, or requires synthesizing \
information from multiple angles, classify as "complex" and decompose into 2-5 \
focused sub-queries that together cover the full scope.

Always provide a refined_query that improves searchability over the raw input.

User query: {query}

Respond with valid JSON matching the QueryAnalysis schema only."""


class QueryAnalyzer:
    """Classifies queries and decomposes complex ones into sub-queries."""

    ANALYSIS_SEED = 42  # Fixed seed for reproducible decomposition

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
            seed=self.ANALYSIS_SEED,  # Reproducible decomposition
            session_id=session_id,  # Correlated traces
            client=client,  # Client reuse
        )

        if not result.structured or not result.structured.success:
            error = (
                result.structured.error if result.structured else "No structured output"
            )
            logger.warning(
                "⚠️ Query analysis parse failed (%s), defaulting to simple", error
            )
            return QueryAnalysis(
                complexity="simple",
                reasoning=f"Parse failed: {error}",
                sub_queries=[],
                refined_query=query,
            )

        analysis: QueryAnalysis = result.structured.parsed
        logger.info(
            "✅ Query classified as %s: %d sub-queries, refined=%r",
            analysis.complexity.value,
            len(analysis.sub_queries),
            analysis.refined_query[:60],
        )
        return analysis
