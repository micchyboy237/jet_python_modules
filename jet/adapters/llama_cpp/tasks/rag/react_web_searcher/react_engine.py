"""ReAct loop engine using llm_utils.achat with tool_registry."""

from __future__ import annotations

import logging

from jet.adapters.llama_cpp.llm_utils import achat

from .query_analyzer import QueryAnalyzer
from .tools import get_tool_definitions, get_tool_registry
from .types import AgentStep, FinalAnswer, QueryComplexity
from .validator import PostAnswerValidator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a thorough web research agent. Your goal is to answer the user's question \
accurately and completely using web search.

You have access to three tools:
1. searxng_search - Search the web for information
2. read_url - Read the full content of a specific web page
3. synthesize - Combine your findings into a final answer

Follow this process:
1. Think about what information you need to answer the question
2. Search for that information using searxng_search
3. If snippets are insufficient, use read_url on promising results
4. For complex questions, search for each sub-query separately
5. When you have enough information, call synthesize to produce the final answer
6. Always cite your sources

IMPORTANT:
- Do NOT make up information. Only use what you find via search.
- If you cannot find sufficient information after multiple searches, say so.
- Call synthesize ONLY when ready to produce the final answer.
- Be thorough: for complex questions, search each sub-query."""


class ReactEngine:
    """Orchestrates the full ReAct search pipeline."""

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
        self.tool_registry = get_tool_registry()

    async def search(self, query: str) -> FinalAnswer:
        """Run the full ReAct web search pipeline for a query."""
        logger.info("🚀 Starting ReAct search for: %r", query[:80])

        # Step 1: Analyze and optionally decompose
        analysis = await self.analyzer.analyze(query)

        # Step 2: Build initial message based on complexity
        if analysis.complexity == QueryComplexity.COMPLEX and analysis.sub_queries:
            user_content = (
                f"Original Question: {query}\n\n"
                f"This is a complex question. Break it down and search for each sub-query:\n"
                + "\n".join(f"- {sq}" for sq in analysis.sub_queries)
                + f"\n\nRefined query: {analysis.refined_query}\n\n"
                f"Search each sub-query, gather findings, then synthesize a complete answer."
            )
        else:
            user_content = (
                f"Question: {analysis.refined_query}\n\n"
                f"Search for information and provide a complete, accurate answer."
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Step 3: Run ReAct loop via llm_utils.achat with tool_registry
        logger.info(
            "🔄 Starting ReAct loop (max_iterations=%d, tools=%d)",
            self.max_iterations,
            len(self.tool_registry),
        )

        result = await achat(
            prompt_or_messages=messages,
            model=self.model,
            project_name="react-web-searcher",
            temperature=0.3,
            max_tokens=4096,
            tools=self.tool_definitions,
            tool_choice="auto",
            tool_registry=self.tool_registry,
            max_tool_rounds=self.max_iterations,
            enable_thinking=False,
            capture_content=True,
        )

        # Extract steps from tool calls in the result
        steps: list[AgentStep] = []
        for tc in result.tool_calls:
            steps.append(
                AgentStep(
                    thought="",  # Thoughts are embedded in assistant content
                    action=tc.name,
                    action_input=tc.arguments,
                    observation=str(
                        tc.arguments.get("query", tc.arguments.get("url", ""))
                    )[:200],
                )
            )

        total_tokens = result.usage.get("total_tokens", 0) if result.usage else 0
        answer_text = result.content or ""

        logger.info(
            "✅ ReAct loop complete: %d steps, %d tokens, %d chars",
            len(steps),
            total_tokens,
            len(answer_text),
        )

        # Step 4: Post-answer validation
        eval_result = None
        confidence = "high"
        if self.enable_validation and self.validator and answer_text:
            # Collect all search snippets as context for validation
            search_contexts = [
                s.observation for s in steps if s.action == "searxng_search"
            ]
            if search_contexts:
                eval_result = await self.validator.validate(
                    query=query,
                    response=answer_text,
                    contexts=search_contexts,
                )
                if eval_result.get("has_critical_failure"):
                    confidence = "low"
                    logger.warning(
                        "⚠️ Validation flagged critical failure: faith=%.3f halluc=%.3f",
                        eval_result.get("faithfulness", -1),
                        eval_result.get("hallucination_rate", -1),
                    )
                else:
                    logger.info(
                        "✅ Validation passed: faith=%.3f halluc=%.3f relevancy=%.3f",
                        eval_result.get("faithfulness", -1),
                        eval_result.get("hallucination_rate", -1),
                        eval_result.get("answer_relevancy", -1),
                    )

        return FinalAnswer(
            answer=answer_text,
            sources=[],  # Sources extracted from tool call observations
            steps=steps,
            confidence=confidence,
            total_tokens=total_tokens,
            eval_result=eval_result,
        )
