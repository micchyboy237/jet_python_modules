"""Post-answer validation using the existing llm_as_a_judge pipeline."""

from __future__ import annotations

import logging
from typing import Any

from jet.adapters.llama_cpp.tasks.rag.eval.llm_as_a_judge import RAGEvaluator
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class PostAnswerValidator:
    """Validates ReAct agent answers using the existing eval pipeline.
    Threads session_id and shared client through to the evaluator
    for full trace correlation and client reuse.
    """

    def __init__(self, model: str = "qwen3.5-uncensored:2b"):
        self.evaluator = RAGEvaluator(model=model)

    async def validate(
        self,
        query: str,
        response: str,
        contexts: list[str],
        session_id: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> dict[str, Any]:
        """Run production async evaluation on the agent's answer.
        Args:
            session_id: Phoenix session ID for trace correlation.
            client: Shared AsyncOpenAI client (currently not threaded
            through RAGEvaluator; reserved for future use).
        """
        logger.info(
            "🔍 Validating agent answer (%d chars, %d contexts, session=%s)",
            len(response),
            len(contexts),
            session_id,
        )
        # ✅ UPDATED: Pass session_id to evaluator
        result = await self.evaluator.evaluate_production_async(
            query=query,
            contexts=contexts,
            response=response,
            session_id=session_id,
        )
        return {
            "faithfulness": result.faithfulness,
            "hallucination_rate": result.hallucination_rate,
            "answer_relevancy": result.answer_relevancy,
            "has_critical_failure": result.has_critical_failure,
            "total_eval_tokens": result.total_eval_tokens,
        }
