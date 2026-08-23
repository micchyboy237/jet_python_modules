"""Post-answer validation using the existing llm_as_a_judge pipeline."""

from __future__ import annotations

import logging
from typing import Any

from jet.adapters.llama_cpp.tasks.rag.eval.llm_as_a_judge import (
    RAGEvaluator,
)

logger = logging.getLogger(__name__)


class PostAnswerValidator:
    """Validates ReAct agent answers using the existing eval pipeline."""

    def __init__(self, model: str = "qwen3.5-uncensored:2b"):
        self.evaluator = RAGEvaluator(model=model)

    async def validate(
        self,
        query: str,
        response: str,
        contexts: list[str],
    ) -> dict[str, Any]:
        """Run production async evaluation on the agent's answer."""
        logger.info(
            "🔍 Validating agent answer (%d chars, %d contexts)",
            len(response),
            len(contexts),
        )

        result = await self.evaluator.evaluate_production_async(
            query=query,
            contexts=contexts,
            response=response,
        )

        return {
            "faithfulness": result.faithfulness,
            "hallucination_rate": result.hallucination_rate,
            "answer_relevancy": result.answer_relevancy,
            "has_critical_failure": result.has_critical_failure,
            "total_eval_tokens": result.total_eval_tokens,
        }
