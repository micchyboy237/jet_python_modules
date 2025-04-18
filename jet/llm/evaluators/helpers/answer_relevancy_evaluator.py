import re
from jet.llm.evaluators.helpers.base import EvaluationResult, default_parser_function, parse_feedback
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import Document
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.evaluation import AnswerRelevancyEvaluator as BaseAnswerRelevancyEvaluator
from typing import Any, Callable, Optional, Sequence

_DEFAULT_SCORE_THRESHOLD = 2.0


class AnswerRelevancyEvaluator(BaseAnswerRelevancyEvaluator):
    """Enhanced Answer Relevancy Evaluator.

    Extends the base AnswerRelevancyEvaluator and introduces additional logic
    for determining if the response is passing or not.
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        raise_error: bool = False,
        eval_template: str | BasePromptTemplate | None = None,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        parser_function: Callable[[
            str], tuple[Optional[float], Optional[str]]] = default_parser_function,
        passing_score_threshold: float = 1.0  # Custom passing score threshold
    ) -> None:
        """Init params."""
        super().__init__(
            llm=llm,
            raise_error=raise_error,
            eval_template=eval_template,
            score_threshold=score_threshold,
            parser_function=parser_function,
        )
        self.passing_score_threshold = passing_score_threshold

    async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate whether the answer is accurate based on the query, and 
        return whether the evaluation passes based on a custom threshold."""
        result = await super().aevaluate(query, response, contexts, sleep_time_in_seconds, **kwargs)

        details = parse_feedback(result.feedback)
        score = result.score if result.score is not None else sum(
            comment.score for comment in details.comments)
        # Check if the score passes the custom threshold
        passing = score >= self.passing_score_threshold

        # Update the result to include the passing flag
        result.passing = passing

        extended_result = EvaluationResult(
            **result.model_dump(),
            details=details
        )

        return extended_result
