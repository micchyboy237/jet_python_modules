"""Guideline evaluation."""

import asyncio
import logging
from typing import Any, Optional, Sequence, Union, cast

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.settings import Settings

logger = logging.getLogger(__name__)


DEFAULT_GUIDELINES = [
    "The response should fully answer the query.",
    "The response should avoid being vague or ambiguous.",
    "The response should be specific and use statistics or numbers when possible.",
]

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    "Here is the original query:\n"
    "Query: {query}\n"
    "Critique the following response based on the guidelines below:\n"
    "Response: {response}\n"
    "Guidelines: {guidelines}\n"
    "Now please provide constructive criticism.\n"
)


class EvaluationData(BaseModel):
    passing: bool = Field(
        ..., description="Indicates whether the response meets the evaluation guidelines.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Evaluation score ranging from 0.0 (lowest) to 1.0 (highest), representing guideline adherence. Scores 0.5 and above are considered passing."
    )
    feedback: str = Field(
        ...,
        description="Detailed feedback on the response, highlighting strengths and areas for improvement."
    )


class GuidelineEvaluator(BaseEvaluator):
    """Guideline evaluator.

    Evaluates whether a query and response pair passes the given guidelines.

    This evaluator only considers the query string and the response string.

    Args:
        guidelines(Optional[str]): User-added guidelines to use for evaluation.
            Defaults to None, which uses the default guidelines.
        eval_template(Optional[Union[str, BasePromptTemplate]] ):
            The template to use for evaluation.
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        guidelines: Optional[str | list[str]] = None,
        eval_template: Optional[Union[str, BasePromptTemplate]] = None,
        output_parser: Optional[PydanticOutputParser] = None,
    ) -> None:
        if isinstance(guidelines, list):
            guidelines = "\n".join(guidelines)

        self._llm = llm or Settings.llm
        self._guidelines = guidelines or DEFAULT_GUIDELINES

        self._eval_template: BasePromptTemplate
        if isinstance(eval_template, str):
            self._eval_template = PromptTemplate(eval_template)
        else:
            self._eval_template = eval_template or DEFAULT_EVAL_TEMPLATE

        self._output_parser = output_parser or PydanticOutputParser(
            output_cls=EvaluationData
        )
        self._eval_template.output_parser = self._output_parser

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "eval_template": self._eval_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "eval_template" in prompts:
            self._eval_template = prompts["eval_template"]

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate whether the query and response pair passes the guidelines."""
        del contexts  # Unused
        del kwargs  # Unused
        if query is None or response is None:
            raise ValueError("query and response must be provided")

        logger.debug("prompt: %s", self._eval_template)
        logger.debug("query: %s", query)
        logger.debug("response: %s", response)
        logger.debug("guidelines: %s", self._guidelines)

        await asyncio.sleep(sleep_time_in_seconds)

        eval_response = await self._llm.apredict(
            self._eval_template,
            query=query,
            response=response,
            guidelines=self._guidelines,
        )
        eval_data = self._output_parser.parse(eval_response)
        eval_data = cast(EvaluationData, eval_data)

        return EvaluationResult(
            query=query,
            response=response,
            passing=eval_data.passing,
            score=eval_data.score,
            feedback=eval_data.feedback,
        )


CONTEXT_EVAL_GUIDELINES = [
    "The response should be relevant to the provided context.",
    "The response should be accurate and factually correct based on the context.",
    "The response should avoid making unsupported assumptions beyond the given context.",
    # "If the context lacks necessary information, the response should acknowledge this rather than speculate.",
]


CONTEXT_EVAL_TEMPLATE = PromptTemplate("""
Here is the original query:
Query:
{query}

Context:
{context}

Critique the following response based on the context above and guidelines below:

Response:
{response}

Guidelines:
{guidelines}

Now please provide constructive criticism given the context and response.
""".strip())


class ContextEvaluationData(BaseModel):
    passing: bool = Field(
        ...,
        description="Indicates whether the response adheres to the context evaluation guidelines, considering the provided context."
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Evaluation score indicating how well the response aligns with the provided context and adheres to the context-specific guidelines. Scores 0.5 and above are considered passing."
    )
    feedback: str = Field(
        ...,
        description="Constructive feedback on how the response aligns with the given context, highlighting strengths, weaknesses, hallucinations, and any inaccuracies or unsupported assumptions made by the response."
    )


class GuidelineContextEvaluator(GuidelineEvaluator):
    def __init__(
        self,
        *args,
        guidelines: Optional[str | list[str]] = None,
        eval_template: Optional[Union[str, BasePromptTemplate]] = None,
        output_parser: Optional[PydanticOutputParser] = None,
        **kwargs,
    ) -> None:
        guidelines = guidelines or CONTEXT_EVAL_GUIDELINES
        eval_template = eval_template or CONTEXT_EVAL_TEMPLATE
        output_parser = output_parser or PydanticOutputParser(
            output_cls=ContextEvaluationData
        )

        super().__init__(
            *args,
            guidelines=guidelines,
            eval_template=eval_template,
            output_parser=output_parser,
            **kwargs
        )

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate whether the query and response pair passes the guidelines."""
        del kwargs  # Unused
        if query is None or not contexts or response is None:
            raise ValueError("query, contexts and response must be provided")

        logger.debug("prompt: %s", self._eval_template)
        logger.debug("query: %s", query)
        logger.debug("contexts: %s", contexts)
        logger.debug("response: %s", response)
        logger.debug("guidelines: %s", self._guidelines)

        await asyncio.sleep(sleep_time_in_seconds)

        eval_response = await self._llm.apredict(
            self._eval_template,
            query=query,
            context="\n\n".join(contexts),
            response=response,
            guidelines=self._guidelines,
        )
        eval_data = self._output_parser.parse(eval_response)
        eval_data = cast(ContextEvaluationData, eval_data)

        return EvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            passing=eval_data.passing,
            score=eval_data.score,
            feedback=eval_data.feedback,
        )
