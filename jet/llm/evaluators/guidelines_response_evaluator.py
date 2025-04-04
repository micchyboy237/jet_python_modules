from typing import Any, Optional, Sequence, TypedDict, Union, cast
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from llama_index.core.llms.llm import LLM
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate
from llama_index.core.settings import Settings
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.bridge.pydantic import BaseModel, Field

# Define a base template for evaluation
DEFAULT_RESPONSE_GUIDELINES = [
    "The response should fully answer the query.",
    "The response should avoid being vague or ambiguous.",
    "The response should be specific and use statistics or numbers when possible.",
]

RESPONSE_GUIDELINES_EVAL_TEMPLATE = PromptTemplate(
    "Here is the original query:\n"
    "Query: {query}\n"
    "Critique the following response based on the guidelines below:\n"
    "Response: {response}\n"
    "Guidelines: {guidelines}\n"
    "Now please provide constructive criticism.\n"
)


class EvaluationData(BaseModel):
    passing: bool = Field(
        ..., description="Indicates whether the response meets the evaluation guidelines."
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Evaluation score ranging from 0.0 (lowest) to 1.0 (highest)."
    )
    feedback: str = Field(
        ...,
        description="Detailed feedback on the response."
    )


def evaluate_response_guidelines(
    model: str | OLLAMA_MODEL_NAMES,
    query: str,
    response: str,
    guidelines: Optional[Union[str, list[str]]] = None,
    eval_template: Optional[Union[str, BasePromptTemplate]] = None,
    output_parser: Optional[PydanticOutputParser] = None,
) -> EvaluationResult:
    """Evaluates a query and response pair based on guidelines."""
    if not query or not response:
        raise ValueError("query and response must be provided")

    # Default guidelines if none provided
    guidelines = guidelines or DEFAULT_RESPONSE_GUIDELINES
    if isinstance(guidelines, list):
        guidelines = "\n".join(guidelines)

    # Default evaluation template if none provided
    eval_template = eval_template or RESPONSE_GUIDELINES_EVAL_TEMPLATE

    # Ensure the output parser is set
    output_parser = output_parser or PydanticOutputParser(
        output_cls=EvaluationData)

    # Prepare the prompt template
    prompt_template = eval_template if isinstance(
        eval_template, BasePromptTemplate) else PromptTemplate(eval_template)

    # Prepare language model (if not provided, use Settings)
    llm = Ollama(model)

    eval_response = llm.predict(
        prompt_template,
        query=query,
        response=response,
        guidelines=guidelines,
    )

    # Parse and return the evaluation result
    eval_data = output_parser.parse(eval_response)
    eval_data = cast(EvaluationData, eval_data)

    result = EvaluationResult(
        query=query,
        response=response,
        passing=eval_data.passing,
        score=eval_data.score,
        feedback=eval_data.feedback,
    )

    return result
