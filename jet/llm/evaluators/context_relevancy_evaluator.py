from typing import TypedDict
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from llama_index.core.evaluation import ContextRelevancyEvaluator
from jet.logger import logger
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.evaluation.base import EvaluationResult

CONTEXT_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the retrieved context from the document sources are relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Does the retrieved context match the subject matter of the user's query?\n"
    "2. Can the retrieved context be used exclusively to provide a full answer to the user's query?\n"
    "Each question above is worth 2 points, where partial marks are allowed and encouraged. Provide detailed feedback on the response "
    "according to the criteria questions previously mentioned. "
    "After your feedback provide a final result by strictly following this format: '[RESULT] followed by the floating number representing the total score assigned to the response'\n\n"
    "Example feedback format:\nFeedback:\n<generated_feedback>\n\n[RESULT] <total_score:.2f>\n\n"
    "Query: \n {query_str}\n"
    "Context: \n {context_str}\n"
    "Feedback:"
)


def evaluate_context_relevancy(
    model: str | OLLAMA_MODEL_NAMES,
    query: str,
    contexts: list[str],
    eval_template: PromptTemplate = CONTEXT_EVAL_TEMPLATE,
) -> EvaluationResult:
    """Evaluates the relevancy of the context to the query."""
    llm = Ollama(model)
    evaluator = ContextRelevancyEvaluator(
        llm=llm,
        eval_template=eval_template,
    )
    logger.debug("Evaluating context relevancy...")
    result = evaluator.evaluate(
        query=query,
        contexts=contexts,
    )
    return result
