from typing import TypedDict
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from llama_index.core.evaluation import AnswerRelevancyEvaluator
from jet.logger import logger
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.evaluation.base import EvaluationResult

ANSWER_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the response is relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Does the provided response match the subject matter of the user's query?\n"
    "2. Does the provided response attempt to address the focus or perspective "
    "on the subject matter taken on by the user's query?\n"
    "Each question above is worth 1 point. Provide detailed feedback on response according to the criteria questions above  "
    "After your feedback provide a final result by strictly following this format: '[RESULT] followed by the integer number representing the total score assigned to the response'\n\n"
    "Example feedback format:\nFeedback:\n<generated_feedback>\n\n[RESULT] <total_int_score>\n\n"
    "Query: \n {query}\n"
    "Response: \n {response}\n"
    "Feedback:"
)


def evaluate_answer_relevancy(
    model: str | OLLAMA_MODEL_NAMES,
    query: str,
    response: str,
    eval_template: PromptTemplate = ANSWER_EVAL_TEMPLATE,
) -> EvaluationResult:
    """Evaluates the relevancy of an answer to the query."""
    llm = Ollama(model)
    evaluator = AnswerRelevancyEvaluator(
        llm=llm,
        eval_template=eval_template,
    )
    logger.debug("Evaluating answer relevancy...")
    result = evaluator.evaluate(
        query=query,
        response=response,
    )
    return result
