from typing import TypedDict
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from llama_index.core.evaluation import RelevancyEvaluator
from jet.logger import logger
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.evaluation.base import EvaluationResult

RELEVANCY_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the response for the query \
    is in line with the context information provided.\n"
    "You have two options to answer. Either YES/ NO.\n"
    "Answer - YES, if the response for the query \
    is in line with context information otherwise NO.\n"
    "Query and Response: \n {query_str}\n"
    "Context: \n {context_str}\n"
    "Answer: "
)


def evaluate_relevancy(
    model: str | OLLAMA_MODEL_NAMES,
    query: str,
    contexts: str | list[str],
    response: str,
    eval_template: PromptTemplate = RELEVANCY_EVAL_TEMPLATE,
) -> EvaluationResult:
    """Evaluates if the response + source nodes match the query."""
    if isinstance(contexts, str):
        contexts = [contexts]

    llm = Ollama(model)
    evaluator = RelevancyEvaluator(
        llm=llm,
        eval_template=eval_template,
    )
    logger.debug("Evaluating relevancy...")
    result = evaluator.evaluate(
        query=query,
        contexts=contexts,
        response=response,
    )
    return result
