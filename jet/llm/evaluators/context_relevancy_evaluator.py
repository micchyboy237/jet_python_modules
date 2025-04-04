from typing import TypedDict
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.llm.evaluators.helpers.context_relevancy_evaluator import ContextRelevancyEvaluator
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.evaluation.base import EvaluationResult

EVAL_QUESTIONS = [
    "Does the retrieved context match the subject matter of the user's query?",
    "Can the retrieved context be used exclusively to provide a full answer to the user's query?",
]

CONTEXT_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the retrieved context from the document sources is relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "{questions_str}\n"
    "Each question is worth 1.0 point, and partial scores are allowed.\n\n"
    "Answer YES on top if all questions are YES, otherwise answer NO. Then provide YES or NO for each question, followed by a brief explanation and score for that specific question.\n"
    "After your feedback provide a final result by strictly following this format: '[RESULT] followed by the floating number representing the total score assigned to the response'\n\n"
    "Example format:\n"
    "Answer: NO\n"
    "Feedback:\n"
    "Q1: YES - The context clearly aligns with the subject of the query. (Score: 1.0)\n"
    "Q2: NO - The context provides partial information but does not fully answer the query. (Score: 0.5)\n\n"
    "[RESULT] 1.5\n\n"
    "Query: \n{query_str}\n"
    "Context: \n{context_str}\n"
    "Feedback:"
)

_DEFAULT_SCORE_THRESHOLD = 2.0


def evaluate_context_relevancy(
    model: str | OLLAMA_MODEL_NAMES,
    query: str,
    contexts: str | list[str],
    questions: list[str] = EVAL_QUESTIONS,
    eval_template: PromptTemplate = CONTEXT_EVAL_TEMPLATE,
) -> EvaluationResult:
    """Evaluates the relevancy of the context to the query."""
    if isinstance(contexts, str):
        contexts = [contexts]

    partial_template = eval_template.partial_format(
        questions_str="\n".join([f"{idx + 1}. {q}" for idx, q in enumerate(questions)]))

    llm = Ollama(model)
    evaluator = ContextRelevancyEvaluator(
        llm=llm,
        eval_template=partial_template,
        score_threshold=_DEFAULT_SCORE_THRESHOLD,
    )
    logger.debug("Evaluating context relevancy...")
    result = evaluator.evaluate(
        query=query,
        contexts=contexts,
    )
    return result
