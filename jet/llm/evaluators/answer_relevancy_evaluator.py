from typing import TypedDict
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from jet.llm.evaluators.helpers.answer_relevancy_evaluator import AnswerRelevancyEvaluator
from jet.logger import logger
from llama_index.core.prompts.base import PromptTemplate
from jet.llm.evaluators.helpers.base import EvaluationResult

EVAL_QUESTIONS = [
    "Does the provided response match the subject matter of the user's query?",
    "Does the provided response attempt to address the focus or perspective on the subject matter taken on by the user's query?",
]

ANSWER_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate whether the provided response contains specific, concrete, and logically complete information that answers the user's query.\n"
    "Evaluate the following questions step-by-step:\n"
    "{questions_str}\n"
    "Each question is worth 1.0 point. Partial scores are allowed. Use strict criteria—generic mentions or vague claims are not sufficient.\n\n"
    "**Important:** For each question:\n"
    "- Answer YES or NO\n"
    "- Provide a short explanation\n"
    "- Provide a score (between 0.0 and 1.0)\n\n"
    "**Scoring Guide:**\n"
    "- 1.0 = Fully and concretely answered with clear, complete info\n"
    "- 0.5 = Partially answered or missing important details\n"
    "- 0.0 = Not answered at all or only vaguely mentioned\n\n"
    "At the end, write the result in the following exact format:\n"
    "[RESULT] <total_score>  # This is the sum of the individual question scores.\n\n"
    "Example:\n"
    "Feedback:\n"
    "Q1: YES - The response clearly addresses the topic of the query. (Score: 1.0)\n"
    "Q2: NO - The response does not adopt the requested perspective. (Score: 0.0)\n\n"
    "[RESULT] 1.0\n\n"
    "Query:\n{query}\n"
    "Response:\n{response}\n"
    "Feedback:"
)

_DEFAULT_SCORE_THRESHOLD = 2.0


def evaluate_answer_relevancy(
    model: str | OLLAMA_MODEL_NAMES,
    query: str,
    response: str,
    questions: list[str] = EVAL_QUESTIONS,
    eval_template: PromptTemplate = ANSWER_EVAL_TEMPLATE,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
) -> EvaluationResult:
    """Evaluates the relevancy of an answer to the query."""
    partial_template = eval_template.partial_format(
        questions_str="\n".join([f"{idx + 1}. {q}" for idx, q in enumerate(questions)]))

    llm = Ollama(model)
    evaluator = AnswerRelevancyEvaluator(
        llm=llm,
        eval_template=partial_template,
        score_threshold=score_threshold,
    )
    logger.debug("Evaluating answer relevancy...")
    result = evaluator.evaluate(
        query=query,
        response=response,
    )
    return result
