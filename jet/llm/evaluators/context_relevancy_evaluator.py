from typing import TypedDict
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.llm.evaluators.helpers.context_relevancy_evaluator import ContextRelevancyEvaluator
from llama_index.core.prompts.base import PromptTemplate
from jet.llm.evaluators.helpers.base import EvaluationResult

EVAL_QUESTIONS = [
    "Does the retrieved context match the subject matter of the user's query?",
    "Can the retrieved context be used exclusively to provide a full to the user's query?",
]


CONTEXT_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate whether the provided context contains specific, concrete, and logically complete information that answers the user's query.\n"
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
    "[RESULT] <total_score>  # This is the sum of the individual question scores.\n"
    "[EXCERPTS] ```json\n<valid JSON array of strings representing all relevant parts of the context that directly answer the query>\n```\n"
    "If no relevant parts exist, use an empty array.\n\n"
    "Example:\n"
    "Feedback:\n"
    "Q1: YES - The topic of National ID is clearly discussed. (Score: 1.0)\n"
    "Q2: NO - The context only claims to offer steps but doesn't actually provide them. (Score: 0.0)\n\n"
    "[RESULT] 1.0\n"
    "[EXCERPTS] ```json\n[]\n```\n\n"
    "Query:\n{query_str}\n"
    "Context:\n{context_str}\n"
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
