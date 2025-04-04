from typing import TypedDict
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import Ollama
from llama_index.core.evaluation import FaithfulnessEvaluator
from jet.logger import logger
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.evaluation.base import EvaluationResult

FAITHFULNESS_EVAL_TEMPLATE = PromptTemplate(
    "Please tell if a given piece of information "
    "is supported by the context.\n"
    "You need to answer with either YES or NO.\n"
    "Answer YES if any of the context supports the information, even "
    "if most of the context is unrelated. "
    "Some examples are provided below. \n\n"
    "Information: Apple pie is generally double-crusted.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: YES\n"
    "Information: Apple pies tastes bad.\n"
    "Context: An apple pie is a fruit pie in which the principal filling "
    "ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream "
    "('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above "
    "and below the filling; the upper crust may be solid or "
    "latticed (woven of crosswise strips).\n"
    "Answer: NO\n"
    "Information: {query_str}\n"
    "Context: {context_str}\n"
    "Answer: "
)


def evaluate_faithfulness(
    model: str | OLLAMA_MODEL_NAMES,
    query: str,
    contexts: str | list[str],
    response: str,
    eval_template: PromptTemplate = FAITHFULNESS_EVAL_TEMPLATE,
) -> EvaluationResult:
    """Evaluates the response from a query if it matches any source nodes."""
    if isinstance(contexts, str):
        contexts = [contexts]

    llm = Ollama(model)
    evaluator = FaithfulnessEvaluator(
        llm=llm,
        eval_template=eval_template,
    )
    logger.debug("Evaluating faithfulness...")
    result = evaluator.evaluate(
        query=query,
        contexts=contexts,
        response=response,
    )
    return result
