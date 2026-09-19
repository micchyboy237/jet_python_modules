"""Faithfulness evaluation using local llama.cpp models."""

from __future__ import annotations

from typing import Optional, Sequence

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.llm_utils import chat
from jet.logger import logger
from llama_index.core.evaluation_base import EvaluationResult

# Prompt template optimized for local llama.cpp models
# Based on LLAMA3_8B_EVAL_TEMPLATE from reference but simplified for direct chat usage
FAITHFULNESS_PROMPT = """Please tell if a given piece of information is supported by the context.
You need to answer with either YES or NO.
Answer YES if **any part** of the context supports the information, even if most of the context is unrelated.
Answer NO if the context does not support the information at all.

Example 1:
Information: The Eiffel Tower is located in Paris.
Context: The Eiffel Tower, a symbol of French culture, stands prominently in the city of Paris.
Answer: YES

Example 2:
Information: Bananas are a type of berry.
Context: Bananas are a popular fruit enjoyed worldwide and are rich in potassium.
Answer: NO

Information: {query_str}
Context: {context_str}
Answer:"""


def evaluate_faithfulness_llamacpp(
    query: Optional[str] = None,
    response: Optional[str] = None,
    contexts: Optional[Sequence[str]] = None,
    model: str = LLM_MODEL,
    raise_error: bool = False,
) -> EvaluationResult:
    """
    Evaluate whether the response is faithful to the contexts using llama.cpp.

    Args:
        query: The original query string (optional, used for metadata).
        response: The generated response string to evaluate.
        contexts: List of context strings retrieved from the knowledge base.
        model: The llama.cpp model identifier to use for evaluation.
        raise_error: Whether to raise an error if the evaluation fails.

    Returns:
        EvaluationResult with passing status, score, and feedback.
    """
    if contexts is None or response is None:
        raise ValueError("contexts and response must be provided")

    if not contexts:
        return EvaluationResult(
            query=query,
            response=response,
            contexts=contexts,
            passing=False,
            score=0.0,
            feedback="No contexts provided for evaluation.",
        )

    logger.info(
        f"Starting faithfulness evaluation for response length: {len(response)}"
    )
    logger.debug(f"Using model: {model}")

    passing = False
    feedback_parts = []

    # Check each context segment
    for i, context in enumerate(contexts):
        prompt = FAITHFULNESS_PROMPT.format(query_str=response, context_str=context)

        try:
            logger.debug(f"Evaluating context segment {i + 1}/{len(contexts)}")
            result = chat(
                prompt_or_messages=prompt,
                model=model,
                temperature=0.0,  # Deterministic for YES/NO
                max_tokens=10,  # Short response expected
            )

            raw_response = result.content.strip()
            logger.debug(f"LLM response for segment {i + 1}: '{raw_response}'")

            # Parse YES/NO
            if "yes" in raw_response.lower():
                passing = True
                feedback_parts.append(f"Segment {i + 1}: SUPPORTED ({raw_response})")
                logger.info(f"Faithfulness confirmed by segment {i + 1}")
                break  # Early stop: if any context supports it, it's faithful
            else:
                feedback_parts.append(
                    f"Segment {i + 1}: NOT SUPPORTED ({raw_response})"
                )

        except Exception as e:
            logger.error(f"Error evaluating segment {i + 1}: {e}")
            if raise_error:
                raise
            feedback_parts.append(f"Segment {i + 1}: ERROR ({str(e)})")

    final_feedback = "; ".join(feedback_parts)
    score = 1.0 if passing else 0.0

    logger.info(f"Faithfulness evaluation complete: passing={passing}, score={score}")

    return EvaluationResult(
        query=query,
        response=response,
        contexts=contexts,
        passing=passing,
        score=score,
        feedback=final_feedback,
    )


async def aevaluate_faithfulness_llamacpp(
    query: Optional[str] = None,
    response: Optional[str] = None,
    contexts: Optional[Sequence[str]] = None,
    model: str = LLM_MODEL,
    raise_error: bool = False,
) -> EvaluationResult:
    """
    Async version of evaluate_faithfulness_llamacpp.
    Note: Current jet.adapters.llama_cpp.llm_utils.chat is synchronous.
    This wrapper maintains API compatibility for async evaluators.
    """
    # Run sync function in executor to avoid blocking event loop if needed,
    # but for local llama.cpp which is already I/O bound via HTTP, direct call is often fine.
    # For true async, we'd need an async chat implementation.
    return evaluate_faithfulness_llamacpp(
        query=query,
        response=response,
        contexts=contexts,
        model=model,
        raise_error=raise_error,
    )


if __name__ == "__main__":
    # Example usage
    test_query = "What is the capital of France?"
    test_response = "The capital of France is Paris."
    test_contexts = [
        "Paris is the capital and most populous city of France.",
        "London is the capital of the United Kingdom.",
    ]

    print("Running faithfulness evaluation...")
    result = evaluate_faithfulness_llamacpp(
        query=test_query,
        response=test_response,
        contexts=test_contexts,
    )

    print(f"Passing: {result.passing}")
    print(f"Score: {result.score}")
    print(f"Feedback: {result.feedback}")
