"""01_basic_faithfulness.py - Basic usage of the FaithfulnessEvaluator."""

from jet.adapters.llama_cpp.tasks.rag.eval.faithfulness_relevancy_evaluator import (
    evaluate_faithfulness_llamacpp,
)
from jet.logger import logger


def main():
    logger.info("Starting Basic Faithfulness Evaluation Demo")

    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    contexts = [
        "Paris is the capital and most populous city of France.",
        "London is the capital of the United Kingdom.",
    ]

    print("\n--- Evaluation Input ---")
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Contexts: {contexts}")

    result = evaluate_faithfulness_llamacpp(
        query=query,
        response=response,
        contexts=contexts,
    )

    print("\n--- Evaluation Result ---")
    print(f"Passing: {result.passing}")
    print(f"Score: {result.score}")
    print(f"Feedback: {result.feedback}")


if __name__ == "__main__":
    main()
