"""03_rag_integration.py - Simulating a RAG pipeline with faithfulness gating."""

from jet.adapters.llama_cpp.tasks.rag.eval.faithfulness_relevancy_evaluator import (
    evaluate_faithfulness_llamacpp,
)
from jet.logger import logger


def mock_retriever(query: str) -> list[str]:
    """Simulates a vector database retrieval."""
    if "python" in query.lower():
        return [
            "Python is a high-level, general-purpose programming language.",
            "Its design philosophy emphasizes code readability with the use of significant indentation.",
        ]
    return ["No relevant documents found."]


def mock_llm_generator(query: str, contexts: list[str]) -> str:
    """Simulates an LLM generating a response based on contexts."""
    if "python" in query.lower() and contexts:
        return "Python is a popular programming language known for its readability."
    return "I don't have enough information to answer that."


def main():
    logger.info("Starting RAG Integration Demo")

    query = "What is Python?"

    # 1. Retrieval
    logger.info("Step 1: Retrieving contexts...")
    contexts = mock_retriever(query)
    logger.debug(f"Retrieved {len(contexts)} contexts")

    # 2. Generation
    logger.info("Step 2: Generating response...")
    response = mock_llm_generator(query, contexts)
    logger.debug(f"Generated response: {response}")

    # 3. Faithfulness Evaluation (Quality Gate)
    logger.info("Step 3: Evaluating faithfulness...")
    result = evaluate_faithfulness_llamacpp(
        query=query,
        response=response,
        contexts=contexts,
    )

    print("\n--- RAG Pipeline Result ---")
    if result.passing:
        print(f"✅ Final Answer: {response}")
    else:
        print(f"⚠️ Response rejected due to lack of faithfulness.")
        print(f"   Feedback: {result.feedback}")


if __name__ == "__main__":
    main()
