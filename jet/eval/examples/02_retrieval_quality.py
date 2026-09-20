"""
Retrieval Quality Evaluation Demo
Shows how to evaluate Precision and Recall of the retrieval step.
"""

from jet.eval.rag_evaluator import RAGEvaluator


def main():
    evaluator = RAGEvaluator(project_name="demo-retrieval-quality")

    query = "How do I reset my password?"

    # Simulated retrieved chunks from your vector search
    retrieved_chunks = [
        "To reset your password, go to Settings > Account > Security.",
        "Our support team is available 24/7.",
        "Password resets can also be done via email link.",
    ]

    # Ground truth chunks that SHOULD have been retrieved
    expected_chunks = [
        "To reset your password, go to Settings > Account > Security.",
        "Password resets can also be done via email link.",
    ]

    print(f"Evaluating retrieval for: {query}")
    results = evaluator.evaluate_retrieval_ranking(
        query, retrieved_chunks, expected_chunks
    )

    print("\nRetrieval Metrics:")
    if "precision" in results:
        print(f"Precision: {results['precision']}")
        print(f"Recall: {results['recall']}")
    else:
        print(f"Status: {results}")


if __name__ == "__main__":
    main()
