"""
Chunking Quality Analysis Demo
Shows how to evaluate if chunks are semantically relevant to a query.
"""

from jet.eval.rag_evaluator import RAGEvaluator


def main():
    evaluator = RAGEvaluator(project_name="demo-chunking-analysis")

    query = "What are the benefits of the Premium plan?"

    # Simulated chunks from your chunking_utils
    chunks = [
        "The Premium plan includes unlimited storage and priority support.",
        "We also offer a Basic plan for small teams.",
        "Premium users get access to advanced analytics dashboards.",
    ]

    print(f"Evaluating chunk quality for: {query}")
    results = evaluator.evaluate_chunk_quality(chunks, query)

    print("\nChunking Metrics:")
    if "average_score" in results:
        print(f"Average Contextual Relevancy: {results['average_score']:.4f}")
    else:
        print(f"Status: {results}")


if __name__ == "__main__":
    main()
