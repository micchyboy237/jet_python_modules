"""
Full Pipeline Evaluation Demo
Simulates the end-to-end flow of search, retrieval, and generation.
"""

from jet.eval.rag_evaluator import RAGEvaluator


def main():
    evaluator = RAGEvaluator(project_name="demo-full-pipeline")

    # 1. Simulate Search & Retrieval
    query = "Top isekai anime 2026"
    retrieved_chunks = [
        "Source: animenews.com\nRe:Zero Season 3 is confirmed for 2026.",
        "Source: wikianime.org\nMushoku Tensei continues its journey in 2026.",
        "Source: blog.net\nSome unrelated news about sports.",
    ]

    # 2. Simulate LLM Generation
    final_answer = (
        "In 2026, top isekai anime include Re:Zero Season 3 and Mushoku Tensei."
    )

    print(f"Running full pipeline evaluation for: {query}")
    results = evaluator.run_pipeline_evaluation(
        query=query, retrieved_chunks=retrieved_chunks, final_answer=final_answer
    )

    print("\nPipeline Report:")
    print(f"Query: {results['query']}")
    print(f"Retrieval Status: {results['retrieval']}")
    print(f"Generation Faithfulness: {results['generation'].get('faithfulness')}")
    print(f"Generation Relevancy: {results['generation'].get('answer_relevancy')}")


if __name__ == "__main__":
    main()
