"""
Example: using metadata filters to narrow down candidates before ranking
"""

from jet.search.hybrid.retrieval_pipeline import Document, RetrievalPipeline


def main():
    docs = [
        Document(
            "art1",
            "Leonardo da Vinci painted the Mona Lisa in the Renaissance period.",
            {"type": "art", "period": "renaissance"},
        ),
        Document(
            "art2",
            "Vincent van Gogh created Starry Night using oil on canvas.",
            {"type": "art", "period": "post-impressionism"},
        ),
        Document(
            "tech1",
            "Python 3.11 introduced pattern matching with match-case.",
            {"type": "tech", "lang": "python"},
        ),
        Document(
            "tech2",
            "Rust 1.70 added support for async closures.",
            {"type": "tech", "lang": "rust"},
        ),
        Document(
            "art3",
            "Claude Monet is famous for his Water Lilies series.",
            {"type": "art", "period": "impressionism"},
        ),
    ]

    pipeline = RetrievalPipeline()
    pipeline.add_documents(docs)

    query = "painting renaissance artist"

    print("=== Without filter ===")
    results = pipeline.retrieve(query, top_k=3)
    for i, d in enumerate(results, 1):
        print(f"{i}. {d.id} → {d.text[:70]}... | {d.metadata}")

    print("\n=== With metadata filter: only art documents ===")
    results_filtered = pipeline.retrieve(
        query, metadata_filters={"type": "art"}, top_k=3
    )
    for i, d in enumerate(results_filtered, 1):
        print(f"{i}. {d.id} → {d.text[:70]}... | {d.metadata}")


if __name__ == "__main__":
    main()
