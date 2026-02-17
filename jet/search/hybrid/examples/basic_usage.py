"""
Basic usage example: index documents → retrieve with hybrid search
"""

from jet.search.hybrid.retrieval_pipeline import Document, RetrievalPipeline


def main():
    # Sample documents (in real apps these would come from DB, files, API, etc.)
    documents = [
        Document(
            id="doc1",
            text="Python is a high-level, interpreted programming language known for its readability.",
            metadata={"category": "programming", "lang": "python", "year": 2023},
        ),
        Document(
            id="doc2",
            text="JavaScript is the primary language for web development and runs in browsers.",
            metadata={"category": "programming", "lang": "javascript", "year": 2024},
        ),
        Document(
            id="doc3",
            text="The quick brown fox jumps over the lazy dog.",
            metadata={"category": "pangram", "lang": "english"},
        ),
        Document(
            id="doc4",
            text="Machine learning is a field of artificial intelligence that uses statistical techniques.",
            metadata={"category": "ai", "year": 2025},
        ),
        Document(
            id="doc5",
            text="Rust is a systems programming language focused on safety and performance.",
            metadata={"category": "programming", "lang": "rust"},
        ),
    ]

    # Initialize pipeline
    pipeline = RetrievalPipeline()

    print("Adding documents...")
    pipeline.add_documents(documents)
    print(f"→ {len(documents)} documents indexed\n")

    # Simple query – no metadata filter
    query = "programming language safety performance"
    print(f"Query: {query}")

    results = pipeline.retrieve(query, top_k=3)

    print("\nTop results:")
    for i, doc in enumerate(results, 1):
        print(f"  {i}. [{doc.id}] {doc.text[:80]}...")
        print(f"     metadata: {doc.metadata}\n")


if __name__ == "__main__":
    main()
