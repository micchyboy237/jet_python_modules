# jet_python_modules/jet/libs/stanza/examples/examples_rag_stanza_faiss.py
"""
examples_rag_stanza_faiss.py
Usage examples for `rag_retriever_faiss.py`.

Demonstrates how to:
  1. Create example RAG context chunks.
  2. Build a FAISS index using SentenceTransformer embeddings.
  3. Retrieve the most relevant chunks for a given query.
  4. Produce a human-readable retrieval summary.

Run:
    python examples_rag_stanza_faiss.py
"""

from jet.libs.stanza.rag_retriever_faiss import ContextChunk, RagRetrieverFAISS


def example_build_chunks() -> list[ContextChunk]:
    """Example: create sample chunks manually (simulating parsed RAG output)."""
    print("=== Example: Building example chunks ===")
    chunks = [
        ContextChunk(
            text=(
                "OpenAI released the GPT-5 model in October 2025, advancing multimodal reasoning "
                "and multilingual understanding for enterprise RAG applications."
            ),
            salience=0.92,
            entities=["OpenAI", "GPT-5"],
            sentence_indices=[0, 1],
        ),
        ContextChunk(
            text=(
                "Universities like MIT and ETH Zurich are developing syntax-aware retrieval systems "
                "that integrate Stanza parsing with transformer embeddings."
            ),
            salience=0.84,
            entities=["MIT", "ETH Zurich", "Stanza"],
            sentence_indices=[2],
        ),
        ContextChunk(
            text=(
                "Analysts from Gartner predict that RAG systems using adaptive chunking and FAISS "
                "retrieval will reduce hallucination rates by over 40%."
            ),
            salience=0.78,
            entities=["Gartner", "FAISS"],
            sentence_indices=[3],
        ),
    ]
    print(f"Built {len(chunks)} context chunks.")
    return chunks


def example_build_index(chunks: list[ContextChunk]):
    """Example: build a FAISS index from chunks."""
    print("\n=== Example: Building FAISS Index ===")
    retriever = RagRetrieverFAISS()
    retriever.build_index(chunks)
    print("FAISS index successfully built.")
    return retriever


def example_retrieve_query(retriever: RagRetrieverFAISS):
    """Example: run retrieval for a given query."""
    print("\n=== Example: Query Retrieval ===")
    query = "What is the role of FAISS in RAG systems?"
    results = retriever.retrieve(query, top_k=2)
    print(f"Retrieved {len(results)} results for query: '{query}'\n")
    for rank, (dist, chunk) in enumerate(results, 1):
        print(f"Rank {rank} | Distance: {dist:.4f}")
        print(f"Salience: {chunk.salience:.2f}")
        print(f"Entities: {', '.join(chunk.entities)}")
        print(f"Text: {chunk.text[:150]}...\n")


def example_human_readable_summary(retriever: RagRetrieverFAISS):
    """Example: generate a formatted summary of retrieval results."""
    print("\n=== Example: Human-Readable Retrieval Summary ===")
    query = "How is syntax-aware retrieval used with transformers?"
    summary = retriever.describe_result(query, top_k=2)
    print(f"Query: {summary['query']}")
    for r in summary["results"]:
        print(f"\nRank {r['rank']} | Distance: {r['distance']}")
        print(f"Salience: {r['salience']}")
        print(f"Entities: {', '.join(r['entities'])}")
        print(f"Preview: {r['preview']}")
    return summary


def example_full_demo():
    """Run the full end-to-end FAISS retrieval demonstration."""
    print("\n=== Example: Full RAG Retriever FAISS Demo ===")
    chunks = example_build_chunks()
    retriever = example_build_index(chunks)
    example_retrieve_query(retriever)
    example_human_readable_summary(retriever)


if __name__ == "__main__":
    example_full_demo()
