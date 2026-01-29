from jet.adapters.llama_cpp.hybrid_search import HybridSearcher
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS


if __name__ == "__main__":
    # Prepare data
    docs = [
        {
            "id": "d1",
            "content": "Hybrid vector search best practices 2025. Use RRF for combining BM25 and dense embeddings. Run both retrievers in parallel and fuse with reciprocal rank fusion...",
        },
        {
            "id": "d2",
            "content": "nomic-embed-text-v1.5 performance. Very fast on llama.cpp especially with Q5_K_M quantization. Low memory usage and excellent latency for local inference...",
        },
        {
            "id": "d3",
            "content": "Reciprocal Rank Fusion. Simple yet powerful fusion method used in Elastic, Weaviate, Azure Search, and many production RAG systems...",
        },
        {
            "id": "d4",
            "content": "Local embedding servers. llama.cpp provides OpenAI compatible API for embedding models like nomic-embed-text-v1.5. Easy to run on CPU/GPU...",
        },
        {
            "id": "d5",
            "content": "BM25 is still very strong. Especially good at rare terms, IDs, exact matches, product codes, and keyword precision in hybrid search...",
        },
    ]

    model: LLAMACPP_EMBED_KEYS = "nomic-embed-text"

    hybrid = HybridSearcher.from_documents(
        documents=docs,
        model=model,
        k_candidates=10,
        k_final=5,
        bm25_weight=1.2,
        vector_weight=1.0,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )

    # Query
    query = "fast local embeddings with llama.cpp"
    results = hybrid.search(query)

    print(f"\nResults for: {query!r}\n")
    for i, res in enumerate(results, 1):
        doc = res.item
        preview = (
            doc["content"][:80] + "..." if len(doc["content"]) > 80 else doc["content"]
        )
        print(f"{i:2d}. {res.score:6.4f}  {doc['id']}  {preview}")
