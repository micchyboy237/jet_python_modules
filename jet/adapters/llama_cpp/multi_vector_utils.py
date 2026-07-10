"""Multi-vector search utilities for fine-grained document matching."""

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed


def chunk_document(
    document: str,
    chunk_size: int = 256,
    chunk_overlap: int = 50,
    separator: str = " ",
) -> list[str]:
    """
    Split a document into overlapping chunks for fine-grained matching.
    """
    if not document.strip():
        return []

    # Ensure chunk_size > chunk_overlap to avoid infinite loop
    if chunk_size <= chunk_overlap:
        chunk_overlap = max(0, chunk_size - 1)

    words = document.split(separator)

    # If document is shorter than chunk_size, return as single chunk
    if len(words) <= chunk_size:
        return [document]

    chunks = []
    step = chunk_size - chunk_overlap  # This is now guaranteed to be > 0
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = separator.join(words[start:end])
        chunks.append(chunk)
        start += step

    return chunks


def multi_vector_search(
    query: str,
    documents: list[str],
    top_k: int = 10,
    chunk_size: int = 256,
    chunk_overlap: int = 50,
    aggregation: str = "max",  # "max", "mean", "sum"
    return_chunks: bool = False,
) -> list[dict]:
    """
    Multi-vector search using late interaction (ColBERT-style).
    Each document is chunked, and query is compared against all chunks.

    Args:
        query: Search query
        documents: List of documents to search
        top_k: Number of final results to return
        chunk_size: Number of words per chunk
        chunk_overlap: Overlap between chunks
        aggregation: How to aggregate chunk scores ("max", "mean", "sum")
        return_chunks: If True, return best matching chunks

    Returns:
        Ranked results with scores and optional chunk information
    """
    query_emb = embed(query)

    # Process each document
    doc_results = []

    for doc_idx, document in enumerate(documents):
        # Chunk the document
        chunks = chunk_document(document, chunk_size, chunk_overlap)

        if not chunks:
            continue

        # Embed all chunks
        chunk_embs = embed(chunks)

        # Calculate similarity with each chunk
        chunk_scores = [
            _cosine_similarity(query_emb, chunk_emb) for chunk_emb in chunk_embs
        ]

        # Aggregate chunk scores
        if aggregation == "max":
            doc_score = max(chunk_scores) if chunk_scores else 0
        elif aggregation == "mean":
            doc_score = np.mean(chunk_scores) if chunk_scores else 0
        elif aggregation == "sum":
            doc_score = sum(chunk_scores) if chunk_scores else 0
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Find best chunk
        best_chunk_idx = np.argmax(chunk_scores) if chunk_scores else 0

        result = {
            "document": document,
            "score": float(doc_score),
            "index": doc_idx,
        }

        if return_chunks:
            result["best_chunk"] = chunks[best_chunk_idx]
            result["chunk_score"] = float(chunk_scores[best_chunk_idx])
            result["total_chunks"] = len(chunks)
            result["chunk_scores"] = [float(s) for s in chunk_scores]

        doc_results.append(result)

    # Sort by score and return top_k
    doc_results.sort(key=lambda x: x["score"], reverse=True)
    return doc_results[:top_k]


def passage_retrieval(
    query: str,
    passages: list[str],
    top_k: int = 10,
    return_scores: bool = True,
) -> list[dict]:
    """
    Direct passage-level retrieval for academic/research search.
    Each passage is treated as an independent unit.

    Args:
        query: Search query
        passages: List of passages to search
        top_k: Number of results to return
        return_scores: If True, return scores

    Returns:
        Ranked passages with scores
    """
    query_emb = embed(query)
    passage_embs = embed(passages)

    similarities = [
        _cosine_similarity(query_emb, passage_emb) for passage_emb in passage_embs
    ]

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        result = {
            "passage": passages[idx],
            "index": idx,
        }
        if return_scores:
            result["score"] = float(similarities[idx])
        results.append(result)

    return results


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    # Long documents for multi-vector demonstration
    documents = [
        "The giant panda (Ailuropoda melanoleuca), also known as the panda bear, "
        "is a bear species endemic to China. It is characterised by its bold black-and-white "
        "coat and rotund body. The name 'giant panda' is sometimes used to distinguish it from "
        "the red panda. Though it belongs to the order Carnivora, the giant panda is a folivore, "
        "with bamboo shoots and leaves making up more than 99% of its diet. Giant pandas in the "
        "wild will occasionally eat other grasses, wild tubers, or even meat in the form of birds, "
        "rodents, or carrion. In captivity, they may receive honey, eggs, fish, yams, shrub leaves, "
        "oranges, or bananas along with specially prepared food.",
        "Python is an interpreted high-level general-purpose programming language. Its design "
        "philosophy emphasizes code readability with its use of significant indentation. Its "
        "language constructs and object-oriented approach aim to help programmers write clear, "
        "logical code for small and large-scale projects. Python is dynamically-typed and "
        "garbage-collected. It supports multiple programming paradigms, including structured "
        "(particularly, procedural), object-oriented and functional programming. It is often "
        "described as a 'batteries included' language due to its comprehensive standard library.",
    ]

    query = "What do pandas eat?"

    print("=" * 60)
    print("MULTI-VECTOR DOCUMENT SEARCH")
    print("=" * 60)
    print(f"Query: {query}\n")

    results = multi_vector_search(
        query, documents, top_k=3, chunk_size=30, return_chunks=True
    )

    for i, r in enumerate(results, 1):
        print(f"{i}. [score:{r['score']:.4f}] Document {r['index']}")
        if "best_chunk" in r:
            print(f'   Best chunk [{r["chunk_score"]:.4f}]: "{r["best_chunk"]}..."')
        print()

    # Passage retrieval example
    passages = [
        "The giant panda's diet is over 99% bamboo.",
        "Pandas may occasionally eat small rodents and birds.",
        "Python uses indentation for code blocks.",
        "Conservation programs provide pandas with special diets.",
        "Bamboo is a type of grass that grows rapidly.",
    ]

    print("=" * 60)
    print("PASSAGE-LEVEL RETRIEVAL")
    print("=" * 60)
    print(f"Query: {query}\n")

    results = passage_retrieval(query, passages, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.4f}] {r['passage']}")
