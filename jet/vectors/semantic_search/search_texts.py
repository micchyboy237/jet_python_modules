import uuid
import numpy as np
from typing import List, Optional, TypedDict
from jet.libs.llama_cpp.embeddings import LlamacppEmbedding
from jet.search.rag.base import preprocess_texts
from jet._token.token_utils import token_counter



class SearchResult(TypedDict):
    id: str
    rank: int
    doc_index: int
    score: float
    tokens: int
    text: str

def search(
    query: str,
    documents: List[str],
    model: str = "embeddinggemma",
    top_k: int = None,
    ids: Optional[List[str]] = None
) -> List[SearchResult]:
    """Search for documents most similar to the query.
    If top_k is None, return all results sorted by similarity.
    If ids is None, generate UUIDs for each document.
    """
    if not documents:
        return []
    client = LlamacppEmbedding(model=model)
    preprocessed_query = preprocess_texts(query)
    preprocessed_docs = preprocess_texts(documents)
    vectors = client.get_embeddings(preprocessed_query + preprocessed_docs, batch_size=16, show_progress=True)
    query_vector = vectors[0]
    doc_vectors = vectors[1:]
    similarities = np.dot(doc_vectors, query_vector) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-10
    )
    sorted_indices = np.argsort(similarities)[::-1]
    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]
    # Generate UUIDs if ids not provided, else use provided ids
    doc_ids = [str(uuid.uuid4()) for _ in documents] if ids is None else ids
    doc_tokens: List[int] = token_counter(documents, model, prevent_total=True)
    return [
        {
            "id": doc_ids[sorted_indices[i]],
            "rank": i + 1,
            "doc_index": int(sorted_indices[i]),
            "score": float(similarities[sorted_indices[i]]),
            "tokens": doc_tokens[sorted_indices[i]],
            "text": documents[sorted_indices[i]],
        }
        for i in range(len(sorted_indices))
    ]
