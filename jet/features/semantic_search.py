from typing import Optional, TypedDict, List, Dict
import numpy as np
from jet.data.header_utils._prepare_for_rag import preprocess_text
from jet.data.utils import generate_unique_id
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_types import EmbedModelType


class SearchResult(TypedDict):
    rank: int
    score: float
    header: str
    content: str
    id: str
    metadata: Dict


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def vector_search(
    query: str,
    texts: List[str],
    embed_model: EmbedModelType,
    top_k: Optional[int] = None,
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[Dict]] = None
) -> List[SearchResult]:
    """Perform vector search with chunking and return ranked results."""
    # Validate inputs
    if ids is not None and len(ids) != len(texts):
        raise ValueError("Length of ids must match length of texts")
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("Length of metadatas must match length of texts")

    # Generate IDs if not provided
    if ids is None:
        ids = [generate_unique_id() for _ in texts]

    # Use empty dicts for metadata if not provided
    if metadatas is None:
        metadatas = [{} for _ in texts]

    # Chunk texts if needed
    chunk_to_doc = []
    for doc_idx, (text, doc_id, metadata) in enumerate(zip(texts, ids, metadatas)):
        chunk_to_doc.append((doc_idx, text, doc_id, metadata))

    # Preprocess text
    preprocessed_texts = [preprocess_text(text) for text in texts]

    # Generate embeddings for query and all chunks
    embeddings = generate_embeddings(
        [query] + preprocessed_texts,
        embed_model,
        return_format="numpy",
        show_progress=True,
        batch_size=64
    )

    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]

    # Calculate similarities
    similarities = [
        (cosine_similarity(query_embedding, chunk_emb),
         doc_idx, orig_text, doc_id, metadata)
        for chunk_emb, (doc_idx, orig_text, doc_id, metadata) in zip(chunk_embeddings, chunk_to_doc)
    ]

    # Aggregate scores by document (take max score across chunks)
    doc_scores = {}
    for score, doc_idx, orig_text, doc_id, metadata in similarities:
        if doc_idx not in doc_scores or score > doc_scores[doc_idx][0]:
            doc_scores[doc_idx] = (score, orig_text, doc_id, metadata)

    # Sort by score and create results
    if not top_k:
        top_k = len(texts)
    results = []
    for rank, (doc_idx, (score, content, doc_id, metadata)) in enumerate(
        sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)[
            :top_k], 1
    ):
        header = content.splitlines()[0]
        results.append(SearchResult(
            rank=rank,
            score=float(score),
            header=header,
            content=content,
            id=doc_id,
            metadata=metadata
        ))

    return results
