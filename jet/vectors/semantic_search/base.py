from typing import Optional, TypedDict, List, Dict, Union
import numpy as np
from jet.data.utils import generate_unique_id
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_types import EmbedModelType
from jet.wordnet.keywords.helpers import preprocess_texts


class Match(TypedDict):
    text: str
    start_idx: int
    end_idx: int


class Metadata(TypedDict, total=False):
    query_scores: Dict[str, float]


class SearchResult(TypedDict):
    rank: int
    score: float
    header: str
    content: str
    id: str
    metadata: Metadata
    matches: List[Match]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def vector_search(
    query: Union[str, List[str]],
    texts: List[str],
    embed_model: EmbedModelType,
    top_k: Optional[int] = None,
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[Dict]] = None,
    batch_size: int = 32,
) -> List[SearchResult]:
    """Perform vector search with chunking and return ranked results. Supports single query or list of queries."""
    # Validate inputs
    if ids is not None and len(ids) != len(texts):
        raise ValueError("Length of ids must match length of texts")
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("Length of metadatas must match length of texts")

    # Convert single query to list for uniform processing
    queries = [query] if isinstance(query, str) else query
    if not queries:
        raise ValueError("Query list cannot be empty")

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

    # Preprocess texts and queries
    preprocessed_texts = preprocess_texts(texts)
    preprocessed_queries = preprocess_texts(queries)

    # Generate embeddings for queries and all chunks
    embeddings = generate_embeddings(
        preprocessed_queries + preprocessed_texts,
        embed_model,
        return_format="numpy",
        show_progress=True,
        batch_size=batch_size
    )

    query_embeddings = embeddings[:len(queries)]
    chunk_embeddings = embeddings[len(queries):]

    # Calculate similarities and collect matches
    similarities = []
    for chunk_emb, (doc_idx, orig_text, doc_id, metadata) in zip(chunk_embeddings, chunk_to_doc):
        # Compute similarity for each query
        scores = [cosine_similarity(query_emb, chunk_emb)
                  for query_emb in query_embeddings]
        max_score = float(np.max(scores))
        # Create a new metadata dict with individual scores if query is a list
        new_metadata = metadata.copy()
        if isinstance(query, List):
            new_metadata['query_scores'] = {
                q: float(score) for q, score in zip(queries, scores)}

        # Find matches for query words/phrases in preprocessed text
        preprocessed_text = preprocessed_texts[doc_idx]
        matches: List[Match] = []
        for q in preprocessed_queries:
            # Split query into words for single-word matching
            query_words = q.split()
            text_lower = preprocessed_text.lower()
            for word in query_words:
                word_lower = word.lower()
                start_idx = 0
                while True:
                    start_idx = text_lower.find(word_lower, start_idx)
                    if start_idx == -1:
                        break
                    end_idx = start_idx + len(word_lower)
                    matches.append(Match(
                        text=word,
                        start_idx=start_idx,
                        end_idx=end_idx
                    ))
                    start_idx = end_idx
        similarities.append(
            (max_score, doc_idx, orig_text, doc_id, new_metadata, matches))

    # Aggregate scores by document (take max score across chunks)
    doc_scores = {}
    for score, doc_idx, orig_text, doc_id, metadata, matches in similarities:
        if doc_idx not in doc_scores or score > doc_scores[doc_idx][0]:
            doc_scores[doc_idx] = (score, orig_text, doc_id, metadata, matches)

    # Sort by score and create results
    if not top_k:
        top_k = len(texts)
    results = []
    for rank, (doc_idx, (score, orig_text, doc_id, metadata, matches)) in enumerate(
        sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)[
            :top_k], 1
    ):
        text_lines = orig_text.splitlines()
        header = text_lines[0] if text_lines else ""
        content = "\n".join(text_lines[1:]).strip()
        results.append(SearchResult(
            rank=rank,
            score=float(score),
            header=header,
            content=content,
            id=doc_id,
            metadata=metadata,
            matches=matches
        ))

    return results
