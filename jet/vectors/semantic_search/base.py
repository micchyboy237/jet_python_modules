import re
from typing import Optional, TypedDict, List, Dict, Union
import numpy as np
from jet.data.utils import generate_unique_id
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_types import EmbedModelType
from jet.wordnet.keywords.helpers import preprocess_texts
from jet.wordnet.words import get_words
from .search_types import Match, SearchResult
from .booster import boost_ngram_score
from jet.logger import logger


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
    if ids is not None and len(ids) != len(texts):
        raise ValueError("Length of ids must match length of texts")
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("Length of metadatas must match length of texts")
    queries = [query] if isinstance(query, str) else query
    if not queries:
        raise ValueError("Query list cannot be empty")
    if ids is None:
        ids = [generate_unique_id() for _ in texts]
    if metadatas is None:
        metadatas = [{} for _ in texts]
    chunk_to_doc = []
    for doc_idx, (text, doc_id, metadata) in enumerate(zip(texts, ids, metadatas)):
        chunk_to_doc.append((doc_idx, text, doc_id, metadata))
    preprocessed_texts = preprocess_texts(texts)
    preprocessed_queries = preprocess_texts(queries)
    embeddings = generate_embeddings(
        preprocessed_queries + preprocessed_texts,
        embed_model,
        return_format="numpy",
        show_progress=True,
        batch_size=batch_size
    )
    query_embeddings = embeddings[:len(queries)]
    chunk_embeddings = embeddings[len(queries):]
    similarities = []
    # Inside the loop for each chunk
    for chunk_emb, (doc_idx, orig_text, doc_id, metadata) in zip(chunk_embeddings, chunk_to_doc):
        scores = [cosine_similarity(query_emb, chunk_emb)
                  for query_emb in query_embeddings]
        max_score = float(np.max(scores))
        logger.debug(f"Chunk {doc_id}: max_score={max_score}, scores={scores}")
        new_metadata = metadata.copy()
        if isinstance(query, List):
            new_metadata['query_scores'] = {
                q: float(score) for q, score in zip(queries, scores)}
        preprocessed_text = preprocessed_texts[doc_idx]
        matches: List[Match] = []
        text_lower = preprocessed_text.lower()

        def generate_ngrams(query: str, max_n: int) -> List[str]:
            words = get_words(query)
            ngrams = []
            for n in range(1, min(max_n, len(words)) + 1):
                for i in range(len(words) - n + 1):
                    ngrams.append(" ".join(words[i:i+n]))
            return ngrams

        all_search_terms = []
        for q in preprocessed_queries:
            max_words = len(get_words(q))
            ngrams = generate_ngrams(q, max_words)
            all_search_terms.extend([(ngram, len(ngram)) for ngram in ngrams])
        all_search_terms = sorted(
            all_search_terms, key=lambda x: x[1], reverse=True)
        logger.debug(f"Chunk {doc_id}: search_terms={all_search_terms}")

        matches: List[Match] = []
        text_lower = preprocessed_text.lower()
        for term, term_len in all_search_terms:
            term_lower = term.lower()
            # Use regex with word boundaries to ensure exact term matches
            pattern = rf'\b{re.escape(term_lower)}\b'
            for match in re.finditer(pattern, text_lower):
                start_idx = match.start()
                end_idx = match.end()
                matches.append(Match(
                    text=term,
                    start_idx=start_idx,
                    end_idx=end_idx
                ))

        matches.sort(key=lambda m: (m["start_idx"], -m["end_idx"]))
        filtered_matches: List[Match] = []
        last_end = -1
        for match in matches:
            if match["start_idx"] >= last_end:
                filtered_matches.append(match)
                last_end = match["end_idx"]
        logger.debug(f"Chunk {doc_id}: matches={filtered_matches}")

        boosted_score = boost_ngram_score(filtered_matches, max_score)
        logger.debug(
            f"Chunk {doc_id}: boosted_score={boosted_score}, max_score={max_score}, match_lengths={[m['end_idx'] - m['start_idx'] for m in filtered_matches]}")
        similarities.append(
            (boosted_score, doc_idx, orig_text, doc_id, new_metadata, filtered_matches))

    doc_scores = {}
    for score, doc_idx, orig_text, doc_id, metadata, matches in similarities:
        logger.debug(f"Doc {doc_idx}: score={score}, id={doc_id}")
        if doc_idx not in doc_scores or score > doc_scores[doc_idx][0]:
            doc_scores[doc_idx] = (score, orig_text, doc_id, metadata, matches)
    if not top_k:
        top_k = len(texts)
    results = []
    sorted_docs = sorted(doc_scores.items(),
                         key=lambda x: x[1][0], reverse=True)[:top_k]
    logger.debug(
        f"Sorted docs: {[(doc_idx, score) for doc_idx, (score, _, _, _, _) in sorted_docs]}")
    for rank, (doc_idx, (score, orig_text, doc_id, metadata, matches)) in enumerate(sorted_docs, 1):
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
    logger.debug(
        f"Final results: {[(r['id'], r['score'], r['matches']) for r in results]}")
    return results
