from typing import List, Optional, Tuple, Union, TypedDict, Literal
import numpy as np
import uuid
from sklearn.feature_extraction.text import CountVectorizer
from jet.models.model_registry.transformers.cross_encoder_model_registry import CrossEncoderRegistry
from jet.models.model_types import EmbedModelType
from jet.wordnet.keywords.helpers import preprocess_texts
from jet.wordnet.keywords.keyword_extraction import SimilarityResult, _count_tokens
from jet.logger import logger
import spacy


class CrossEncoderKeywordResult(TypedDict):
    text: str
    score: float


def extract_keywords_cross_encoder(
    texts: List[str],
    cross_encoder_model: EmbedModelType = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ids: Optional[List[str]] = None,
    candidates: Optional[List[str]] = None,
    vectorizer: Optional[CountVectorizer] = None,
    top_n: int = 5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = "english",
    min_df: int = 2,
) -> List[SimilarityResult]:
    """
    Extract keywords from texts using a cross-encoder model to score document-keyword pairs.

    Args:
        texts: List of input texts to extract keywords from.
        cross_encoder_model: Name of the cross-encoder model to use.
        ids: Optional list of document IDs.
        candidates: Optional list of candidate keywords/phrases.
        vectorizer: Optional custom CountVectorizer for candidate generation.
        top_n: Number of top keywords to return per document.
        keyphrase_ngram_range: Tuple specifying n-gram range for keyword extraction.
        stop_words: Stop words for vectorizer, default is "english".
        min_df: Minimum document frequency for candidate keywords (default: 2).

    Returns:
        List of SimilarityResult dictionaries containing ranked keywords with scores.
    """
    logger.info(
        f"Extracting keywords for {len(texts)} documents using cross-encoder: {cross_encoder_model}")

    nlp = spacy.load("en_core_web_sm")
    model = CrossEncoderRegistry.load_model(cross_encoder_model)

    if not texts:
        logger.warning("Empty text list provided. Returning empty results.")
        return []

    doc_ids = ids if ids and len(ids) == len(texts) else [
        str(uuid.uuid4()) for _ in texts]
    processed_texts = preprocess_texts(texts)

    # Use provided candidates or generate new ones
    final_candidates = candidates
    if final_candidates is None:
        vectorizer = vectorizer or CountVectorizer(
            ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
            # Adjust min_df for small text lists
            min_df=min_df if len(texts) >= min_df else 1
        )
        try:
            vocab = vectorizer.fit(processed_texts).get_feature_names_out()
            final_candidates = list(vocab)
        except ValueError as e:
            logger.warning(
                f"Vectorization failed: {e}. Returning empty keywords.")
            return []

    if not final_candidates:
        logger.warning(
            "No candidate keywords available. Returning empty results.")
        return []

    results = []
    for i, (text, doc_id) in enumerate(zip(processed_texts, doc_ids)):
        # Create document-keyword pairs for cross-encoder scoring
        pairs = [(text, candidate) for candidate in final_candidates]
        try:
            raw_scores = model.predict(pairs)
            # Use raw scores if already in [0, 1], otherwise normalize
            if np.all((raw_scores >= 0) & (raw_scores <= 1)):
                scores = raw_scores
            else:
                min_score, max_score = raw_scores.min(), raw_scores.max()
                if max_score != min_score:
                    scores = (raw_scores - min_score) / (max_score - min_score)
                else:
                    scores = np.ones_like(raw_scores) * 0.5
        except Exception as e:
            logger.error(
                f"Error in cross-encoder prediction for document {i}: {e}")
            scores = np.zeros(len(final_candidates))

        # Sort keywords by score
        keyword_scores = list(zip(final_candidates, scores))
        sorted_keywords = sorted(
            keyword_scores, key=lambda x: x[1], reverse=True)[:top_n]

        max_score = float(
            max((score for _, score in sorted_keywords), default=0.0))
        result_entry = {
            "id": doc_id,
            "rank": 0,
            "doc_index": i,
            "score": max_score,
            "text": texts[i],  # Original text, not preprocessed
            "tokens": _count_tokens(texts[i], nlp),
            "keywords": [{"text": kw, "score": float(score)} for kw, score in sorted_keywords]
        }
        results.append(result_entry)

    # Sort results by max keyword score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    for rank, res in enumerate(sorted_results, 1):
        res['rank'] = rank

    return sorted_results
