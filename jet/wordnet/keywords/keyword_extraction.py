from typing import List, Optional, Tuple, Union, TypedDict, Literal
import os
import re
import spacy
import numpy as np
import uuid
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from jet.code.markdown_utils import parse_markdown
from jet.file.utils import load_file, save_file
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.vectors.document_types import HeaderDocument
from jet.models.embeddings.base import generate_embeddings, load_embed_model
from jet.wordnet.keywords.helpers import SimilarityResult, _count_tokens, setup_keybert
from jet.logger import logger

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEFAULT_EMBED_MODEL: EmbedModelType = "static-retrieval-mrl-en-v1"


def preprocess_text(text: str) -> str:
    """Preprocess text for keyword extraction while preserving original content."""
    logger.debug(f"Preprocessing text: {text}")
    # Remove excessive whitespace and normalize spaces
    cleaned = re.sub(r'\s+', ' ', text.strip())
    # Replace multiple punctuation marks with single (e.g., '!!!' -> '!')
    cleaned = re.sub(r'([!?.]){2,}', r'\1', cleaned)
    # Ensure consistent spacing around punctuation, but preserve decimal points
    cleaned = re.sub(r'\s*([,!?;:])\s*', r' \1 ', cleaned)
    cleaned = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2',
                     cleaned)  # Preserve decimal points
    # Ensure space before final punctuation if it exists
    if cleaned and cleaned[-1] in ',!?;.':
        cleaned = cleaned[:-1] + ' ' + cleaned[-1]
    logger.debug(f"Preprocessed text: {cleaned}")
    return cleaned


def rerank_by_keywords(
    texts: List[str],
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL,
    ids: Optional[List[str]] = None,
    seed_keywords: Optional[Union[List[str], List[List[str]]]] = None,
    candidates: Optional[List[str]] = None,
    vectorizer: Optional[CountVectorizer] = None,
    use_embeddings: bool = False,
    top_n: int = 5,
    use_mmr: bool = False,
    diversity: float = 0.5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = "english",
    keybert_model: Optional[KeyBERT] = None,
) -> List[SimilarityResult]:
    """
    Rerank a list of texts using KeyBERT keyword extraction.

    Args:
        texts: List of text documents to rerank.
        embed_model: Embedding model to use for keyword extraction.
        ids: Optional list of document IDs; if None or length mismatch, UUIDs are generated.
        seed_keywords: Optional seed keywords to guide extraction.
        candidates: Optional candidate keywords for constrained extraction.
        vectorizer: Optional custom CountVectorizer for keyword extraction.
        use_embeddings: If True, use precomputed embeddings for extraction.
        top_n: Number of keywords to extract per document.
        use_mmr: If True, use Maximal Marginal Relevance for keyword diversity.
        diversity: Diversity parameter for MMR (0.0 to 1.0).
        keyphrase_ngram_range: Tuple of (min, max) n-grams for keyphrases.
        stop_words: Stop words for keyword extraction (default: "english").
        keybert_model: Optional KeyBERT model instance.

    Returns:
        List of SimilarityResult objects, sorted by score with ranks assigned.
    """
    logger.info(f"Reranking {len(texts)} documents using KeyBERT")
    nlp = spacy.load("en_core_web_sm")
    keybert_model = keybert_model or setup_keybert(embed_model)
    if use_mmr and not (0.0 <= diversity <= 1.0):
        raise ValueError(
            "Diversity must be between 0.0 and 1.0 when use_mmr is True")
    if not texts:
        logger.warning("Empty text list provided. Returning empty results.")
        return []
    doc_ids = ids if ids and len(ids) == len(texts) else [
        str(uuid.uuid4()) for _ in texts]
    # Preprocess texts for keyword extraction, preserve original texts
    processed_texts = [preprocess_text(text) for text in texts]
    logger.debug(f"Processed texts: {processed_texts}")
    embed_model_obj = SentenceTransformerRegistry.load_model(embed_model)
    if use_embeddings:
        logger.info("Using embedding-based keyword extraction")
        doc_embeddings = generate_embeddings(
            processed_texts, model=embed_model_obj, return_format="numpy")
        vectorizer = vectorizer or CountVectorizer(
            ngram_range=keyphrase_ngram_range)
        try:
            vocab = vectorizer.fit(processed_texts).get_feature_names_out()
        except ValueError as e:
            logger.warning(
                f"Vectorization failed: {e}. Returning empty keywords.")
            return []
        if len(vocab) == 0:
            logger.warning(
                f"Empty vocabulary after vectorization for {len(processed_texts)} document(s).")
            return []
        word_embeddings = generate_embeddings(
            vocab.tolist(), model=embed_model_obj, return_format="numpy")
        try:
            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                seed_keywords=seed_keywords,
                doc_embeddings=doc_embeddings,
                word_embeddings=word_embeddings,
                vectorizer=vectorizer,
                top_n=top_n
            )
        except Exception as e:
            logger.error(f"Error in extract_keywords with embeddings: {e}")
            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                vectorizer=vectorizer,
                top_n=top_n
            )
    elif candidates:
        logger.info(
            f"Using candidate-based keyword extraction with {len(candidates)} candidates")
        try:
            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                candidates=candidates,
                seed_keywords=seed_keywords,
                top_n=top_n,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words,
            )
        except Exception as e:
            logger.error(f"Error extracting keywords with candidates: {e}")
            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                top_n=top_n,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words
            )
    elif vectorizer:
        logger.info("Using custom vectorizer for keyword extraction")
        keywords = keybert_model.extract_keywords(
            docs=processed_texts,
            seed_keywords=seed_keywords,
            vectorizer=vectorizer,
            top_n=top_n
        )
    else:
        logger.info("Using standard KeyBERT keyword extraction")
        keywords = keybert_model.extract_keywords(
            docs=processed_texts,
            seed_keywords=seed_keywords,
            top_n=top_n,
            use_mmr=use_mmr,
            diversity=diversity,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
        )
    logger.debug(
        f"Extracted keywords for {len(keywords)} documents: {keywords}")
    result = []
    for i, (original_text, doc_keywords) in enumerate(zip(texts, keywords)):
        max_score = max((score for _, score in doc_keywords),
                        default=0.0) if doc_keywords else 0.0
        result_entry = {
            "id": doc_ids[i],
            "rank": 0,
            "doc_index": i,
            "score": max_score,
            "text": original_text,
            "tokens": _count_tokens(original_text, nlp),
            "keywords": [{"text": kw, "score": score} for kw, score in doc_keywords]
        }
        logger.debug(f"Result entry for document {i}: {result_entry}")
        result.append(result_entry)
    sorted_results = sorted(result, key=lambda x: x['score'], reverse=True)
    for rank, res in enumerate(sorted_results, 1):
        res['rank'] = rank
    logger.debug(f"Final sorted results: {sorted_results}")
    return sorted_results
