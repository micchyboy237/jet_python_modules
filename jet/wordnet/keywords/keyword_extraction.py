from typing import List, Optional, Tuple, Union, TypedDict, Literal
import os
import spacy
import uuid
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.wordnet.keywords.utils import preprocess_text
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
    min_count: int = 1,
) -> List[SimilarityResult]:
    logger.info(
        f"Reranking {len(texts)} documents using KeyBERT with min_count={min_count}")
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
    processed_texts = [preprocess_text(text) for text in texts]
    logger.debug(f"Processed texts: {processed_texts}")
    embed_model_obj = SentenceTransformerRegistry.load_model(embed_model)

    # Initialize vectorizer for counting keyword occurrences
    count_vectorizer = vectorizer or CountVectorizer(
        ngram_range=keyphrase_ngram_range, stop_words=stop_words)
    logger.debug(
        f"Initialized count_vectorizer with ngram_range={keyphrase_ngram_range}, stop_words={stop_words}")

    # Count keyword occurrences across documents if min_count > 1
    valid_keywords = None
    if min_count > 1:
        logger.debug(
            "Fitting vectorizer to processed texts for keyword counting")
        try:
            count_matrix = count_vectorizer.fit_transform(processed_texts)
            vocab = count_vectorizer.get_feature_names_out()
            logger.debug(f"Vocabulary size: {len(vocab)}, Vocabulary: {vocab}")
        except ValueError as e:
            logger.warning(
                f"Vectorization failed: {e}. Returning empty keywords.")
            return []

        keyword_counts = count_matrix.sum(axis=0).A1
        vocab_counts = {word: count for word,
                        count in zip(vocab, keyword_counts)}
        logger.debug(f"Keyword counts: {vocab_counts}")
        valid_keywords = [word for word,
                          count in vocab_counts.items() if count >= min_count]
        logger.debug(
            f"Valid keywords after min_count={min_count}: {valid_keywords}")

        if not valid_keywords:
            logger.warning(
                f"No keywords meet min_count={min_count}. Returning empty keywords.")
            return []

        # If candidates are provided, filter them by min_count
        if candidates:
            candidates = [
                cand for cand in candidates if cand in valid_keywords]
            logger.debug(f"Filtered candidates: {candidates}")
            if not candidates:
                logger.warning(
                    f"No candidates meet min_count={min_count}. Falling back to standard extraction.")
                candidates = None
    else:
        logger.debug("min_count=1, no keyword filtering applied")
        try:
            count_vectorizer.fit(processed_texts)
            vocab = count_vectorizer.get_feature_names_out()
            logger.debug(f"Vocabulary size: {len(vocab)}, Vocabulary: {vocab}")
        except ValueError as e:
            logger.warning(
                f"Vectorization failed: {e}. Returning empty keywords.")
            return []

    # Create a new vectorizer for keyword extraction if valid_keywords exists
    extraction_vectorizer = CountVectorizer(
        vocabulary=valid_keywords) if valid_keywords else count_vectorizer
    logger.debug(
        f"Using extraction_vectorizer with vocabulary size: {len(extraction_vectorizer.get_feature_names_out()) if hasattr(extraction_vectorizer, 'vocabulary_') else 'not fitted yet'}")

    if use_embeddings:
        logger.info("Using embedding-based keyword extraction")
        doc_embeddings = generate_embeddings(
            processed_texts, model=embed_model_obj, return_format="numpy")
        word_embeddings = generate_embeddings(
            vocab, model=embed_model_obj, return_format="numpy") if len(vocab) > 0 else None
        logger.debug(
            f"Doc embeddings shape: {doc_embeddings.shape}, Word embeddings shape: {word_embeddings.shape if word_embeddings is not None else 'None'}")
        try:
            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                seed_keywords=seed_keywords,
                doc_embeddings=doc_embeddings,
                word_embeddings=word_embeddings,
                vectorizer=extraction_vectorizer,
                top_n=top_n
            )
            logger.debug(f"Keywords extracted with embeddings: {keywords}")
        except Exception as e:
            logger.error(f"Error in extract_keywords with embeddings: {e}")
            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                vectorizer=extraction_vectorizer,
                top_n=top_n
            )
            logger.debug(f"Fallback keywords extracted: {keywords}")
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
            logger.debug(f"Keywords extracted with candidates: {keywords}")
        except Exception as e:
            logger.error(f"Error extracting keywords with candidates: {e}")
            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                top_n=top_n,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words
            )
            logger.debug(f"Fallback keywords extracted: {keywords}")
    elif vectorizer:
        logger.info("Using custom vectorizer for keyword extraction")
        keywords = keybert_model.extract_keywords(
            docs=processed_texts,
            seed_keywords=seed_keywords,
            vectorizer=extraction_vectorizer,
            top_n=top_n
        )
        logger.debug(f"Keywords extracted with custom vectorizer: {keywords}")
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
            vectorizer=extraction_vectorizer,
        )
        logger.debug(f"Keywords extracted with standard KeyBERT: {keywords}")

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
