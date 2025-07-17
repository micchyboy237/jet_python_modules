from typing import List, Optional, Tuple, Union
import os
import spacy
import uuid
from tqdm import tqdm
from jet.data.header_utils._prepare_for_rag import preprocess_text as preprocess_text_for_rag
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.vectors.document_types import HeaderDocument
from jet.models.embeddings.base import generate_embeddings, load_embed_model
from jet.wordnet.keywords.helpers import SimilarityResult, _count_tokens, preprocess_texts, setup_keybert
from jet.logger import logger
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEFAULT_EMBED_MODEL: EmbedModelType = "all-MiniLM-L6-v2"


def rerank_by_keywords(
    texts: Union[List[str], List[List[str]]],
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
    show_progress: bool = True,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> List[SimilarityResult]:
    nlp = spacy.load("en_core_web_sm")
    keybert_model = keybert_model or setup_keybert(embed_model)

    if use_mmr and not (0.0 <= diversity <= 1.0):
        raise ValueError(
            "Diversity must be between 0.0 and 1.0 when use_mmr is True")
    if threshold is not None and not (0.0 <= threshold <= 1.0):
        raise ValueError("Threshold must be between 0.0 and 1.0")

    # Handle matrix input
    is_matrix = isinstance(
        texts, list) and texts and isinstance(texts[0], list)
    if is_matrix:
        # Validate matrix: all inner lists must have the same length
        text_lengths = [len(inner) for inner in texts]
        if len(set(text_lengths)) > 1:
            raise ValueError("All inner text lists must have the same length")
        flat_texts = [text for inner in texts for text in inner]
        doc_count = len(texts)
        texts_per_doc = text_lengths[0]
    else:
        flat_texts = texts
        doc_count = len(texts)
        texts_per_doc = 1

    if not flat_texts:
        logger.warning("Empty text list provided. Returning empty results.")
        return []

    doc_ids = ids if ids and len(ids) == doc_count else [
        str(uuid.uuid4()) for _ in range(doc_count)]

    processed_texts = [preprocess_text_for_rag(text) for text in flat_texts]
    embed_model_obj = SentenceTransformerRegistry.load_model(embed_model)

    count_vectorizer = vectorizer or CountVectorizer(
        ngram_range=keyphrase_ngram_range, stop_words=stop_words
    )

    valid_keywords = None
    if min_count > 1:
        try:
            count_matrix = count_vectorizer.fit_transform(processed_texts)
            vocab = count_vectorizer.get_feature_names_out()
        except ValueError as e:
            logger.warning(
                f"Vectorization failed: {e}. Returning empty keywords.")
            return []

        keyword_counts = count_matrix.sum(axis=0).A1
        vocab_counts = {word: count for word,
                        count in zip(vocab, keyword_counts)}
        valid_keywords = [word for word,
                          count in vocab_counts.items() if count >= min_count]

        if not valid_keywords:
            logger.warning(
                f"No keywords meet min_count={min_count}. Returning empty keywords.")
            return []

        if candidates:
            candidates = [
                cand for cand in candidates if cand in valid_keywords]
            if not candidates:
                logger.warning(
                    f"No candidates meet min_count={min_count}. Falling back to standard extraction.")
                candidates = None
    else:
        try:
            count_vectorizer.fit(processed_texts)
            vocab = count_vectorizer.get_feature_names_out()
        except ValueError as e:
            logger.warning(
                f"Vectorization failed: {e}. Returning empty keywords.")
            return []

    extraction_vectorizer = CountVectorizer(
        vocabulary=valid_keywords) if valid_keywords else count_vectorizer

    # # Handle seed_keywords for matrix input
    # flat_seed_keywords = []
    # if seed_keywords:
    #     if isinstance(seed_keywords[0], list):
    #         flat_seed_keywords = [
    #             keyword for sublist in seed_keywords for keyword in sublist]
    #     else:
    #         flat_seed_keywords = seed_keywords

    try:
        if use_embeddings:
            doc_embeddings = generate_embeddings(
                processed_texts, model=embed_model_obj, return_format="numpy", show_progress=True
            )
            word_embeddings = generate_embeddings(
                vocab, model=embed_model_obj, return_format="numpy", show_progress=True) if len(vocab) > 0 else None

            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                seed_keywords=seed_keywords,
                doc_embeddings=doc_embeddings,
                word_embeddings=word_embeddings,
                vectorizer=extraction_vectorizer,
                top_n=top_n * 2,
            )
        elif candidates:
            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                candidates=candidates,
                seed_keywords=seed_keywords,
                top_n=top_n * 2,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words,
            )
        elif vectorizer:
            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                seed_keywords=seed_keywords,
                vectorizer=extraction_vectorizer,
                top_n=top_n * 2,
            )
        else:
            keywords = keybert_model.extract_keywords(
                docs=processed_texts,
                seed_keywords=seed_keywords,
                top_n=top_n * 2,
                use_mmr=use_mmr,
                diversity=diversity,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words,
                vectorizer=extraction_vectorizer,
            )

        # # Filter keywords to only include those containing any seed keyword
        # if flat_seed_keywords:
        #     filtered_keywords = []
        #     for doc_keywords in keywords:
        #         filtered_doc_keywords = [
        #             (kw, score) for kw, score in doc_keywords
        #             if any(seed.lower() in kw.lower() for seed in flat_seed_keywords)
        #         ]
        #         filtered_keywords.append(filtered_doc_keywords[:top_n])
        #     keywords = filtered_keywords
        # else:
        #     keywords = [doc_keywords[:top_n] for doc_keywords in keywords]

        # Apply threshold filtering if specified
        if threshold is not None:
            keywords = [
                [(kw, score)
                 for kw, score in doc_keywords if score >= threshold]
                for doc_keywords in keywords
            ]

    except Exception as e:
        logger.error(f"Error during keyword extraction: {e}")
        return []

    # Aggregate results for matrix input
    result = []
    if is_matrix:
        iterator = range(doc_count)
        if show_progress:
            iterator = tqdm(iterator, desc="Aggregating matrix results")
        for i in iterator:
            doc_texts = texts[i]
            doc_keywords = keywords[i * texts_per_doc:(i + 1) * texts_per_doc]
            doc_scores = [max((score for _, score in kw_list), default=0.0)
                          for kw_list in doc_keywords]
            avg_score = float(np.mean(doc_scores)) if doc_scores else 0.0
            aggregated_keywords = []
            for kw_list in doc_keywords:
                aggregated_keywords.extend(
                    [{"text": kw, "score": score} for kw, score in kw_list])
            # Deduplicate keywords while preserving order
            seen = set()
            unique_keywords = [kw for kw in aggregated_keywords if not (
                kw["text"] in seen or seen.add(kw["text"]))]
            result_entry = {
                "id": doc_ids[i],
                "rank": 0,
                "doc_index": i,
                "score": avg_score,
                "text": " ".join(doc_texts),  # Join texts for display
                "tokens": sum(_count_tokens(text, nlp) for text in doc_texts),
                "keywords": unique_keywords[:top_n],
            }
            result.append(result_entry)
    else:
        iterator = zip(texts, keywords)
        if show_progress:
            iterator = tqdm(
                list(iterator), desc="Scoring and assembling results")
        for i, (original_text, doc_keywords) in enumerate(iterator):
            max_score = max((score for _, score in doc_keywords),
                            default=0.0) if doc_keywords else 0.0
            result_entry = {
                "id": doc_ids[i],
                "rank": 0,
                "doc_index": i,
                "score": max_score,
                "text": original_text,
                "tokens": _count_tokens(original_text, nlp),
                "keywords": [{"text": kw, "score": score} for kw, score in doc_keywords],
            }
            result.append(result_entry)

    sorted_results = sorted(result, key=lambda x: x['score'], reverse=True)

    if top_k is not None:
        sorted_results = sorted_results[:top_k]

    for rank, res in enumerate(sorted_results, 1):
        res['rank'] = rank

    return sorted_results
