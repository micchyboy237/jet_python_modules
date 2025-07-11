from typing import List, Optional, Tuple, Union, TypedDict, Literal
import os
import re
import spacy
import numpy as np
import uuid
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from jet.code.markdown_utils import parse_markdown
from jet.file.utils import load_file, save_file
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.vectors.document_types import HeaderDocument
from jet.models.embeddings.base import generate_embeddings, load_embed_model
from jet.logger import logger

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEFAULT_EMBED_MODEL: EmbedModelType = "static-retrieval-mrl-en-v1"


class Keyword(TypedDict):
    """
    Represents a single keyword and its score.

    Fields:
        text: The keyword or phrase.
        score: The similarity score of the keyword.
    """
    text: str
    score: float


class SimilarityResult(TypedDict):
    """
    Represents a single keybert result for a text.

    Fields:
        id: Original document id
        rank: Rank based on score (1 for highest, no skips).
        doc_index: Original index of the text in the input list.
        score: Normalized similarity score.
        text: The compared text (or chunk if long).
        tokens: Number of tokens from text.
        keywords: List of top n keywords with their scores.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int
    keywords: List[Keyword]


def _count_tokens(text: str, nlp=None) -> int:
    """Count the number of tokens in a text using spaCy."""
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return len([token for token in doc if not token.is_space])


def setup_keybert(model_name: EmbedModelType = DEFAULT_EMBED_MODEL) -> KeyBERT:
    logger.info(f"Initializing KeyBERT with model: {model_name}")
    embed_model = load_embed_model(model_name)
    return KeyBERT(model=embed_model)


def extract_query_candidates(query: str, nlp=None) -> List[str]:
    """Extract candidate keywords from a query using spaCy NLP, including years."""
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(query.lower())
    candidates = set()
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        chunk_words = chunk_text.split()
        if len(chunk_words) <= 3:
            if all(not token.is_stop and token.pos_ in ["NOUN", "PROPN", "ADJ", "NUM"] for token in chunk):
                candidates.add(chunk_text)
                if len(chunk_words) == 3:
                    for i in range(len(chunk_words) - 1):
                        sub_phrase = " ".join(chunk_words[i:i+2])
                        sub_doc = nlp(sub_phrase)
                        sub_chunks = list(sub_doc.noun_chunks)
                        if sub_chunks and all(not token.is_stop for token in sub_doc):
                            candidates.add(sub_phrase)
    for token in doc:
        if (token.pos_ in ["NOUN", "PROPN"] and not token.is_stop) or \
           (token.pos_ == "NUM" and re.match(r"^\d{4}$", token.text)):
            candidates.add(token.text)
    final_candidates = set()
    for cand in candidates:
        is_prefix = any(
            cand != longer_cand and longer_cand.startswith(cand + " ")
            for longer_cand in candidates
        )
        if not is_prefix:
            final_candidates.add(cand)
    final_candidates = {cand for cand in final_candidates if any(
        not nlp.vocab[word].is_stop for word in cand.split())}
    return list(final_candidates)


def extract_single_doc_keywords(
    text: str,
    model: KeyBERT,
    id: Optional[str] = None,
    seed_keywords: Union[List[str], List[List[str]]] = None,
    top_n: int = 5,
    use_mmr: bool = False,
    diversity: float = 0.5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = "english"
) -> List[SimilarityResult]:
    logger.info(
        f"Extracting keywords from single document (length: {len(text)} chars)")
    nlp = spacy.load("en_core_web_sm")
    keywords = model.extract_keywords(
        docs=text,
        seed_keywords=seed_keywords,
        top_n=top_n,
        use_mmr=use_mmr,
        diversity=diversity,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
    )
    logger.debug(f"Extracted keywords: {keywords}")
    max_score = max((score for _, score in keywords), default=0.0)
    doc_id = id if id is not None else str(uuid.uuid4())
    return [{
        "id": doc_id,
        "rank": 1,
        "doc_index": 0,
        "score": max_score,
        "text": text,
        "tokens": _count_tokens(text, nlp),
        "keywords": [{"text": kw, "score": score} for kw, score in keywords]
    }]


def extract_multi_doc_keywords(
    texts: List[str],
    model: KeyBERT,
    ids: Optional[List[str]] = None,
    seed_keywords: Union[List[str], List[List[str]]] = None,
    top_n: int = 5,
    use_mmr: bool = False,
    diversity: float = 0.5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = "english"
) -> List[SimilarityResult]:
    logger.info(f"Extracting keywords from {len(texts)} documents")
    nlp = spacy.load("en_core_web_sm")
    keywords = model.extract_keywords(
        docs=texts,
        seed_keywords=seed_keywords,
        top_n=top_n,
        use_mmr=use_mmr,
        diversity=diversity,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
    )
    logger.debug(f"Extracted keywords for {len(keywords)} documents")
    # Generate IDs if not provided or length mismatch
    doc_ids = ids if ids and len(ids) == len(texts) else [
        str(uuid.uuid4()) for _ in texts]
    result = []
    for i, (text, doc_keywords) in enumerate(zip(texts, keywords)):
        max_score = max((score for _, score in doc_keywords),
                        default=0.0) if doc_keywords else 0.0
        result.append({
            "id": doc_ids[i],
            "rank": 0,  # Will be updated after sorting
            "doc_index": i,
            "score": max_score,
            "text": text,
            "tokens": _count_tokens(text, nlp),
            "keywords": [{"text": kw, "score": score} for kw, score in doc_keywords]
        })
    # Assign ranks based on sorted scores (1 for highest, no skips)
    sorted_results = sorted(result, key=lambda x: x['score'], reverse=True)
    for rank, res in enumerate(sorted_results, 1):
        res['rank'] = rank
    return sorted_results


def extract_keywords_with_candidates(
    texts: List[str],
    model: KeyBERT,
    candidates: List[str],
    ids: Optional[List[str]] = None,
    seed_keywords: Union[List[str], List[List[str]]] = None,
    top_n: int = 5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = "english"
) -> List[SimilarityResult]:
    logger.info(
        f"Extracting keywords with {len(candidates)} candidates for {len(texts)} documents")
    nlp = spacy.load("en_core_web_sm")
    try:
        keywords = model.extract_keywords(
            docs=texts,
            candidates=candidates,
            seed_keywords=seed_keywords,
            top_n=top_n,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
        )
        logger.debug(f"Extracted keywords: {keywords}")
    except Exception as e:
        logger.error(f"Error extracting keywords with candidates: {e}")
        keywords = model.extract_keywords(docs=texts, top_n=top_n)
        logger.debug(f"Fallback extracted keywords: {keywords}")
    # Generate IDs if not provided or length mismatch
    doc_ids = ids if ids and len(ids) == len(texts) else [
        str(uuid.uuid4()) for _ in texts]
    result = []
    for i, (text, doc_keywords) in enumerate(zip(texts, keywords)):
        max_score = max((score for _, score in doc_keywords),
                        default=0.0) if doc_keywords else 0.0
        result.append({
            "id": doc_ids[i],
            "rank": 0,  # Will be updated after sorting
            "doc_index": i,
            "score": max_score,
            "text": text,
            "tokens": _count_tokens(text, nlp),
            "keywords": [{"text": kw, "score": score} for kw, score in doc_keywords]
        })
    # Assign ranks based on sorted scores (1 for highest, no skips)
    sorted_results = sorted(result, key=lambda x: x['score'], reverse=True)
    for rank, res in enumerate(sorted_results, 1):
        res['rank'] = rank
    return sorted_results


def extract_keywords_with_custom_vectorizer(
    texts: List[str],
    model: KeyBERT,
    vectorizer: CountVectorizer,
    ids: Optional[List[str]] = None,
    seed_keywords: Union[List[str], List[List[str]]] = None,
    top_n: int = 5
) -> List[SimilarityResult]:
    logger.info(
        f"Extracting keywords with custom vectorizer for {len(texts)} document(s)")
    nlp = spacy.load("en_core_web_sm")
    keywords = model.extract_keywords(
        docs=texts,
        seed_keywords=seed_keywords,
        vectorizer=vectorizer,
        top_n=top_n
    )
    logger.debug(f"Extracted keywords: {keywords}")
    # Generate IDs if not provided or length mismatch
    doc_ids = ids if ids and len(ids) == len(texts) else [
        str(uuid.uuid4()) for _ in texts]
    result = []
    for i, (text, doc_keywords) in enumerate(zip(texts, keywords)):
        max_score = max((score for _, score in doc_keywords),
                        default=0.0) if doc_keywords else 0.0
        result.append({
            "id": doc_ids[i],
            "rank": 0,  # Will be updated after sorting
            "doc_index": i,
            "score": max_score,
            "text": text,
            "tokens": _count_tokens(text, nlp),
            "keywords": [{"text": kw, "score": score} for kw, score in doc_keywords]
        })
    # Assign ranks based on sorted scores (1 for highest, no skips)
    sorted_results = sorted(result, key=lambda x: x['score'], reverse=True)
    for rank, res in enumerate(sorted_results, 1):
        res['rank'] = rank
    return sorted_results


def extract_keywords_with_embeddings(
    texts: List[str],
    model: KeyBERT,
    ids: Optional[List[str]] = None,
    seed_keywords: Union[List[str], List[List[str]]] = None,
    top_n: int = 5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2)
) -> List[SimilarityResult]:
    logger.info(
        f"Extracting keywords with precomputed embeddings for {len(texts)} document(s)")
    nlp = spacy.load("en_core_web_sm")
    if not texts:
        return []
    model_name = DEFAULT_EMBED_MODEL
    doc_embeddings = generate_embeddings(
        texts, model=model_name, return_format="numpy")
    vectorizer = CountVectorizer(ngram_range=keyphrase_ngram_range)
    try:
        vocab = vectorizer.fit(texts).get_feature_names_out()
    except ValueError as e:
        logger.warning(f"Vectorization failed: {e}. Returning empty keywords.")
        return []
    if len(vocab) == 0:
        logger.warning(
            f"Empty vocabulary after vectorization for {len(texts)} document(s). Returning empty keywords.")
        return []
    word_embeddings = generate_embeddings(
        vocab.tolist(), model=model_name, return_format="numpy")
    try:
        keywords = model.extract_keywords(
            docs=texts,
            seed_keywords=seed_keywords,
            doc_embeddings=doc_embeddings,
            word_embeddings=word_embeddings,
            vectorizer=vectorizer,
            top_n=top_n
        )
    except Exception as e:
        logger.error(f"Error in extract_keywords with embeddings: {e}")
        keywords = model.extract_keywords(
            docs=texts,
            vectorizer=vectorizer,
            top_n=top_n
        )
    # Generate IDs if not provided or length mismatch
    doc_ids = ids if ids and len(ids) == len(texts) else [
        str(uuid.uuid4()) for _ in texts]
    result = []
    for i, (text, doc_keywords) in enumerate(zip(texts, keywords)):
        max_score = max((score for _, score in doc_keywords),
                        default=0.0) if doc_keywords else 0.0
        result.append({
            "id": doc_ids[i],
            "rank": 0,  # Will be updated after sorting
            "doc_index": i,
            "score": max_score,
            "text": text,
            "tokens": _count_tokens(text, nlp),
            "keywords": [{"text": kw, "score": score} for kw, score in doc_keywords]
        })
    # Assign ranks based on sorted scores (1 for highest, no skips)
    sorted_results = sorted(result, key=lambda x: x['score'], reverse=True)
    for rank, res in enumerate(sorted_results, 1):
        res['rank'] = rank
    return sorted_results
