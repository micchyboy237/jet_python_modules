from typing import List, Optional, Tuple, Union, TypedDict
import os
import re
import spacy
import numpy as np
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from jet.models.model_types import EmbedModelType
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings, load_embed_model

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class KeywordResult(TypedDict):
    doc_index: int
    score: float
    text: str
    document: str


def setup_keybert(model_name: EmbedModelType = "static-retrieval-mrl-en-v1") -> KeyBERT:
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
    doc: str,
    model: KeyBERT,
    top_n: int = 5,
    use_mmr: bool = False,
    diversity: float = 0.5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = "english"
) -> List[KeywordResult]:
    logger.info(
        f"Extracting keywords from single document (length: {len(doc)} chars)")
    keywords = model.extract_keywords(
        docs=doc,
        top_n=top_n,
        use_mmr=use_mmr,
        diversity=diversity,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
    )
    logger.debug(f"Extracted keywords: {keywords}")
    return [{"doc_index": 0, "score": score, "text": kw, "document": doc} for kw, score in keywords]


def extract_multi_doc_keywords(
    docs: List[str],
    model: KeyBERT,
    seed_keywords: Union[List[str], List[List[str]]] = None,
    top_n: int = 5,
    use_mmr: bool = False,
    diversity: float = 0.5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = "english"
) -> List[List[KeywordResult]]:
    logger.info(f"Extracting keywords from {len(docs)} documents")
    keywords = model.extract_keywords(
        docs=docs,
        seed_keywords=seed_keywords,
        top_n=top_n,
        use_mmr=use_mmr,
        diversity=diversity,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
    )
    logger.debug(f"Extracted keywords for {len(keywords)} documents")
    return [[{"doc_index": i, "score": score, "text": kw, "document": docs[i]} for kw, score in doc_keywords]
            for i, doc_keywords in enumerate(keywords)]


def extract_keywords_with_candidates(
    docs: Union[str, List[str]],
    model: KeyBERT,
    candidates: List[str],
    top_n: int = 5
) -> Union[List[KeywordResult], List[List[KeywordResult]]]:
    logger.info(f"Extracting keywords with {len(candidates)} candidates")
    try:
        keywords = model.extract_keywords(
            docs=docs,
            candidates=candidates,
            top_n=top_n
        )
        logger.debug(f"Extracted keywords: {keywords}")
        if isinstance(docs, str):
            # Handle single document case
            if all(isinstance(kw, tuple) for kw in keywords):
                return [{"doc_index": 0, "score": score, "text": kw, "document": docs} for kw, score in keywords]
            else:
                # Fallback for when KeyBERT returns only keywords without scores
                return [{"doc_index": 0, "score": 1.0, "text": kw, "document": docs} for kw in keywords]
        else:
            # Handle multi-document case
            if all(isinstance(doc_kws, list) and all(isinstance(kw, tuple) for kw in doc_kws) for doc_kws in keywords):
                return [[{"doc_index": i, "score": score, "text": kw, "document": docs[i]} for kw, score in doc_keywords]
                        for i, doc_keywords in enumerate(keywords)]
            else:
                # Fallback for when KeyBERT returns only keywords without scores
                return [[{"doc_index": i, "score": 1.0, "text": kw, "document": docs[i]} for kw in doc_keywords]
                        for i, doc_keywords in enumerate(keywords)]
    except Exception as e:
        logger.error(f"Error extracting keywords with candidates: {e}")
        # Fallback to standard extraction without candidates
        keywords = model.extract_keywords(docs=docs, top_n=top_n)
        logger.debug(f"Fallback extracted keywords: {keywords}")
        if isinstance(docs, str):
            return [{"doc_index": 0, "score": score, "text": kw, "document": docs} for kw, score in keywords]
        return [[{"doc_index": i, "score": score, "text": kw, "document": docs[i]} for kw, score in doc_keywords]
                for i, doc_keywords in enumerate(keywords)]


def extract_keywords_with_custom_vectorizer(
    docs: Union[str, List[str]],
    model: KeyBERT,
    vectorizer: CountVectorizer,
    seed_keywords: Union[List[str], List[List[str]]] = None,
    top_n: int = 5
) -> Union[List[KeywordResult], List[List[KeywordResult]]]:
    logger.info(
        f"Extracting keywords with custom vectorizer for {1 if isinstance(docs, str) else len(docs)} document(s)")
    keywords = model.extract_keywords(
        docs=docs,
        seed_keywords=seed_keywords,
        vectorizer=vectorizer,
        top_n=top_n
    )
    logger.debug(f"Extracted keywords: {keywords}")
    if isinstance(docs, str):
        return [{"doc_index": 0, "score": score, "text": kw, "document": docs} for kw, score in keywords]
    return [[{"doc_index": i, "score": score, "text": kw, "document": docs[i]} for kw, score in doc_keywords]
            for i, doc_keywords in enumerate(keywords)]


def extract_keywords_with_embeddings(
    docs: Union[str, List[str]],
    model: KeyBERT,
    top_n: int = 5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2)
) -> Union[List[KeywordResult], List[List[KeywordResult]]]:
    logger.info(
        f"Extracting keywords with precomputed embeddings for {1 if isinstance(docs, str) else len(docs)} document(s)")
    if not isinstance(docs, (str, list)):
        logger.error(
            f"Invalid input type: {type(docs)}. Expected str or List[str].")
        raise ValueError("Input must be a string or a list of strings")
    if isinstance(docs, list) and not all(isinstance(doc, str) for doc in docs):
        logger.error("All elements in docs must be strings.")
        raise ValueError("All elements in docs must be strings")
    if not docs:
        return [] if isinstance(docs, list) else []
    model_name = 'static-retrieval-mrl-en-v1'
    doc_embeddings = generate_embeddings(
        docs, model=model_name, return_format="numpy")
    vectorizer = CountVectorizer(ngram_range=keyphrase_ngram_range)
    try:
        vocab = vectorizer.fit([docs] if isinstance(
            docs, str) else docs).get_feature_names_out()
    except ValueError as e:
        logger.warning(f"Vectorization failed: {e}. Returning empty keywords.")
        return [] if isinstance(docs, str) else [[] for _ in docs]
    if len(vocab) == 0:
        logger.warning(
            f"Empty vocabulary after vectorization for {len(docs if isinstance(docs, list) else 1)} document(s). Returning empty keywords.")
        return [] if isinstance(docs, str) else [[] for _ in docs]
    word_embeddings = generate_embeddings(
        vocab.tolist(), model=model_name, return_format="numpy")
    try:
        keywords = model.extract_keywords(
            docs=docs,
            doc_embeddings=doc_embeddings,
            word_embeddings=word_embeddings,
            vectorizer=vectorizer,
            top_n=top_n
        )
    except Exception as e:
        logger.error(f"Error in extract_keywords with embeddings: {e}")
        keywords = []
    if not keywords:
        keywords = model.extract_keywords(
            docs=docs,
            vectorizer=vectorizer,
            top_n=top_n
        )
    if isinstance(docs, str):
        return [{"doc_index": 0, "score": score, "text": kw, "document": docs} for kw, score in keywords]
    return [[{"doc_index": i, "score": score, "text": kw, "document": docs[i]} for kw, score in doc_keywords]
            for i, doc_keywords in enumerate(keywords)]
