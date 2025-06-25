import os
import spacy
import numpy as np
from typing import List, Optional, Tuple, Union
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from jet.models.model_types import EmbedModelType
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings, load_embed_model

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def setup_keybert(model_name: EmbedModelType = "static-retrieval-mrl-en-v1") -> KeyBERT:
    """Initialize KeyBERT with a specified model.

    Args:
        model_name: Name of the embedding model (default: static-retrieval-mrl-en-v1).

    Returns:
        KeyBERT: Initialized KeyBERT instance.
    """
    logger.info(f"Initializing KeyBERT with model: {model_name}")
    embed_model = load_embed_model(model_name)
    return KeyBERT(model=embed_model)


def extract_query_candidates(query: str, nlp=None) -> list[str]:
    """Extract candidate keywords from a query using spaCy NLP."""
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(query.lower())
    candidates = set()

    # Extract noun chunks and filter out those containing stop words
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        chunk_words = chunk_text.split()
        if len(chunk_words) <= 3:
            if all(not token.is_stop and token.pos_ in ["NOUN", "PROPN", "ADJ"] for token in chunk):
                candidates.add(chunk_text)
                # Add valid 2-word sub-phrases only if they are noun chunks
                if len(chunk_words) == 3:
                    for i in range(len(chunk_words) - 1):
                        sub_phrase = " ".join(chunk_words[i:i+2])
                        sub_doc = nlp(sub_phrase)
                        # Convert generator to list
                        sub_chunks = list(sub_doc.noun_chunks)
                        if sub_chunks and all(not token.is_stop for token in sub_doc):
                            candidates.add(sub_phrase)

    # Add single-word nouns and proper nouns
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            candidates.add(token.text)

    # Remove candidates that are prefixes of longer candidates
    final_candidates = set()
    for cand in candidates:
        is_prefix = any(
            cand != longer_cand and longer_cand.startswith(cand + " ")
            for longer_cand in candidates
        )
        if not is_prefix:
            final_candidates.add(cand)

    # Remove any candidates that are purely stop words
    final_candidates = {cand for cand in final_candidates if any(
        not nlp.vocab[word].is_stop for word in cand.split())}

    return list(final_candidates)


def extract_single_doc_keywords(
    doc: str,
    model: KeyBERT,
    top_n: int = 5,
    use_mmr: bool = False,
    diversity: float = 0.5
) -> List[Tuple[str, float]]:
    """Extract keywords from a single document.

    Args:
        doc: Input document text.
        model: Initialized KeyBERT model.
        top_n: Number of keywords to return (default: 5).
        use_mmr: Whether to use MMR for diversity (default: False).
        diversity: Diversity level for MMR (default: 0.5).

    Returns:
        List of tuples containing keywords and their similarity scores.
    """
    logger.info(
        f"Extracting keywords from single document (length: {len(doc)} chars)")
    keywords = model.extract_keywords(
        docs=doc,
        top_n=top_n,
        use_mmr=use_mmr,
        diversity=diversity
    )
    logger.debug(f"Extracted keywords: {keywords}")
    return keywords


def extract_multi_doc_keywords(
    docs: List[str],
    model: KeyBERT,
    top_n: int = 5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = "english"
) -> List[List[Tuple[str, float]]]:
    """Extract keywords from multiple documents.

    Args:
        docs: List of input document texts.
        model: Initialized KeyBERT model.
        top_n: Number of keywords to return per document (default: 5).
        keyphrase_ngram_range: Range of n-grams for keyphrases (default: (1, 2)).
        stop_words: Stop words to filter out (default: "english").

    Returns:
        List of lists of tuples containing keywords and their similarity scores.
    """
    logger.info(f"Extracting keywords from {len(docs)} documents")
    keywords = model.extract_keywords(
        docs=docs,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words=stop_words,
        top_n=top_n
    )
    logger.debug(f"Extracted keywords for {len(keywords)} documents")
    return keywords


def extract_keywords_with_candidates(
    doc: str,
    model: KeyBERT,
    candidates: List[str],
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """Extract keywords from a document using provided candidate keywords.

    Args:
        doc: Input document text.
        model: Initialized KeyBERT model.
        candidates: List of candidate keywords.
        top_n: Number of keywords to return (default: 5).

    Returns:
        List of tuples containing keywords and their similarity scores.
    """
    logger.info(f"Extracting keywords with {len(candidates)} candidates")
    keywords = model.extract_keywords(
        docs=doc,
        candidates=candidates,
        top_n=top_n
    )
    logger.debug(f"Extracted keywords: {keywords}")
    return keywords


def extract_keywords_with_custom_vectorizer(
    docs: Union[str, List[str]],
    model: KeyBERT,
    vectorizer: CountVectorizer,
    top_n: int = 5
) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
    """Extract keywords using a custom CountVectorizer.

    Args:
        docs: Single document or list of documents.
        model: Initialized KeyBERT model.
        vectorizer: Custom CountVectorizer instance.
        top_n: Number of keywords to return (default: 5).

    Returns:
        Keywords for single document or list of keywords for multiple documents.
    """
    logger.info(
        f"Extracting keywords with custom vectorizer for {1 if isinstance(docs, str) else len(docs)} document(s)")
    keywords = model.extract_keywords(
        docs=docs,
        vectorizer=vectorizer,
        top_n=top_n
    )
    logger.debug(f"Extracted keywords: {keywords}")
    return keywords


def extract_keywords_with_embeddings(
    docs: Union[str, List[str]],
    model: KeyBERT,
    top_n: int = 5,
    keyphrase_ngram_range: Tuple[int, int] = (1, 2)
) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
    """Extract keywords using precomputed embeddings.

    Args:
        docs: Single document (str) or list of documents (List[str]).
        model: Initialized KeyBERT model.
        top_n: Number of keywords to extract per document.
        keyphrase_ngram_range: N-gram range for keyword extraction (e.g., (1, 2)).

    Returns:
        List of (keyword, score) tuples for a single document, or list of such lists for multiple documents.

    Raises:
        ValueError: If docs is not a string or a list of strings.
    """
    logger.info(
        f"Extracting keywords with precomputed embeddings for {1 if isinstance(docs, str) else len(docs)} document(s)")

    # Validate input
    if not isinstance(docs, (str, list)):
        logger.error(
            f"Invalid input type: {type(docs)}. Expected str or List[str].")
        raise ValueError("Input must be a string or a list of strings")
    if isinstance(docs, list) and not all(isinstance(doc, str) for doc in docs):
        logger.error("All elements in docs must be strings.")
        raise ValueError("All elements in docs must be strings")

    # Handle empty input
    if not docs:
        return [] if isinstance(docs, list) else []

    # Generate document embeddings
    model_name = 'static-retrieval-mrl-en-v1'
    doc_embeddings = generate_embeddings(
        docs, model=model_name, return_format="numpy")

    # Initialize vectorizer
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

    # Generate word embeddings
    word_embeddings = generate_embeddings(
        vocab.tolist(), model=model_name, return_format="numpy")

    # Try extraction with precomputed embeddings
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

    # Fallback to default extraction if empty
    if not keywords:
        keywords = model.extract_keywords(
            docs=docs,
            vectorizer=vectorizer,
            top_n=top_n
        )

    return keywords
