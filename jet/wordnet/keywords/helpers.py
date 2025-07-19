from tqdm import tqdm
from jet.wordnet.n_grams import count_ngrams
# from jet.wordnet.pos_tagger import POSTagger
from jet.wordnet.pos_tagger_light import POSTagger
from jet.scrapers.utils import clean_newlines, clean_punctuations, clean_spaces
from jet.search.formatters import clean_string
from jet.wordnet.lemmatizer import lemmatize_text
import nltk
from nltk.corpus import stopwords
from typing import List, Optional, Tuple, Union, TypedDict, Literal
import os
import re
import sklearn
import spacy
import numpy as np
import uuid
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.wordnet.words import get_words
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from jet.code.markdown_utils import parse_markdown
from jet.file.utils import load_file, save_file
from jet.models.model_types import EmbedModelType, LLMModelType
from jet.vectors.document_types import HeaderDocument
from jet.models.embeddings.base import generate_embeddings, load_embed_model
from jet.logger import logger

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
DEFAULT_EMBED_MODEL: EmbedModelType = "all-MiniLM-L6-v2"


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


def preprocess_texts(texts: str | list[str]) -> list[str]:

    # Download stopwords if not already downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    if isinstance(texts, str):
        texts = [texts]

    # Lowercase
    texts = [text.lower() for text in texts]
    preprocessed_texts: list[str] = texts.copy()
    stop_words = set(stopwords.words('english'))

    tagger = POSTagger()

    for idx, text in enumerate(tqdm(preprocessed_texts, desc="Preprocessing texts")):
        # Filter words by tags not in includes_pos
        includes_pos = ["PROPN", "NOUN", "VERB", "ADJ", "ADV"]

        text = clean_newlines(text, max_newlines=1)
        text = clean_punctuations(text)
        text = clean_spaces(text)
        text = clean_string(text)

        preprocessed_lines = []
        for line in text.splitlines():
            pos_results = tagger.filter_pos(line, includes_pos)
            filtered_text = [pos_result['word'] for pos_result in pos_results]
            text = " ".join(filtered_text).lower()

            # Remove stopwords
            words = get_words(line)
            filtered_words = [
                word for word in words if word.lower() not in stop_words]
            preprocessed_lines.append(' '.join(filtered_words))
        text = '\n'.join(preprocessed_lines)

        preprocessed_texts[idx] = text

    return preprocessed_texts


def setup_keybert(model_name: EmbedModelType = DEFAULT_EMBED_MODEL) -> KeyBERT:
    logger.info(f"Initializing KeyBERT with model: {model_name}")
    embed_model = SentenceTransformerRegistry.load_model(model_name)
    return KeyBERT(model=embed_model)


def extract_query_candidates(query: Union[str, List[str]], ngram_range: Tuple[int, int] = (1, 2)) -> List[str]:
    """Extract candidate keywords from a query using spaCy NLP, including years."""
    if isinstance(query, str):
        query = [query]
    texts = preprocess_texts(query)
    min_words, max_words = ngram_range
    all_ngrams = count_ngrams([text.lower()
                               for text in texts], min_words=min_words, max_words=max_words)
    # Sort candidates by ngram count in descending order, then alphabetically
    sorted_candidates = sorted(
        all_ngrams.items(), key=lambda x: (-x[1], x[0])
    )
    candidates = [ngram for ngram, count in sorted_candidates]
    return candidates


def extract_keyword_candidates(
    texts: List[str] | str,  # Allow single string or list of strings
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str = "english",
    min_df: int = 2,
    top_n: Optional[int] = None,
) -> List[str]:
    """
    Extract candidate keywords from a list of texts or a single text using CountVectorizer.

    Args:
        texts: List of input texts or a single text string.
        ngram_range: Tuple specifying n-gram range for keyword extraction.
        stop_words: Stop words for vectorizer, default is "english".
        min_df: Minimum document frequency for candidate keywords (default: 2).
        top_n: If specified, return only the top_n most frequent candidates.

    Returns:
        List of candidate keywords as strings.
    """
    logger.debug(f"scikit-learn version: {sklearn.__version__}")

    # Convert single string to list if necessary
    if isinstance(texts, str):
        texts = [texts]
    logger.debug(f"Input texts: {texts}")
    logger.debug(
        f"Parameters: ngram_range={ngram_range}, stop_words={stop_words}, min_df={min_df}, top_n={top_n}")

    # Set effective_min_df: use 1 for single text, else respect min_df
    effective_min_df = 1 if len(texts) == 1 else min(min_df, len(texts))
    logger.debug(f"Effective min_df: {effective_min_df}")

    # Use NLTK stop words to ensure consistency
    stop_words_list = stopwords.words(
        'english') if stop_words == "english" else stop_words
    logger.debug(f"Stop words: {stop_words_list[:10]}...")

    # Extended stop words for bigram filtering only
    extended_stop_words = set(stop_words_list).union(
        {'is', 'are', 'key', 'performance', 'improve'})
    logger.debug(f"Extended stop words: {list(extended_stop_words)[:10]}...")

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=list(stop_words_list),
        min_df=effective_min_df,
        token_pattern=r'(?u)\b\w+\b'
    )
    try:
        X = vectorizer.fit_transform(texts)
        logger.debug(f"X shape: {X.shape}")
        vocab = vectorizer.get_feature_names_out()
        logger.debug(f"Vocabulary: {list(vocab)}")

        # Sum term frequencies across all documents
        freqs = X.sum(axis=0).A1
        logger.debug(f"Frequencies: {freqs.tolist()}")

        # Create list of (term, frequency, ngram_length) tuples
        term_freqs = [(vocab[i], freqs[i], len(get_words(vocab[i])))
                      for i in range(len(vocab))]
        logger.debug(f"Term-Frequency pairs: {term_freqs}")

        # Filter and score terms
        valid_term_freqs = []
        unigram_freqs = {term: freq for term, freq,
                         ngram_len in term_freqs if ngram_len == 1}
        logger.debug(f"Unigram frequencies: {unigram_freqs}")

        for term, freq, ngram_len in term_freqs:
            words = get_words(term)
            if ngram_len > 1:
                is_significant = all(word not in extended_stop_words and
                                     word in unigram_freqs and
                                     unigram_freqs[word] >= effective_min_df
                                     for word in words)
                if not is_significant:
                    logger.debug(f"Filtered out insignificant bigram: {term}")
                    continue
            score = freq * (3.0 if ngram_len == 2 else 1.0)
            valid_term_freqs.append((term, score, ngram_len))
            logger.debug(
                f"Scored term: {term}, score: {score}, ngram_len: {ngram_len}")

        logger.debug(
            f"Valid term-frequency pairs before sorting: {valid_term_freqs}")

        # Sort by score (descending), then ngram length (bigrams first), then alphabetically
        valid_term_freqs.sort(key=lambda x: (-x[1], -x[2], x[0]))
        logger.debug(f"Sorted term-frequency pairs: {valid_term_freqs}")

        # Extract sorted terms
        final_candidates = [term for term, _, _ in valid_term_freqs]
        logger.debug(f"Final candidates before top_n: {final_candidates}")

        # Apply top_n filter if specified
        if top_n is not None and top_n > 0:
            final_candidates = final_candidates[:top_n]
        logger.debug(f"Final candidates after top_n: {final_candidates}")

        return final_candidates
    except ValueError as e:
        logger.warning(f"Vectorization failed: {e}. Returning empty keywords.")
        return []


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
