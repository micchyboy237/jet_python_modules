from typing import List, TypedDict, Optional, Dict
import math
from collections import defaultdict
from jet.wordnet.words import get_words
from nltk.corpus import stopwords


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text.
        rank: Rank based on score (1 for highest, no skips).
        doc_index: Original index of the text in the input list.
        score: Normalized similarity score.
        text: The compared text (or chunk if long).
        tokens: Number of tokens from text.
        matched: Dictionary mapping matched query terms to their counts.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int
    matched: Dict[str, int]


class BM25PlusResult(TypedDict):
    """
    Represents the complete BM25+ result, including ranked documents and query match counts.

    Fields:
        results: List of SimilarityResult dictionaries.
        matched: Dictionary mapping query terms to their total counts across all documents.
    """
    results: List[SimilarityResult]
    matched: Dict[str, int]


def bm25_plus(corpus: List[str], query: str, doc_ids: Optional[List[str]] = None,
              k1: float = 1.5, b: float = 0.75, delta: float = 1.0) -> BM25PlusResult:
    """
    Compute BM25+ scores to rank documents based on their relevance to a query.

    Args:
        corpus: List of document texts to score.
        query: Search query string.
        doc_ids: Optional list of unique identifiers for documents; defaults to "doc_{index}".
        k1: Controls term frequency impact on score (default: 1.5).
        b: Controls document length impact on score (default: 0.75).
        delta: Constant added to scores for non-negative values (default: 1.0).

    Returns:
        BM25PlusResult containing ranked SimilarityResult list and query term match counts.
    """
    # Handle edge cases
    if not corpus or not query:
        return {
            "results": [
                {
                    "id": doc_ids[i] if doc_ids else f"doc_{i}",
                    "rank": 0,  # Will be calculated later
                    "doc_index": i,
                    "score": 0.0,
                    "text": doc,
                    "tokens": len(get_words(doc.lower())) if doc else 0,
                    "matched": {}
                }
                for i, doc in enumerate(corpus)
            ] if corpus else [],
            "matched": {}
        }

    # Load NLTK stopwords and add URL-specific stopwords
    try:
        nltk_stopwords = set(stopwords.words('english'))
    except LookupError:
        import nltk
        nltk.download('stopwords')
        nltk_stopwords = set(stopwords.words('english'))
    url_stopwords = {'https', 'www', 'com'}
    all_stopwords = nltk_stopwords | url_stopwords

    # Tokenize documents and query, preserving numbers
    def tokenize(text: str) -> List[str]:
        tokens: list[str] = get_words(text.lower())
        return [token for token in tokens if token not in all_stopwords]

    docs = [tokenize(doc) for doc in corpus]
    query_terms = tokenize(query)

    # Compute document lengths and average (after stopword removal)
    doc_count = len(docs)
    doc_lengths = [len(doc) for doc in docs]
    avg_doc_length = sum(doc_lengths) / doc_count if doc_count > 0 else 1.0

    # Compute document frequency (DF) and total match counts for query terms
    df = defaultdict(int)
    query_match_counts = defaultdict(int)
    for doc in docs:
        unique_terms = set(doc)
        for term in query_terms:
            if term in unique_terms:
                df[term] += 1
            # Count total occurrences across all documents
            query_match_counts[term] += doc.count(term)

    # Compute raw BM25+ scores and track matched terms
    scores = []
    matched_terms = []
    for doc_idx, doc in enumerate(docs):
        score = 0.0
        doc_matched = {}
        for term in query_terms:
            tf = doc.count(term)
            if tf > 0:  # Only include terms that appear in the document
                doc_matched[term] = tf
            # IDF with lower bound
            idf = math.log(
                (doc_count - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            if idf < 0:
                idf = 0.0
            # BM25+ score component
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * \
                (1 - b + b * (doc_lengths[doc_idx] / avg_doc_length))
            score += idf * (numerator / denominator + delta)
        scores.append(score)
        matched_terms.append(doc_matched)

    # Normalize scores
    max_score = max(scores, default=1.0)
    normalized_scores = [score / max_score if max_score >
                         0 else 0.0 for score in scores]

    # Create results list with initial ranks
    results = [
        {
            "id": doc_ids[i] if doc_ids else f"doc_{i}",
            "rank": 0,  # Temporary rank
            "doc_index": i,
            "score": normalized_scores[i],
            "text": corpus[i],
            "tokens": doc_lengths[i],
            "matched": dict(sorted(matched_terms[i].items(), key=lambda x: x[1], reverse=True))
        }
        for i in range(len(corpus))
    ]

    # Sort by score in descending order
    results.sort(key=lambda x: x["score"], reverse=True)

    # Assign unique ranks starting from 1
    for i, result in enumerate(results, 1):
        result["rank"] = i

    return {
        "matched": dict(sorted(query_match_counts.items(), key=lambda x: x[1], reverse=True)),
        "results": results,
    }


def bm25_plus_with_keyword_counts(
    corpus: List[str],
    keyword_counts: Dict[str, int],
    query: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 1.0,
    boost_factor: float = 1.5
) -> BM25PlusResult:
    """
    Compute BM25+ scores with keyword count-based boosting to rank documents based on their relevance.
    Documents containing keywords from keyword_counts are boosted based on the keyword's count, applied once per keyword.
    If query is not provided, keywords from keyword_counts are used as query terms.

    Args:
        corpus: List of document texts to score.
        keyword_counts: Dictionary mapping keywords to their counts for boosting.
        query: Optional search query string. If None, uses keywords from keyword_counts.
        doc_ids: Optional list of unique identifiers for documents; defaults to "doc_{index}".
        k1: Controls term frequency impact on score (default: 1.5).
        b: Controls document length impact on score (default: 0.75).
        delta: Constant added to scores for non-negative values (default: 1.0).
        boost_factor: Multiplier for keyword count-based boosting (default: 1.5).

    Returns:
        BM25PlusResult containing ranked SimilarityResult list and query term match counts.
    """
    # Handle edge cases
    if not corpus or (not query and not keyword_counts):
        return {
            "results": [
                {
                    "id": doc_ids[i] if doc_ids else f"doc_{i}",
                    "rank": 0,
                    "doc_index": i,
                    "score": 0.0,
                    "text": doc,
                    "tokens": len(get_words(doc.lower())) if doc else 0,
                    "matched": {}
                }
                for i, doc in enumerate(corpus)
            ] if corpus else [],
            "matched": {}
        }

    # Load NLTK stopwords and add URL-specific stopwords
    try:
        nltk_stopwords = set(stopwords.words('english'))
    except LookupError:
        import nltk
        nltk.download('stopwords')
        nltk_stopwords = set(stopwords.words('english'))
    url_stopwords = {'https', 'www', 'com'}
    all_stopwords = nltk_stopwords | url_stopwords

    # Tokenize documents and query, preserving numbers
    def tokenize(text: str) -> List[str]:
        tokens: list[str] = get_words(text.lower())
        return [token for token in tokens if token not in all_stopwords]

    docs = [tokenize(doc) for doc in corpus]
    query_terms = tokenize(query) if query else list(keyword_counts.keys())

    # Compute document lengths and average (after stopword removal)
    doc_count = len(docs)
    doc_lengths = [len(doc) for doc in docs]
    avg_doc_length = sum(doc_lengths) / doc_count if doc_count > 0 else 1.0

    # Compute document frequency (DF) and total match counts for query terms
    df = defaultdict(int)
    query_match_counts = defaultdict(int)
    for doc in docs:
        unique_terms = set(doc)
        for term in query_terms:
            if term in unique_terms:
                df[term] += 1
            query_match_counts[term] += doc.count(term)

    # Compute raw BM25+ scores with keyword boosting and track matched terms
    scores = []
    matched_terms = []
    for doc_idx, doc in enumerate(docs):
        score = 0.0
        doc_matched = {}
        unique_terms = set(doc)
        for term in query_terms:
            tf = doc.count(term)
            if tf > 0:
                doc_matched[term] = tf
            # IDF with lower bound
            idf = math.log(
                (doc_count - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            if idf < 0:
                idf = 0.0
            # BM25+ score component
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * \
                (1 - b + b * (doc_lengths[doc_idx] / avg_doc_length))
            term_score = idf * (numerator / denominator + delta)
            # Apply boost if term is in keyword_counts and appears in document
            if term in keyword_counts and term in unique_terms:
                term_score *= (1 +
                               math.log1p(keyword_counts[term]) * boost_factor)
            score += term_score
        scores.append(score)
        matched_terms.append(doc_matched)

    # Normalize scores
    max_score = max(scores, default=1.0)
    normalized_scores = [score / max_score if max_score >
                         0 else 0.0 for score in scores]

    # Create results list with initial ranks
    results = [
        {
            "id": doc_ids[i] if doc_ids else f"doc_{i}",
            "rank": 0,
            "doc_index": i,
            "score": normalized_scores[i],
            "text": corpus[i],
            "tokens": doc_lengths[i],
            "matched": dict(sorted(matched_terms[i].items(), key=lambda x: x[1], reverse=True))
        }
        for i in range(len(corpus))
    ]

    # Sort by score in descending order
    results.sort(key=lambda x: x["score"], reverse=True)

    # Assign unique ranks starting from 1
    for i, result in enumerate(results, 1):
        result["rank"] = i

    return {
        "matched": dict(sorted(query_match_counts.items(), key=lambda x: x[1], reverse=True)),
        "results": results,
    }
