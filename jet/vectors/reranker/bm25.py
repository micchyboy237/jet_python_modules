import math
from collections import Counter
from typing import List
import numpy as np
from jet.wordnet.similarity import filter_highest_similarity
from jet.wordnet.sentence import adaptive_split, split_sentences
from jet.utils.text import remove_non_alphanumeric
from jet.data.utils import generate_unique_hash
from tqdm import tqdm
from jet.wordnet.words import get_words
import re
import os
from jet.file.utils import load_file

from jet.search.formatters import clean_string
from typing import List, Dict, Any, Optional, TypedDict
from jet.transformers.formatters import format_json
from jet.wordnet.n_grams import count_ngrams, extract_ngrams, get_most_common_ngrams
from jet.wordnet.words import count_words, get_words
from shared.data_types.job import JobData
from jet.cache.cache_manager import CacheManager

cache_manager = CacheManager()


class Match(TypedDict):
    score: float
    start_idx: int
    end_idx: int
    sentence: str
    text: str


class SimilarityResult(TypedDict):
    id: str  # Document ID
    text: str  # The document's content/text
    score: float  # Normalized similarity score
    similarity: Optional[float]  # Raw BM25 similarity score
    matched: dict[str, int]  # Query match counts
    matched_sentences: dict[str, List[Match]]  # Query to sentence matches


class SimilarityRequestData(TypedDict):
    queries: List[str]
    data_file: str


# class SimilarityDataItem(TypedDict):
#     id: str
#     score: float
#     similarity: Optional[float]
#     words: int
#     matched: dict[str, int]
#     text: str


class SimilarityResultData(TypedDict):
    queries: list[str]
    count: int
    matched: dict[str, int]
    data: List[SimilarityResult]


def rerank_bm25(queries: list[str], sentences: list[str], ids: list[str]) -> SimilarityResultData:
    """Processes BM25+ similarity search by handling cache, cleaning data, generating n-grams, and computing similarities."""
    if not queries:  # Handle empty queries explicitly
        return {
            "queries": queries,
            "count": 0,
            "matched": {},
            "data": [],
        }
    similarity_results = get_bm25_similarities(queries, sentences, ids)
    similarity_results = [
        result for result in similarity_results if result["matched"]]
    matched = {query.lower(): 0 for query in queries}
    for result in similarity_results:
        result_matched = result["matched"]
        for match_query, match in result_matched.items():
            matched[match_query] += 1
    return {
        "queries": queries,
        "count": len(similarity_results),
        "matched": matched,
        "data": similarity_results,
    }


def get_bm25_similarities(queries: List[str], documents: List[str], ids: Optional[List[str]] = None, *, k1=1.2, b=0.75, delta=1.0) -> List[SimilarityResult]:
    if not queries or not documents:
        raise ValueError("queries and documents must not be empty")

    if ids is None:
        ids = [generate_unique_hash() for _ in documents]
    elif len(documents) != len(ids):
        raise ValueError("documents and ids must have the same lengths")

    lowered_queries = [query.lower() for query in queries]
    lowered_documents = [doc_text.lower() for doc_text in documents]

    tokenized_docs = [doc.split() for doc in lowered_documents]
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths)

    df = {}
    total_docs = len(lowered_documents)
    for doc in tokenized_docs:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1

    idf = {term: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
           for term, freq in df.items()}
    all_scores: List[SimilarityResult] = []

    for idx, doc_text in tqdm(enumerate(lowered_documents), total=len(lowered_documents), unit="doc"):
        orig_doc_text = documents[idx]
        doc_id = ids[idx]
        sentences: list[str] = split_sentences(doc_text)
        doc_length = doc_lengths[idx]
        term_frequencies = Counter(tokenized_docs[idx])
        score = 0
        matched: dict[str, int] = {}  # Phrase match counts
        matched_terms: dict[str, int] = {}  # Track matched term frequencies
        matched_sentences: dict[str, List[Match]] = {}

        for query in lowered_queries:
            query_terms = query.split()
            query_score = 0
            matched_sentence_list: List[Match] = []

            for sentence_idx, sentence in enumerate(sentences):
                sentence_terms = sentence.split()
                sentence_score = 0

                for term in query_terms:
                    if term in idf and term in sentence_terms:
                        tf = term_frequencies[term]
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * \
                            (1 - b + b * (doc_length / avg_doc_len)) + delta
                        term_score = idf[term] * (numerator / denominator)
                        sentence_score += term_score

                        # Track matched terms
                        matched_terms[term] = matched_terms.get(term, 0) + 1

                # Dynamic exact phrase match boost (scaled by query length)
                if re.search(rf'\b{re.escape(query)}\b', sentence):
                    sentence_score += 0.2 * len(query_terms)  # Scaled boost

                    sentence_to_match = adaptive_split(sentence)[0]
                    try:
                        start_idx = orig_doc_text.lower().index(sentence_to_match)
                    except:
                        lowered_orig_sentences = split_sentences(
                            orig_doc_text.lower())
                        matched_orig_sentence = filter_highest_similarity(
                            sentence_to_match, lowered_orig_sentences)
                        start_idx = orig_doc_text.lower().index(
                            matched_orig_sentence["text"])

                    end_idx = start_idx + len(sentence)
                    sentence = orig_doc_text[start_idx:end_idx]
                    matched_sentence_list.append(Match(
                        score=sentence_score,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        sentence=sentence,
                        text=orig_doc_text,
                    ))

                query_score += sentence_score

            if query_score > 0 and matched_sentence_list:
                matched_sentences[query] = matched_sentence_list
                matched[query] = len(matched_sentence_list)

            score += query_score

        if score > 0:
            adjusted_score = adjust_score_with_rewards_and_penalties(
                base_score=score,
                matched_terms=matched_terms,
                query_terms=lowered_queries,
                idf=idf
            )

            all_scores.append(SimilarityResult(
                id=doc_id,
                score=adjusted_score,
                similarity=score,
                matched=matched,
                matched_sentences=matched_sentences,
                text=orig_doc_text
            ))

    if all_scores:
        max_similarity = max(entry["score"] for entry in all_scores)
        for entry in all_scores:
            entry["score"] = entry["score"] / \
                max_similarity if max_similarity > 0 else 0

    return sorted(all_scores, key=lambda x: x["score"], reverse=True)


def get_bm25_similarities_old(
    queries: List[str], documents: List[str], *, ids: Optional[List[str]] = None, k1=1.2, b=0.75, delta=1.0
) -> List[SimilarityResult]:
    """
    Compute BM25+ similarities between queries and a list of documents.

    Args:
        queries (List[str]): List of query strings.
        documents (List[str]): List of document strings.
        ids (List[str]): List of document ids corresponding to the documents.
        k1 (float): Term frequency scaling factor.
        b (float): Length normalization parameter.
        delta (float): BM25+ correction factor to reduce the bias against short documents.

    Returns:
        List[SimilarityResult]: A list of similarity results with scores and match details.
    """

    if not queries or not documents:
        raise ValueError("queries and documents must not be empty")

    if ids is None:
        ids = [generate_unique_hash() for _ in documents]
    elif len(documents) != len(ids):
        raise ValueError("documents and ids must have the same lengths")

    tokenized_docs = [doc.split() for doc in documents]
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avg_doc_len = sum(doc_lengths) / len(doc_lengths)

    df = {}
    total_docs = len(documents)
    for doc in tokenized_docs:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1

    idf = {term: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
           for term, freq in df.items()}
    all_scores: List[SimilarityResult] = []

    for idx, doc in enumerate(tokenized_docs):
        doc_length = doc_lengths[idx]
        term_frequencies = Counter(doc)
        score = 0
        matched: dict[str, int] = {}
        matched_sentences: dict[str, List[Match]] = {}

        for query in queries:
            query_terms = query.split()
            query_score = 0

            for term in query_terms:
                if term in idf:
                    tf = term_frequencies[term]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * \
                        (1 - b + b * (doc_length / avg_doc_len)) + delta
                    query_score += idf[term] * (numerator / denominator)

            if query_score > 0:
                matched[query] = matched.get(
                    query, 0) + 1  # Count query occurrences

            score += query_score

        if score > 0:
            all_scores.append(SimilarityResult(
                id=ids[idx],
                score=score,
                similarity=score,
                matched=matched,
                # Empty since old version lacks sentence matching
                matched_sentences=matched_sentences,
                text=documents[idx],
            ))

    if all_scores:
        max_similarity = max(entry["score"] for entry in all_scores)
        for entry in all_scores:
            entry["score"] = entry["score"] / \
                max_similarity if max_similarity > 0 else 0

    return sorted(all_scores, key=lambda x: x["score"], reverse=True)


def adjust_score_with_rewards_and_penalties(base_score: float, matched_terms: dict[str, int], query_terms: List[str], idf: dict[str, float]) -> float:
    """Adjusts the base score with rewards for matched terms and penalties for missing terms."""
    if not query_terms:
        return base_score
    reward = sum(idf.get(term, 0) for term in matched_terms) * 0.8
    missing_terms_count = max(len(query_terms) - len(matched_terms), 0)
    penalty = math.log1p(missing_terms_count) / \
        math.log1p(len(query_terms)) if len(query_terms) > 0 else 0
    adjusted_score = base_score * (1 + reward) * (1 - penalty)
    return max(adjusted_score, 0)
