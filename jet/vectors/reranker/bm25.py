import math
from collections import Counter
from typing import List, Tuple
import numpy as np
from jet.wordnet.keywords.helpers import extract_query_candidates, preprocess_texts
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
    rank: int
    id: str  # Document ID
    score: float  # Normalized similarity score
    similarity: Optional[float]  # Raw BM25 similarity score
    text: str  # The document's content/text
    matched: dict[str, int]  # Query match counts
    metadata: Dict


class SimilarityResultOld(TypedDict):
    id: str  # Document ID
    text: str  # The document's content/text
    score: float  # Normalized similarity score
    similarity: Optional[float]  # Raw BM25 similarity score
    matched: dict[str, int]  # Query match counts
    matched_sentences: dict[str, List[Match]]  # Query to sentence matches


class SimilarityRequestData(TypedDict):
    queries: List[str]
    data_file: str


class SimilarityResultData(TypedDict):
    queries: list[str]
    count: int
    matched: dict[str, int]
    data: List[SimilarityResult]


def rerank_bm25(query: str, documents: List[str], ids: Optional[List[str]] = None, metadatas: Optional[List[Dict]] = None) -> Tuple[List[str], List[SimilarityResult]]:
    query_candidates = extract_query_candidates(query)
    results = get_bm25_similarities(
        query_candidates, documents, ids=ids, metadatas=metadatas)
    return query_candidates, results


def get_bm25_similarities(
    queries: List[str],
    documents: List[str],
    *,
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[Dict]] = None,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 1.0
) -> List[SimilarityResult]:
    """
    Compute BM25+ similarities between queries and a list of documents.

    Args:
        queries (List[str]): List of query strings.
        documents (List[str]): List of document strings.
        ids (Optional[List[str]]): Optional list of document IDs corresponding to the documents.
        metadatas (Optional[List[Dict]]): Optional list of metadata dictionaries corresponding to the documents.
        k1 (float): Term frequency scaling factor.
        b (float): Length normalization parameter.
        delta (float): BM25+ correction factor to reduce the bias against short documents.

    Returns:
        List[SimilarityResult]: A list of similarity results with rank, scores, match details, and optional metadata.
    """
    if not queries or not documents:
        raise ValueError("queries and documents must not be empty")

    if ids is None:
        ids = [generate_unique_hash() for _ in documents]
    elif len(documents) != len(ids):
        raise ValueError("documents and ids must have the same lengths")

    if metadatas is not None and len(documents) != len(metadatas):
        raise ValueError("documents and metadatas must have the same lengths")

    original_documents = documents  # Store original documents
    documents = preprocess_texts(documents)
    tokenized_docs = [get_words(doc) for doc in documents]
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
        matched: Dict[str, int] = {}
        metadata_text = ""

        # Stringify metadata values if provided
        if metadatas is not None and metadatas[idx]:
            metadata_text = " ".join(str(value)
                                     for value in metadatas[idx].values())

        for query in queries:
            query_terms = get_words(query)
            terms_present = True

            for term in query_terms:
                if term not in term_frequencies:
                    terms_present = False
                    break

            # Count exact phrase occurrences in the original document text and metadata
            pattern = re.compile(
                r'\b' + re.escape(query) + r'\b', re.IGNORECASE)
            match_count = len(pattern.findall(original_documents[idx]))
            if metadata_text:
                match_count += len(pattern.findall(metadata_text))
            if match_count > 0:  # Only include queries with non-zero matches
                matched[query] = match_count

            query_score = 0
            if terms_present:
                for term in query_terms:
                    if term in idf:
                        tf = term_frequencies[term]
                        numerator = tf * (k1 + 1)
                        denominator = tf + k1 * \
                            (1 - b + b * (doc_length / avg_doc_len)) + delta
                        query_score += idf[term] * (numerator / denominator)

            score += query_score

        result: SimilarityResult = {
            "rank": 0,
            "id": ids[idx],
            "score": score,
            "similarity": score,
            "matched": matched,
            "text": original_documents[idx],
            "metadata": metadatas[idx] if metadatas is not None else {}
        }
        all_scores.append(result)

    # Normalize scores if there are any non-zero scores
    if all_scores:
        max_similarity = max(entry["score"] for entry in all_scores)
        for entry in all_scores:
            entry["score"] = entry["score"] / \
                max_similarity if max_similarity > 0 else 0

    # Sort by number of matched queries (descending) only, do not sort by score
    all_scores.sort(key=lambda x: len(x["matched"]), reverse=True)

    # Assign ranks directly to sorted entries
    for rank, entry in enumerate(all_scores, 1):
        entry["rank"] = rank

    return all_scores


def get_bm25_similarities_old(queries: List[str], documents: List[str], ids: Optional[List[str]] = None, *, k1=1.2, b=0.75, delta=1.0) -> List[SimilarityResultOld]:
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
    all_scores: List[SimilarityResultOld] = []

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

            all_scores.append(SimilarityResultOld(
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


def get_bm25_similarities_oldest(
    queries: List[str], documents: List[str], *, ids: Optional[List[str]] = None, k1=1.2, b=0.75, delta=1.0
) -> List[SimilarityResultOld]:
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
        List[SimilarityResultOld]: A list of similarity results with scores and match details.
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
    all_scores: List[SimilarityResultOld] = []

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
            all_scores.append(SimilarityResultOld(
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
