import os
from typing import List, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from jet.file.utils import load_file, save_file
from jet.scrapers.utils import clean_text
from rank_bm25 import BM25Plus
import numpy as np


class SearchResult(TypedDict):
    doc_index: int
    score: float
    text: str


def tfidf_search(corpus: List[str], query: str) -> List[SearchResult]:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([query])
    similarities = (tfidf_matrix * query_vector.T).toarray().flatten()
    ranked_indices = np.argsort(similarities)[::-1]
    ranked_scores = similarities[ranked_indices]

    results: List[SearchResult] = [
        {"doc_index": int(idx), "score": float(score), "text": corpus[idx]}
        for idx, score in zip(ranked_indices, ranked_scores)
    ]
    return results


def bm25_plus_search(corpus: List[str], query: str) -> List[SearchResult]:
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query = query.lower().split()
    # bm25 = BM25Plus(tokenized_corpus)

    # Set b=0 to disable length normalization
    # This prevents penalizing long documents
    bm25 = BM25Plus(tokenized_corpus, b=0.0)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = np.argsort(scores)[::-1]
    ranked_scores = scores[ranked_indices]

    results: List[SearchResult] = [
        {"doc_index": int(idx), "score": float(score), "text": corpus[idx]}
        for idx, score in zip(ranked_indices, ranked_scores)
    ]
    return results
