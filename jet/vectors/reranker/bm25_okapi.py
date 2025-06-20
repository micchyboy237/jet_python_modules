from typing import List, Dict, Tuple
import math
import numpy as np
from rank_bm25 import BM25Okapi


def prepare_corpus() -> Tuple[List[List[str]], List[str]]:
    """
    Prepares a tokenized corpus and raw documents for BM25 ranking.

    Returns:
        Tuple containing tokenized corpus and original documents.
    """
    raw_corpus = [
        "Machine learning is a method of data analysis that automates analytical model building. It allows systems to learn from data and improve without explicit programming.",
        "Natural language processing involves the interaction between computers and human language. It enables machines to understand and generate human text effectively.",
        "Deep learning is a subset of machine learning using neural networks with many layers. It excels in tasks like image recognition and speech processing.",
        "Information retrieval systems rank documents based on their relevance to a user query. BM25 is a popular algorithm used in search engines for this purpose.",
        "Text mining extracts valuable insights from unstructured text data. It combines techniques from natural language processing and data mining."
    ]
    tokenized_corpus = [doc.lower().split() for doc in raw_corpus]
    return tokenized_corpus, raw_corpus


def get_top_k_results(bm25_model, query: List[str], raw_corpus: List[str], k: int = 3) -> List[Dict[str, any]]:
    """
    Retrieves top-k documents ranked by the BM25 model for a given query.

    Args:
        bm25_model: Initialized BM25Okapi model.
        query: Tokenized query terms.
        raw_corpus: Original documents for display.
        k: Number of top results to return.

    Returns:
        List of dictionaries containing document, score, and index.
    """
    scores = bm25_model.get_scores(query)
    scored_docs = [(score, idx) for idx, score in enumerate(scores)]
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [
        {"index": idx, "score": score, "document": raw_corpus[idx]}
        for score, idx in scored_docs[:k]
    ]


def inspect_idf(bm25_model, terms: List[str]) -> Dict[str, float]:
    """
    Inspects IDF values for given terms in the BM25 model.

    Args:
        bm25_model: Initialized BM25Okapi model.
        terms: List of terms to check IDF for.

    Returns:
        Dictionary mapping terms to their IDF values.
    """
    return {term: bm25_model.idf.get(term, 0) for term in terms}
