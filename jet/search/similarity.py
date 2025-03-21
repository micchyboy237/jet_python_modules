import math
import numpy as np
from typing import List
from collections import Counter
from gensim.similarities.annoy import AnnoyIndexer
from gensim.models import TfidfModel
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.models import Word2Vec
from typing import Optional, TypedDict
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.gensim_scripts.phrase_detector import PhraseDetector
from jet.file.utils import load_file
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.wordnet.words import get_words
from shared.data_types.job import JobData


class SimilarityResult(TypedDict):
    id: str  # Document ID
    text: str  # The document's content/text
    score: float  # The score for how relevant the document is to the query
    similarity: float
    matched: list[str]  # List of queries that matched with this document


class BM25SimilarityResult(SimilarityResult):
    similarity: Optional[float]  # Similarity score for BM25 calculation


def get_bm25_similarities(queries: List[str], documents: List[str], ids: List[str], *, k1=1.2, b=0.75, delta=1.0) -> List[BM25SimilarityResult]:
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
        List[BM25SimilarityResult]: A list of dictionaries containing scores, similarities, matched queries, ids, and text.
    """

    # Tokenize documents into words
    tokenized_docs = [doc.split() for doc in documents]

    # Calculate the length of each document (i.e., number of tokens in it)
    doc_lengths = [len(doc) for doc in tokenized_docs]

    # Compute the average document length across all documents
    avg_doc_len = sum(doc_lengths) / len(doc_lengths)

    # Compute document frequency (DF) for each term across all documents
    df = {}
    total_docs = len(documents)

    # Calculate document frequency for each unique term in the corpus
    for doc in tokenized_docs:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] = df.get(term, 0) + 1

    # Precompute inverse document frequency (IDF) for each term
    idf = {term: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
           for term, freq in df.items()}

    # This will store the final results
    all_scores: list[BM25SimilarityResult] = []

    # Iterate through each document to compute its BM25 score
    for idx, doc in enumerate(tokenized_docs):
        doc_length = doc_lengths[idx]  # Length of the current document
        term_frequencies = Counter(doc)  # Frequency of terms in the document
        score = 0  # Accumulator for the total BM25 score for this document
        matched_queries = []  # List to track which queries matched with the document

        # Iterate through each query to compute its BM25 score against this document
        for query in queries:
            query_terms = query.split()  # Tokenize the query
            query_score = 0  # Accumulator for the query's score

            # Iterate through each term in the query and compute its contribution to the score
            for term in query_terms:
                if term in idf:  # Only process terms that appear in the IDF dictionary
                    # Term frequency in the document
                    tf = term_frequencies[term]
                    # BM25+ formula: TF-IDF component for the term
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * \
                        (1 - b + b * (doc_length / avg_doc_len)) + delta
                    query_score += idf[term] * (numerator / denominator)

            # If the query has a non-zero score, consider it as a matched query
            if query_score > 0:
                matched_queries.append(query)

            # Add the query's score to the total document score
            score += query_score

        # If the document has any relevance (score > 0), add it to the results
        if score > 0:
            all_scores.append({
                "id": ids[idx],  # Store the document ID
                "score": score,  # Raw BM25+ score (before normalization)
                "similarity": score,  # Retain original similarity for reference
                "matched": matched_queries,  # List of queries that matched
                "text": documents[idx]  # Store the document text itself
            })

    # Normalize the scores to the range [0, 1] based on the maximum score
    if all_scores:
        max_similarity = max(entry["score"]
                             for entry in all_scores)  # Find the maximum score
        for entry in all_scores:
            entry["score"] = entry["score"] / \
                max_similarity if max_similarity > 0 else 0  # Normalize score

    # Sort the results by the normalized score in descending order (higher score first)
    return sorted(all_scores, key=lambda x: x["score"], reverse=True)


def transform_corpus(sentences: list[str]):
    corpus = []
    for sentence in sentences:
        corpus.append(get_words(sentence))
    return corpus


def get_cosine_similarities(queries: list[str], sentences: list[str]) -> list[SimilarityResult]:
    corpus = transform_corpus(sentences)

    dictionary = Dictionary(corpus)
    bow_corpus = [dictionary.doc2bow(line) for line in corpus]
    index = SparseMatrixSimilarity(
        bow_corpus, num_docs=len(corpus), num_terms=len(dictionary)
    )

    bow_query = dictionary.doc2bow(queries)
    similarities = index[bow_query]

    results: list[SimilarityResult] = sorted(
        [
            {
                "score": float(score),
                "matched": [query for query in queries if query in " ".join(corpus[i])],
                "text": " ".join(corpus[i]),
            }
            for i, score in enumerate(similarities) if score > 0
        ],
        key=lambda x: x["score"], reverse=True
    )
    return results


def get_annoy_similarities(queries: list[str], sentences: list[str]) -> list[SimilarityResult]:
    corpus = transform_corpus(sentences)

    model = Word2Vec(sentences=corpus, vector_size=100,
                     window=5, min_count=1, workers=4)

    dictionary = Dictionary(corpus)
    tfidf = TfidfModel(dictionary=dictionary)

    indexer = AnnoyIndexer(model.wv, num_trees=2)
    termsim_index = WordEmbeddingSimilarityIndex(
        model.wv, kwargs={'indexer': indexer})
    similarity_matrix = SparseTermSimilarityMatrix(
        termsim_index, dictionary, tfidf)

    tfidf_corpus = [tfidf[dictionary.doc2bow(doc)] for doc in corpus]
    docsim_index = SoftCosineSimilarity(tfidf_corpus, similarity_matrix)

    results = []
    for query in queries:
        bow_query = dictionary.doc2bow(query.split())

        if not bow_query:
            logger.warning(
                f"Query '{query}' resulted in an empty BoW representation, skipping.")
            continue

        # Ensure sims is always an iterable array
        similarities = np.atleast_1d(docsim_index[bow_query])

        ranked_results = sorted(
            [
                {
                    "score": float(score),
                    "matched": [query for query in queries if query in " ".join(corpus[i])],
                    "text": " ".join(corpus[i]),
                }
                for i, score in enumerate(similarities) if score > 0
            ],
            key=lambda x: x["score"], reverse=True
        )
        results.extend(ranked_results)

    return results
