import numpy as np
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
    text: str
    score: float


class BM25SimilarityResult(SimilarityResult):
    similarity: Optional[float]


def transform_corpus(sentences: list[str]):
    corpus = []
    for sentence in sentences:
        corpus.append(get_words(sentence))
    return corpus


def get_bm25_similarities(queries: list[str], sentences: list[str]) -> list[BM25SimilarityResult]:
    corpus = transform_corpus(sentences)

    dictionary = Dictionary(corpus)
    query_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
    document_model = OkapiBM25Model(dictionary=dictionary)

    bow_corpus = [dictionary.doc2bow(line) for line in corpus]
    bm25_corpus = document_model[bow_corpus]
    index = SparseMatrixSimilarity(
        bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
        normalize_queries=False, normalize_documents=False
    )

    bow_query = dictionary.doc2bow(queries)
    bm25_query = query_model[bow_query]
    similarities = index[bm25_query]

    max_similarity = max(similarities)

    results: list[BM25SimilarityResult] = sorted(
        [{"text": " ".join(corpus[i]), "score": float(score / max_similarity), "similarity": float(score)}
         for i, score in enumerate(similarities)],
        key=lambda x: x["score"], reverse=True
    )

    return results


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
        [{"text": " ".join(corpus[i]), "score": float(score)}
         for i, score in enumerate(similarities)],
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
        sims = np.atleast_1d(docsim_index[bow_query])

        ranked_results = sorted(
            [{"text": " ".join(corpus[i]), "score": float(score)}
             for i, score in enumerate(sims)],
            key=lambda x: x["score"], reverse=True
        )
        results.extend(ranked_results)

    return results
