import math
import re

from jet.wordnet.words import get_words
from tqdm import tqdm
from jet.data.utils import generate_unique_hash
from jet.utils.text import remove_non_alphanumeric
from jet.wordnet.sentence import adaptive_split, split_sentences
from jet.wordnet.similarity import filter_highest_similarity
import numpy as np
from typing import List
from collections import Counter
# from gensim.similarities.annoy import AnnoyIndexer
# from gensim.models import TfidfModel
# from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
# from gensim.models import Word2Vec
from typing import Optional, TypedDict
# from gensim.corpora import Dictionary
# from gensim.models import TfidfModel, OkapiBM25Model
# from gensim.similarities import SparseMatrixSimilarity
from jet.logger import logger
from jet.scrapers.utils import clean_newlines, clean_punctuations, clean_spaces
from jet.search.formatters import clean_string
from jet.wordnet.lemmatizer import lemmatize_text


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


def preprocess_reranker_texts(texts: str | list[str]) -> list[str]:
    if isinstance(texts, str):
        texts = [texts]

    # Lowercase
    # texts = [text.lower() for text in texts]
    preprocessed_texts: list[str] = texts.copy()

    for idx, text in enumerate(preprocessed_texts):
        text = clean_newlines(text, max_newlines=1)
        text = clean_spaces(text)
        text = clean_string(text)
        text = clean_punctuations(text)
        text = lemmatize_text(text)

        preprocessed_texts[idx] = text

    return preprocessed_texts


def transform_corpus(sentences: list[str]):
    from jet.wordnet.words import get_words

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
