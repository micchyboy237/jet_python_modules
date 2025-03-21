import math
from jet.data.utils import generate_unique_hash
from jet.wordnet.sentence import split_sentences
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
from jet.wordnet.gensim_scripts.phrase_detector import PhraseDetector
from jet.logger import logger
from jet.scrapers.utils import clean_newlines, clean_punctuations, clean_spaces
from jet.search.transformers import clean_string
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
    preprocessed_texts: list[str] = texts.copy()

    for idx, text in enumerate(preprocessed_texts):
        text = clean_newlines(text, max_newlines=1)
        text = clean_spaces(text)
        text = clean_string(text)
        text = clean_punctuations(text)
        text = lemmatize_text(text)

        preprocessed_texts[idx] = text

    return preprocessed_texts


def adjust_score_with_rewards_and_penalties(base_score: float, match_count: int, max_query_count: int) -> float:
    """
    Adjusts the score based on query match count using rewards and penalties.

    - Rewards: Boost score if more queries match (up to 50% boost).
    - Penalties: Reduce score if fewer queries match (up to 30% penalty).

    :param base_score: Original BM25 similarity score.
    :param match_count: Number of matched n-grams.
    :param max_query_count: Total number of queries.
    :return: Adjusted similarity score.
    """
    if max_query_count == 0:
        return base_score  # Avoid division by zero

    boost_factor = (match_count / max_query_count) * 0.5  # Max 50% boost
    penalty_factor = (1 - match_count / max_query_count) * \
        0.3  # Max 30% penalty

    return base_score * (1 + boost_factor - penalty_factor)


def get_bm25_similarities(queries: List[str], documents: List[str], ids: Optional[List[str]] = None, *, k1=1.2, b=0.75, delta=1.0) -> List[SimilarityResult]:
    if not queries or not documents:
        raise ValueError("queries and documents must not be empty")

    if ids is None:
        ids = [generate_unique_hash()
               for _ in documents]  # Generate unique IDs
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

    for idx, doc_text in enumerate(lowered_documents):
        orig_doc_text = documents[idx]
        doc_id = ids[idx]  # Use provided ID or generated unique ID
        sentences: list[str] = split_sentences(doc_text)
        doc_length = doc_lengths[idx]
        term_frequencies = Counter(tokenized_docs[idx])
        score = 0
        matched: dict[str, int] = {}  # Query match counts
        matched_sentences: dict[str, List[Match]] = {}

        match_count = 0  # Count of queries that matched in the document

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
                        sentence_score += idf[term] * (numerator / denominator)

                if query in sentence:
                    start_idx = orig_doc_text.lower().index(sentence)
                    end_idx = start_idx + len(sentence)
                    sentence = orig_doc_text[start_idx:end_idx]
                    matched_sentence_list.append(Match(
                        score=sentence_score,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        sentence=sentence,
                        text=doc_text,
                    ))

                query_score += sentence_score

            if query_score > 0 and matched_sentence_list:
                matched_sentences[query] = matched_sentence_list
                matched[query] = len(matched_sentence_list)
                match_count += 1

            score += query_score

        if score > 0:
            max_query_count = len(lowered_queries)
            adjusted_score = adjust_score_with_rewards_and_penalties(
                base_score=score,
                match_count=match_count,
                max_query_count=max_query_count
            )

            all_scores.append(SimilarityResult(
                id=doc_id,  # Use provided or generated unique ID
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
