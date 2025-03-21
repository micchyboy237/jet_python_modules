import faiss
from jet.search.similarity import preprocess_reranker_texts
from jet.wordnet.n_grams import count_ngrams, get_most_common_ngrams, get_ngrams
import numpy as np
from numpy import ndarray

from typing import Any, List, Optional, TypedDict
from jet.data.utils import generate_key, generate_unique_hash
from jet.logger.timer import time_it
from jet.scrapers.utils import clean_newlines, clean_punctuations, clean_spaces
from jet.utils.text import extract_substrings, find_sentence_indexes
from jet.wordnet.lemmatizer import lemmatize_text
from llama_index.core import Document
from jet.file.utils import load_file
from jet.search.transformers import clean_string
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.vectors.reranker.bm25 import (
    # SimilarityDataItem,
    SimilarityResultData,
    rerank_bm25,
)
from jet.logger import logger
from jet.wordnet.sentence import group_sentences, merge_sentences, split_sentences
from jet.wordnet.words import count_words, get_words
from sentence_transformers import SentenceTransformer


class SearchResult(TypedDict):
    text: str
    score: float


# class RerankResult(SimilarityDataItem):
#     metadata: dict[str, Any]
#     _matched_sentences: dict[str, list[str]]
#     _data: str


# class QueryResult(TypedDict):
#     queries: list[str]
#     count: int
#     matched: dict[str, int]
#     data: List[RerankResult]


def preprocess_texts(texts: str | list[str]) -> list[str]:
    return preprocess_reranker_texts(texts)


def split_text_by_docs(texts: list[str], max_tokens: int) -> list[Document]:
    docs: list[Document] = []

    for idx, text in enumerate(texts):
        token_count = count_words(text)
        if token_count > max_tokens:
            grouped_sentences = group_sentences(text, max_tokens)

            for sentence in grouped_sentences:
                start_idx = text.index(sentence.splitlines()[0])
                end_idx = start_idx + len(sentence)

                node_id = generate_unique_hash()

                docs.append(Document(
                    node_id=node_id,
                    text=sentence,
                    metadata={
                        "data_id": idx,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                    }
                ))
        else:
            node_id = generate_unique_hash()

            docs.append(Document(
                node_id=node_id,
                text=text,
                metadata={
                    "data_id": idx,
                    "start_idx": 0,
                    "end_idx": len(text),
                }
            ))

    return docs


def transform_queries_to_ngrams(query: str | list[str], ngrams: dict[str, int]) -> list[str]:
    queries = query
    if isinstance(queries, str):
        queries = [queries]

    # Lowercase
    queries = [text.lower() for text in queries]

    query_ngrams = get_most_common_ngrams(queries, min_count=1, min_words=1)
    query_ngrams = dict(sorted(query_ngrams.items(),
                        key=lambda item: len(item[0]), reverse=True))

    transformed_queries: list[str] = []
    current_query = "\n".join(queries)
    for ngram in query_ngrams:
        if ngram in ngrams and ngram in current_query:
            transformed_queries.append(ngram)
            current_query = current_query.replace(ngram, ' ').strip()

    return transformed_queries


# def search_and_rerank(query: str | List[str], texts: List[str], *, max_tokens: int = 200) -> QueryResult:
#     queries = query
#     if isinstance(queries, str):
#         queries = [queries]

#     data = preprocess_texts(texts)

#     docs = split_text_by_docs(data, max_tokens)

#     doc_texts = [doc.text for doc in docs]

#     queries = [
#         "Season",
#         "episode",
#         "synopsis",
#     ]

#     queries = preprocess_texts(queries)

#     # ids: List[str] = [doc.node_id for doc in docs]
#     ids: List[str] = [str(idx) for idx, doc in enumerate(docs)]

#     reranked_results = rerank_bm25(queries, doc_texts, ids)

#     results = []
#     for result in reranked_results["data"]:
#         idx = int(result["id"])
#         doc = docs[idx]
#         orig_data: str = data[doc.metadata["data_id"]]

#         matched = result["matched"]
#         matched_sentences: dict[str, list[str]] = {
#             key.lower(): [] for key in matched.keys()
#         }
#         for ngram, count in matched.items():
#             lowered_ngram = ngram.lower()
#             sentence_indexes = find_sentence_indexes(
#                 orig_data.lower(), lowered_ngram)
#             word_sentences = extract_substrings(orig_data, sentence_indexes)
#             matched_sentences[lowered_ngram] = [
#                 word_sentence for word_sentence in word_sentences
#                 if word_sentence.lower() in result["text"].lower()
#             ]

#         results.append({
#             **result,
#             "metadata": doc.metadata,
#             "_matched_sentences": matched_sentences,
#             "_data": orig_data,
#         })

#     copy_to_clipboard({
#         "query": " ".join(queries),
#         "count": reranked_results["count"],
#         "matched": reranked_results["matched"],
#         "data": results
#     })

#     response = QueryResult(
#         query=" ".join(queries),
#         count=reranked_results["count"],
#         matched=reranked_results["matched"],
#         data=results
#     )

#     return response


class HybridSearch:
    def __init__(self, model_name="paraphrase-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.data: list[str] = []
        self.docs: list[Document] = []
        self.ids: list[str] = []
        self.doc_texts: list[str] = []
        self.ngrams: dict[str, int] = {}
        self.embeddings: ndarray

    def _preprocess_text(self, texts: str | list[str]) -> list[str]:
        preprocessed_texts = preprocess_texts(texts)
        return preprocessed_texts

    @time_it
    def build_index(self, texts: list[str], max_tokens: Optional[int] = None, *, batch_size: int = 32):
        preprocessed_texts = self._preprocess_text(texts)

        if not max_tokens:
            max_tokens = self.model.max_seq_length

        self.data = texts
        self.docs = split_text_by_docs(preprocessed_texts, max_tokens)
        self.ids = [str(idx) for idx, doc in enumerate(self.docs)]
        self.doc_texts = [doc.text for doc in self.docs]

        self.ngrams = count_ngrams(
            [text.lower() for text in self.doc_texts], min_words=1)

        # Generate embeddings
        self.embeddings = self.model.encode(
            self.doc_texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
        )

        # Create FAISS index
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings)

    # @time_it
    # def semantic_search(self, query: str | List[str], top_k: Optional[int] = None) -> List[SearchResult]:
    #     if not self.doc_texts:
    #         return []

    #     top_k = min(top_k or len(self.doc_texts), len(self.doc_texts))

    #     # Encode query
    #     query_embedding = self.model.encode(
    #         [query], convert_to_numpy=True, normalize_embeddings=True
    #     )

    #     # Compute cosine similarity
    #     scores = util.cos_sim(query_embedding, self.doc_embeddings)[0]

    #     # Get top_k results
    #     top_results = np.argsort(-scores)[:top_k]

    #     return [
    #         {"text": self.doc_texts[idx], "score": float(scores[idx])}
    #         for idx in top_results
    #     ]

    @time_it
    def semantic_search(self, query: str | List[str], top_k: Optional[int] = None) -> List[SearchResult]:
        queries = self._preprocess_text(query)

        # queries = query
        # if isinstance(queries, str):
        #     queries = [queries]

        top_k = min(top_k or len(self.doc_texts), len(self.doc_texts))

        # Encode query
        query_embedding = self.model.encode(
            queries, convert_to_numpy=True, normalize_embeddings=True
        )

        # Perform FAISS search
        scores, indices = self.index.search(query_embedding, top_k)

        # Extract top results with scores
        results: List[SearchResult] = [
            {"text": self.doc_texts[idx], "score": scores[0][i]}
            for i, idx in enumerate(indices[0]) if idx < len(self.doc_texts)
        ]

        return results

    @time_it
    def rerank(self, query: str | List[str]) -> SimilarityResultData:
        queries = self._preprocess_text(query)

        # queries = query
        # if isinstance(queries, str):
        #     queries = [queries]

        # Transform queries
        queries = transform_queries_to_ngrams(
            [query.lower() for query in queries], self.ngrams)

        reranked_results = rerank_bm25(queries, self.doc_texts, self.ids)

        return reranked_results
        # results: list[RerankResult] = []
        # for result in reranked_results["data"]:
        #     idx = int(result["id"])
        #     doc = self.docs[idx]
        #     orig_data: str = self.data[doc.metadata["data_id"]]

        #     matched = result["matched"]
        #     matched_sentences: dict[str, list[str]] = {
        #         key.lower(): [] for key in matched.keys()
        #     }
        #     for ngram, count in matched.items():
        #         lowered_ngram = ngram.lower()
        #         sentence_indexes = find_sentence_indexes(
        #             orig_data.lower(), lowered_ngram)
        #         word_sentences = extract_substrings(
        #             orig_data, sentence_indexes)
        #         matched_sentences[lowered_ngram] = [
        #             word_sentence for word_sentence in word_sentences
        #             if word_sentence.lower() in result["text"].lower()
        #         ]

        #     results.append({
        #         **result,
        #         "metadata": doc.metadata,
        #         "_matched_sentences": matched_sentences,
        #         "_data": orig_data,
        #     })

        # response = QueryResult(
        #     queries=queries,
        #     count=reranked_results["count"],
        #     matched=reranked_results["matched"],
        #     data=results
        # )

        # return response

    def search(self, query: str, top_k: Optional[int] = None) -> SimilarityResultData:
        semantic_results = self.semantic_search(
            query, top_k=top_k)
        reranked_results = self.rerank(query)

        return reranked_results


if __name__ == "__main__":
    from jet.token.token_utils import get_token_counts_info

    # data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/generated/search_web_data/scraped_texts.json"
    data = load_file(data_file)
    docs = []
    for item in data:
        cleaned_sentence = clean_string(item)
        docs.append(cleaned_sentence)

    # Sample HTML docs
    # docs = [
    #     "<html><body><p>AI is transforming the world with deep learning.</p></body></html>",
    #     "<html><body><p>Quantum computing is the future of high-performance computing.</p></body></html>",
    #     "<html><body><p>Neural networks are a crucial part of artificial intelligence.</p></body></html>"
    # ]

    # Initialize search system
    search_engine = HybridSearch()

    # Index the documents
    search_engine.build_index(docs)

    # Perform a search
    query = "Season and episode"
    top_k = 10

    results = search_engine.search(query, top_k=top_k)

    logger.info("Token Info:")
    token_info = get_token_counts_info(
        search_engine.doc_texts, search_engine.model_name)
    del token_info["results"]
    logger.debug(format_json(token_info))

    copy_to_clipboard({
        "query": query,
        "count": len(results),
        **token_info,
        "data": results[:50]
    })

    for idx, result in enumerate(results[:10]):
        logger.log(f"{idx + 1}:", result["text"]
                   [:30], colors=["WHITE", "DEBUG"])
        logger.success(f"{result['score']:.2f}")
