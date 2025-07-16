import os
from typing import Any, Optional, TypedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from bs4 import BeautifulSoup
from jet.file.utils import load_file, save_file
from jet.logger import logger
from jet.logger.timer import time_it
from jet.search.formatters import clean_string
from jet.token.token_utils import get_token_counts_info, split_texts, token_counter
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.n_grams import extract_ngrams, get_most_common_ngrams
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from collections import Counter


class SearchResult(TypedDict):
    text: str
    score: float


class BertSearch:
    def __init__(self, model_name="paraphrase-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.doc_texts = []
        self.ngrams = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def _preprocess_text(self, texts: str | list[str]) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        ngrams = extract_ngrams([text.lower() for text in texts])
        return ngrams

    @time_it
    def build_index(self, docs: list[str], batch_size=32):
        self.doc_texts = split_texts(
            docs, self.model_name, chunk_size=200, chunk_overlap=50)

        self.ngrams = [self._preprocess_text(
            text) for text in tqdm(self.doc_texts)]
        ngrams_texts = [
            " ".join(ngrams) for ngrams in self.ngrams]

        # Generate embeddings
        embeddings = self.model.encode(
            ngrams_texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
        )

        # Create FAISS index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

        # Build TF-IDF Index
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(ngrams_texts)

    @time_it
    def search(self, query: str, top_k: Optional[int] = None, alpha=0.7):
        if self.index is None:
            raise ValueError("Index is empty! Call build_index() first.")

        top_k = min(top_k or len(self.doc_texts), len(self.doc_texts))

        # Encode query
        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

        # Perform FAISS search
        scores, indices = self.index.search(query_embedding, top_k)

        # Extract top results with scores
        results: list[SearchResult] = [
            {"text": self.doc_texts[idx], "score": scores[0][i]}
            for i, idx in enumerate(indices[0]) if idx < len(self.doc_texts)
        ]

        # Extract top documents
        top_texts = [self.doc_texts[idx]
                     for idx in indices[0] if idx < len(self.doc_texts)]

        # Compute TF-IDF Scores
        query_ngrams = self._preprocess_text(query)
        query_tfidf = self.tfidf_vectorizer.transform(query_ngrams)
        tfidf_scores = np.array(
            (query_tfidf @ self.tfidf_matrix.T).toarray()).flatten()

        # Combine FAISS + TF-IDF scores (weighted)
        combined_scores = alpha * scores[0] + \
            (1 - alpha) * tfidf_scores[indices[0]]

        # Sort by combined score
        sorted_indices = np.argsort(combined_scores)[::-1]
        results = [
            {
                "score": round(float(combined_scores[i]), 4),
                "tokens": token_counter(top_texts[i], self.model_name),
                "text": top_texts[i],
                "matched": get_most_common_ngrams([query, top_texts[i]], min_words=1, min_count=2)
            }
            for i in sorted_indices
        ]

        return [{"rank": i + 1, **result} for i, result in enumerate(results)]


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])

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
    search_engine = BertSearch()

    # Index the documents
    search_engine.build_index(docs)

    # Perform a search
    query = "Season and episode of \"I'll Become a Villainess Who Goes Down in History\" anime"
    top_k = 10

    results = search_engine.search(query, top_k=top_k)

    logger.info("Token Info:")
    token_info = get_token_counts_info(
        search_engine.doc_texts, search_engine.model_name)
    del token_info["results"]
    logger.debug(format_json(token_info))

    save_file({
        "query": query,
        "count": len(results),
        **token_info,
        "results": results,
    }, f"{output_dir}/results.json")

    for idx, result in enumerate(results[:10]):
        logger.log(f"{idx + 1}:", result["text"]
                   [:30], colors=["WHITE", "DEBUG"])
        logger.success(f"{result['score']:.2f}")
