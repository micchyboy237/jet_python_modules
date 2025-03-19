from typing import Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import Document
from tqdm import tqdm
from bs4 import BeautifulSoup
from jet.file.utils import load_file
from jet.logger import logger
from jet.logger.timer import time_it
from jet.search.transformers import clean_string
from jet.token.token_utils import get_token_counts_info, get_tokenizer, split_texts, token_counter
from jet.transformers.formatters import format_json
from jet.utils.commands import copy_to_clipboard
from jet.wordnet.sentence import adaptive_split
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class BertSearch:
    def __init__(self, model_name="paraphrase-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        self.index = None
        self.doc_texts = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def _setup_doc_texts(self, docs, chunk_size, chunk_overlap):
        for i, doc in tqdm(enumerate(docs), total=len(docs)):
            tokens: int = token_counter(doc, self.model_name)
            if tokens > chunk_size:
                splitter = SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    tokenizer=get_tokenizer(self.model_name).encode
                )
                splitted_docs = splitter.split_text(doc)
                self.doc_texts.extend(splitted_docs)
            else:
                self.doc_texts.append(doc)

    def build_index(self, docs, batch_size=32, chunk_size=200, chunk_overlap=50):
        self.doc_texts = []

        self._setup_doc_texts(docs, chunk_size, chunk_overlap)

        # Generate embeddings
        embeddings = self.model.encode(
            self.doc_texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
        )

        # Create FAISS index
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

        # Build TF-IDF Index
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.doc_texts)

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

        # Extract top documents
        top_texts = [self.doc_texts[idx]
                     for idx in indices[0] if idx < len(self.doc_texts)]

        # Compute TF-IDF Scores
        query_tfidf = self.tfidf_vectorizer.transform([query])
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
            }
            for i in sorted_indices
        ]

        return [{"rank": i + 1, **result} for i, result in enumerate(results)]


if __name__ == "__main__":
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
    query = "title, season, episode, synopsis, genre, release date, end date of \"I'll Become a Villainess Who Goes Down in History\" anime"
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
