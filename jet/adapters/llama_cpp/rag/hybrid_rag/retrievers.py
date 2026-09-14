import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .models import Document, RetrievalResult

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""

    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 50) -> RetrievalResult:
        pass


class DenseRetriever(BaseRetriever):
    """
    Dense (semantic) retriever using sentence-transformers + FAISS-like search.
    For simplicity we use brute-force cosine here.
    In production replace with FAISS / Qdrant / Weaviate / etc.
    """

    name = "dense"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        documents: Optional[List[Document]] = None,
    ):
        logger.info(f"Loading dense model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = documents or []
        self.embeddings: Optional[np.ndarray] = None

        if self.documents:
            self._build_index()

    def _build_index(self):
        texts = [d.content for d in self.documents]
        self.embeddings = self.model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        )
        # L2 normalize for cosine similarity
        self.embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        logger.info(f"Dense index built with {len(self.documents)} documents")

    def add_documents(self, documents: List[Document]):
        self.documents.extend(documents)
        self._build_index()

    def retrieve(self, query: str, top_k: int = 50) -> RetrievalResult:
        if not self.documents or self.embeddings is None:
            logger.warning("DenseRetriever has no documents")
            return RetrievalResult(self.name, [])

        query_emb = self.model.encode([query], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        # Cosine similarity
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            # Create a copy so we don't mutate the original
            results.append(
                Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata.copy(),
                    score=float(scores[idx]),
                )
            )

        logger.debug(f"Dense retrieved {len(results)} docs for query: {query[:60]}...")
        return RetrievalResult(self.name, results)


class BM25Retriever(BaseRetriever):
    """Sparse (keyword) retriever using BM25."""

    name = "bm25"

    def __init__(self, documents: Optional[List[Document]] = None):
        self.documents: List[Document] = documents or []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

        if self.documents:
            self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        # Simple whitespace + lower – replace with better tokenizer if needed
        return text.lower().split()

    def _build_index(self):
        self.tokenized_corpus = [self._tokenize(d.content) for d in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index built with {len(self.documents)} documents")

    def add_documents(self, documents: List[Document]):
        self.documents.extend(documents)
        self._build_index()

    def retrieve(self, query: str, top_k: int = 50) -> RetrievalResult:
        if not self.documents or self.bm25 is None:
            logger.warning("BM25Retriever has no documents")
            return RetrievalResult(self.name, [])

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(
                Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata.copy(),
                    score=float(scores[idx]),
                )
            )

        logger.debug(f"BM25 retrieved {len(results)} docs for query: {query[:60]}...")
        return RetrievalResult(self.name, results)
