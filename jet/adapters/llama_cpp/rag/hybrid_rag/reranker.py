import logging
from typing import List

from sentence_transformers import CrossEncoder

from .models import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Precision re-ranker using a cross-encoder.
    Default model is small and fast; swap for larger models in production.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10,
    ) -> List[Document]:
        if not documents:
            return []

        pairs = [[query, doc.content] for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc.rerank_score = float(score)

        ranked = sorted(documents, key=lambda d: d.rerank_score, reverse=True)
        result = ranked[:top_k]

        logger.info(
            f"Re-ranked {len(documents)} → top {len(result)} "
            f"(best score={result[0].rerank_score:.4f})"
        )
        return result
