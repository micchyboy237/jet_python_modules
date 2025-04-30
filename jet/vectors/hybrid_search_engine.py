import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, TypedDict, Optional
import numpy.typing as npt


class SearchResult(TypedDict):
    document: str
    score: float


class HybridSearchEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.dense_model: SentenceTransformer = SentenceTransformer(model_name)
        self.sparse_vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.documents: List[str] = []
        self.dense_embeddings: Optional[npt.NDArray[np.float32]] = None
        self.sparse_matrix: Optional[npt.NDArray[np.float64]] = None

    def fit(self, documents: List[str]) -> None:
        """
        Fit the search engine with a list of documents.

        Args:
            documents: List of documents to index
        """
        self.documents = documents
        self.dense_embeddings = self.dense_model.encode(
            documents, normalize_embeddings=True
        )
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(
            documents).toarray()

    def search(
        self,
        query: str,
        top_n: int = 5,
        alpha: float = 0.5,
        use_mmr: bool = True,
        lambda_param: float = 0.7
    ) -> List[SearchResult]:
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            top_n: Number of results to return
            alpha: Weight for combining dense and sparse scores (0 to 1)
            use_mmr: Whether to use Maximum Marginal Relevance reranking
            lambda_param: MMR trade-off parameter (0 to 1)

        Returns:
            List of dictionaries containing documents and their scores

        Raises:
            ValueError: If fit() hasn't been called
        """
        if not self.documents:
            raise ValueError("Call `fit(documents)` before searching.")

        # Encode query
        query_dense: npt.NDArray[np.float32] = self.dense_model.encode(
            [query], normalize_embeddings=True
        )
        query_sparse: npt.NDArray[np.float64] = self.sparse_vectorizer.transform([
                                                                                 query]).toarray()

        # Dense and Sparse similarities
        dense_scores: npt.NDArray[np.float32] = util.cos_sim(
            query_dense, self.dense_embeddings
        )[0].cpu().numpy()
        sparse_scores: npt.NDArray[np.float64] = (
            query_sparse @ self.sparse_matrix.T)[0]

        # Hybrid score
        hybrid_scores: npt.NDArray[np.float64] = alpha * \
            dense_scores + (1 - alpha) * sparse_scores
        # more candidates for reranking
        top_indices: npt.NDArray[np.int64] = hybrid_scores.argsort()[
            ::-1][:top_n * 2]

        if use_mmr:
            selected: List[int] = self.mmr_rerank(
                top_indices, query_dense, lambda_param, top_n
            )
            return [
                {"document": self.documents[i],
                    "score": float(hybrid_scores[i])}
                for i in selected
            ]

        sorted_indices: npt.NDArray[np.int64] = hybrid_scores.argsort()[
            ::-1][:top_n]
        return [
            {"document": self.documents[i], "score": float(hybrid_scores[i])}
            for i in sorted_indices
        ]

    def mmr_rerank(
        self,
        candidate_indices: npt.NDArray[np.int64],
        query_vec: npt.NDArray[np.float32],
        lambda_param: float,
        top_n: int
    ) -> List[int]:
        """
        Rerank candidates using Maximum Marginal Relevance.

        Args:
            candidate_indices: Indices of candidate documents
            query_vec: Dense embedding of the query
            lambda_param: Trade-off parameter between relevance and diversity
            top_n: Number of results to return

        Returns:
            List of selected document indices
        """
        selected: List[int] = []
        candidates: List[int] = list(candidate_indices)
        embeddings: npt.NDArray[np.float32] = self.dense_embeddings

        while len(selected) < top_n and candidates:
            mmr_scores: List[tuple[float, int]] = []
            for idx in candidates:
                sim_to_query: float = util.cos_sim(
                    query_vec, embeddings[idx]
                )[0].item()
                sim_to_selected: float = max(
                    [
                        util.cos_sim(embeddings[idx], embeddings[j])[0].item()
                        for j in selected
                    ],
                    default=0.0
                )
                mmr_score: float = lambda_param * sim_to_query - \
                    (1 - lambda_param) * sim_to_selected
                mmr_scores.append((mmr_score, idx))

            # Select document with max MMR score
            _, selected_idx = max(mmr_scores, key=lambda x: x[0])
            selected.append(selected_idx)
            candidates.remove(selected_idx)

        return selected
