import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util


class HybridSearchEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.dense_model = SentenceTransformer(model_name)
        self.sparse_vectorizer = TfidfVectorizer()
        self.documents = []
        self.dense_embeddings = None
        self.sparse_matrix = None

    def fit(self, documents):
        self.documents = documents
        self.dense_embeddings = self.dense_model.encode(
            documents, normalize_embeddings=True)
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(documents)

    def search(self, query, top_n=5, alpha=0.5, use_mmr=True, lambda_param=0.7):
        if not self.documents:
            raise ValueError("Call `fit(documents)` before searching.")

        # Encode query
        query_dense = self.dense_model.encode(
            [query], normalize_embeddings=True)
        query_sparse = self.sparse_vectorizer.transform([query])

        # Dense and Sparse similarities
        dense_scores = util.cos_sim(query_dense, self.dense_embeddings)[
            0].cpu().numpy()
        sparse_scores = (query_sparse @ self.sparse_matrix.T).toarray()[0]

        # Hybrid score
        hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
        # more candidates for reranking
        top_indices = hybrid_scores.argsort()[::-1][:top_n * 2]

        if use_mmr:
            selected = self.mmr_rerank(
                top_indices, query_dense, lambda_param, top_n)
            return [{"document": self.documents[i], "score": float(hybrid_scores[i])} for i in selected]

        sorted_indices = hybrid_scores.argsort()[::-1][:top_n]
        return [{"document": self.documents[i], "score": float(hybrid_scores[i])} for i in sorted_indices]

    def mmr_rerank(self, candidate_indices, query_vec, lambda_param, top_n):
        selected = []
        candidates = list(candidate_indices)

        embeddings = self.dense_embeddings

        while len(selected) < top_n and candidates:
            mmr_scores = []
            for idx in candidates:
                sim_to_query = util.cos_sim(
                    query_vec, embeddings[idx])[0].item()
                sim_to_selected = max([util.cos_sim(embeddings[idx], embeddings[j])[0].item()
                                       for j in selected], default=0)
                mmr_score = lambda_param * sim_to_query - \
                    (1 - lambda_param) * sim_to_selected
                mmr_scores.append((mmr_score, idx))

            # Select document with max MMR score
            _, selected_idx = max(mmr_scores, key=lambda x: x[0])
            selected.append(selected_idx)
            candidates.remove(selected_idx)

        return selected
