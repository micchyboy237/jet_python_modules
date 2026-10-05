import logging
from typing import Any, Dict, List, Optional

import numpy as np
from jet.adapters.bertopic import BERTopic
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

logger = logging.getLogger(__name__)
try:
    import faiss

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class TopicIndex:
    def __init__(self, topic_id: int, embeddings: np.ndarray, texts: List[str]):
        self.topic_id = topic_id
        self.embeddings = embeddings
        self.texts = texts
        self.doc_ids = list(range(len(texts)))
        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
        else:
            self.index = None


class TopicRAG:
    def __init__(self, model_name: str = EMBED_MODEL, verbose: bool = False):
        self.verbose = verbose
        self.model = None
        self.topic_indexes: Dict[int, TopicIndex] = {}
        self.embedder = LlamacppEmbedding(model=model_name)

        # Hybrid Search Components
        self.tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.all_docs_for_tfidf: List[str] = []

    def _create_vectorizer(self, n_docs: int) -> TfidfVectorizer:
        """Create a TfidfVectorizer with parameters optimized for dataset size."""
        if n_docs <= 10:
            return TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True,
                min_df=1,
                max_df=0.95,
                norm="l2",
            )
        elif n_docs <= 100:
            return TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
                min_df=2,
                max_df=0.85,
                norm="l2",
            )
        else:
            return TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
                min_df=3,
                max_df=0.8,
                norm="l2",
            )

    def _log(self, msg: str, level: int = logging.INFO):
        if self.verbose:
            logger.log(level, f"[TopicRAG] {msg}")

    def _preprocess_and_filter(self, docs: List[str]) -> List[str]:
        deduplicated_docs = list(dict.fromkeys(docs))
        valid_docs = [d for d in deduplicated_docs if isinstance(d, str) and d.strip()]
        return valid_docs

    def _safe_umap(self, docs: List[str]) -> UMAP:
        """Create a UMAP instance that safely handles very small and normal datasets."""
        n_docs = len(docs)
        if n_docs <= 3:
            n_neighbors, n_components, init = 2, 1, "random"
        elif n_docs <= 10:
            n_neighbors, n_components, init = (
                max(2, n_docs - 1),
                min(2, n_docs - 1),
                "random",
            )
        elif n_docs <= 30:
            n_neighbors, n_components, init = (
                min(10, n_docs - 1),
                min(5, n_docs - 1),
                "random",
            )
        else:
            n_neighbors, n_components, init = 15, 5, "spectral"

        self._log(
            f"_safe_umap: Setting n_neighbors={n_neighbors}, n_components={n_components}, init={init}",
            logging.DEBUG,
        )
        return UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric="cosine",
            random_state=42,
            low_memory=True,
            init=init,
        )

    def fit_topics(
        self, docs: List[str], nr_topics: Any = "auto", min_topic_size: int = 2
    ):
        if not docs:
            raise ValueError("No documents provided for topic fitting.")

        docs = self._preprocess_and_filter(docs)
        n_docs = len(docs)
        self.all_docs_for_tfidf = docs  # Store for hybrid search

        self._log(f"Starting topic fitting on {n_docs} docs")
        vectorizer_model = self._create_vectorizer(n_docs)

        # Fit TF-IDF for hybrid search
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(docs)

        embeddings = self.embedder(docs, show_progress=True)
        umap_model = self._safe_umap(docs)

        self.model = BERTopic(
            embedding_model=None,
            calculate_probabilities=True,
            nr_topics=nr_topics,
            min_topic_size=min_topic_size,
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
        )

        try:
            topics, _ = self.model.fit_transform(docs, embeddings)
        except (ValueError, TypeError) as e:
            self._log(f"Fallback triggered due to: {e}", logging.WARNING)
            topics = [0] * len(docs)

        self._build_indexes(docs, embeddings, topics)

    def _build_indexes(
        self, docs: List[str], embeddings: np.ndarray, topics: List[int]
    ):
        topic_docs: Dict[int, List[str]] = {}
        topic_vecs: Dict[int, List[np.ndarray]] = {}

        for doc, topic, emb in zip(docs, topics, embeddings):
            topic_docs.setdefault(topic, []).append(doc)
            topic_vecs.setdefault(topic, []).append(emb)

        for tid, vecs in topic_vecs.items():
            self.topic_indexes[tid] = TopicIndex(
                topic_id=tid, embeddings=np.vstack(vecs), texts=topic_docs[tid]
            )
        self._log(f"Built {len(self.topic_indexes)} topic partitions")

    def retrieve_for_query(
        self,
        query: str,
        top_topics: int = 3,
        top_k: int = 5,
        unique_by: Optional[str] = None,
        alpha: float = 0.7,  # Weight for vector search (1-alpha for keyword)
    ) -> List[Dict[str, Any]]:
        if not self.model or not self.topic_indexes:
            raise RuntimeError("TopicRAG not yet fitted.")

        # 1. Vector Search Preparation
        qvec = self.embedder(query, show_progress=False)
        if qvec.ndim == 1:
            qvec = qvec.reshape(1, -1)

        topic_centroids = {
            tid: np.mean(ti.embeddings, axis=0)
            for tid, ti in self.topic_indexes.items()
        }
        centroid_norms = {
            tid: v / np.linalg.norm(v) for tid, v in topic_centroids.items()
        }
        q_norm = qvec / np.linalg.norm(qvec)

        similarities = {
            tid: float(np.dot(v, q_norm.T).squeeze())
            for tid, v in centroid_norms.items()
        }
        sorted_topics = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[
            :top_topics
        ]

        # 2. Keyword Search Preparation
        query_tfidf = self.tfidf_vectorizer.transform([query])

        results = []
        seen = set()

        for topic, _ in sorted_topics:
            if topic not in self.topic_indexes:
                continue

            ti = self.topic_indexes[topic]

            # Get indices of docs in this topic relative to the global list
            # Note: This assumes we can map back. For simplicity, we'll search locally within the topic

            # A. Local Vector Scores
            local_scores_vec = []
            if _HAS_FAISS and ti.index is not None:
                scores, idxs = ti.index.search(qvec, min(top_k * 2, len(ti.texts)))
                local_scores_vec = [
                    (int(i), float(s)) for i, s in zip(idxs[0], scores[0])
                ]

            # B. Local Keyword Scores
            local_texts = ti.texts
            local_tfidf_matrix = self.tfidf_matrix[
                [self.all_docs_for_tfidf.index(t) for t in local_texts]
            ]
            keyword_scores = (local_tfidf_matrix * query_tfidf.T).toarray().flatten()
            local_scores_kw = [(i, float(s)) for i, s in enumerate(keyword_scores)]

            # C. Hybrid Fusion
            # Normalize scores to 0-1 range for fair weighting
            max_vec = max([s for _, s in local_scores_vec], default=1)
            max_kw = max([s for _, s in local_scores_kw], default=1)

            score_map = {}
            for i, s in local_scores_vec:
                score_map[i] = score_map.get(i, 0) + alpha * (s / max_vec)
            for i, s in local_scores_kw:
                score_map[i] = score_map.get(i, 0) + (1 - alpha) * (s / max_kw)

            # Sort by combined score
            sorted_local = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[
                :top_k
            ]

            for idx, score in sorted_local:
                text = ti.texts[idx]
                if unique_by == "text" and text in seen:
                    continue
                seen.add(text)
                results.append({"topic": topic, "text": text, "score": float(score)})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results
