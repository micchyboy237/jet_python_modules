# jet/libs/bertopic/rag_bertopic.py

import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
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
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", verbose: bool = False):
        self.verbose = verbose
        self.model = None
        self.topic_indexes: Dict[int, TopicIndex] = {}
        self.embedder = SentenceTransformer(model_name, device="mps")

    def _log(self, msg: str, level: int = logging.INFO):
        if self.verbose:
            logger.log(level, f"[TopicRAG] {msg}")

    def _deduplicate(self, docs: List[str]) -> List[str]:
        return list(dict.fromkeys(docs))

    def _safe_umap(self, docs: List[str]) -> UMAP:
        """
        Create a UMAP instance that safely handles very small and normal datasets.

        - Always returns a valid UMAP (never None) to prevent AttributeError.
        - Automatically adjusts parameters based on corpus size.
        - For small datasets, uses init='random' and minimal n_components
        to avoid ARPACK eigsh (k >= N) failures.
        """

        n_docs = len(docs)

        # Adjust parameters dynamically
        if n_docs <= 3:
            # Extremely tiny corpus
            n_neighbors = 2
            n_components = 1
            init = "random"
        elif n_docs <= 10:
            # Small corpus
            n_neighbors = max(2, n_docs - 1)
            n_components = min(2, n_docs - 1)
            init = "random"
        elif n_docs <= 30:
            # Medium-small corpus
            n_neighbors = min(10, n_docs - 1)
            n_components = min(5, n_docs - 1)
            init = "random"
        else:
            # Normal or large corpus
            n_neighbors = 15
            n_components = 5
            init = "spectral"

        self._log(
            f"UMAP configuration: n_docs={n_docs}, "
            f"n_neighbors={n_neighbors}, n_components={n_components}, init={init}",
            logging.DEBUG,
        )

        # Always safe configuration
        return UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric="cosine",
            random_state=42,
            low_memory=True,
            init=init,
        )

    def fit_topics(self, docs: List[str], nr_topics: Any = "auto", min_topic_size: int = 2):
        if not docs:
            raise ValueError("No documents provided for topic fitting.")

        docs = self._deduplicate(docs)
        self._log(f"Starting topic fitting on {len(docs)} docs")

        # Remove invalid or empty documents
        valid_docs = [d for d in docs if isinstance(d, str) and d.strip()]
        if not valid_docs:
            raise ValueError("No valid text documents to process after preprocessing or filtering.")

        # Compute embeddings
        embeddings = np.array(self.embedder.encode(valid_docs, show_progress_bar=False)).astype("float32")

        # --- 🔒 Safety checks before clustering ---
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("No embeddings were generated. Please check input documents.")
        if embeddings.shape[0] != len(valid_docs):
            raise ValueError(
                f"Mismatch between documents ({len(valid_docs)}) and embeddings ({len(embeddings)})."
            )

        umap_model = self._safe_umap(valid_docs)
        if umap_model is None:
            self._log("Falling back to BERTopic without UMAP.", logging.WARNING)
            self.model = BERTopic(
                embedding_model=None,
                calculate_probabilities=True,
                nr_topics=nr_topics,
                min_topic_size=min_topic_size,
                vectorizer_model=CountVectorizer(stop_words="english"),
                umap_model=None,
            )
        else:
            self.model = BERTopic(
                embedding_model=None,
                calculate_probabilities=True,
                nr_topics=nr_topics,
                min_topic_size=min_topic_size,
                vectorizer_model=CountVectorizer(stop_words="english"),
                umap_model=umap_model,
            )

        # For very small datasets, use fallback clustering
        if embeddings.shape[0] < 5:
            self._log(
                f"Too few samples ({embeddings.shape[0]}). Using safe single-cluster fallback.",
                logging.WARNING,
            )
            topics = [0] * len(valid_docs)
            self.model = BERTopic(
                embedding_model=None,
                calculate_probabilities=False,
                vectorizer_model=CountVectorizer(stop_words="english"),
                umap_model=None,
            )
            self._build_indexes(valid_docs, embeddings, topics)
            return

        try:
            topics, _ = self.model.fit_transform(valid_docs, embeddings)

        except ValueError as e:
            if "Found array with 0 sample" in str(e):
                self._log(f"Empty embedding array detected: {e}", logging.WARNING)
                if embeddings.shape[0] == 0:
                    embeddings = np.zeros((len(valid_docs), 384))
                safe_umap = UMAP(
                    n_neighbors=2,
                    n_components=min(2, len(valid_docs) - 1),
                    metric="cosine",
                    init="random",
                    random_state=42,
                    low_memory=True,
                )
                self.model = BERTopic(umap_model=safe_umap)
                topics, _ = self.model.fit_transform(valid_docs, embeddings)
            else:
                raise

        except TypeError as e:
            if "k >= N" in str(e):
                self._log(f"ARPACK eigsh failure detected: {e}", logging.WARNING)
                safe_umap = UMAP(
                    n_neighbors=max(2, len(valid_docs) - 1),
                    n_components=min(2, len(valid_docs) - 1),
                    metric="cosine",
                    init="random",
                    random_state=42,
                    low_memory=True,
                )
                self.model = BERTopic(umap_model=safe_umap)
                topics, _ = self.model.fit_transform(valid_docs, embeddings)
            else:
                raise

        # --- Diagnostics ---
        topic_info = self.model.get_topic_info()
        self._log(f"Topic summary:\n{topic_info.to_string(index=False)}", logging.DEBUG)

        topic_map = {}
        for doc, topic in zip(valid_docs, topics):
            topic_map.setdefault(topic, []).append(doc)
        for tid, td in topic_map.items():
            self._log(f"Topic {tid}: {len(td)} docs\n - " + "\n - ".join(td), logging.DEBUG)

        self._build_indexes(valid_docs, embeddings, topics)

    def _build_indexes(self, docs: List[str], embeddings: np.ndarray, topics: List[int]):
        topic_docs: Dict[int, List[str]] = {}
        topic_vecs: Dict[int, List[np.ndarray]] = {}

        for doc, topic, emb in zip(docs, topics, embeddings):
            # if topic == -1:
            #     continue
            topic_docs.setdefault(topic, []).append(doc)
            topic_vecs.setdefault(topic, []).append(emb)

        for tid, vecs in topic_vecs.items():
            self.topic_indexes[tid] = TopicIndex(
                topic_id=tid,
                embeddings=np.vstack(vecs),
                texts=topic_docs[tid]
            )

        self._log(f"Built {len(self.topic_indexes)} topic partitions")

    def retrieve_for_query(self, query: str, top_topics: int = 1, top_k: int = 3, unique_by: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.model or not self.topic_indexes:
            raise RuntimeError("TopicRAG not yet fitted.")

        qvec = np.array(self.embedder.encode([query], show_progress_bar=False)).astype("float32")

        # --- replaced old find_topics() call ---
        # topic_scores, _ = self.model.find_topics(query, top_n=top_topics)
        topic_centroids = {
            tid: np.mean(ti.embeddings, axis=0) for tid, ti in self.topic_indexes.items()
        }
        centroid_norms = {
            tid: v / np.linalg.norm(v) for tid, v in topic_centroids.items()
        }
        q_norm = qvec / np.linalg.norm(qvec)
        similarities = {
            tid: float(np.dot(v, q_norm.T).squeeze()) for tid, v in centroid_norms.items()
        }
        sorted_topics = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_topics]
        # --- end replacement ---

        results = []
        seen = set()
        for topic, _ in sorted_topics:
            if topic not in self.topic_indexes:
                continue
            hits = self._search_topic(self.topic_indexes[topic], qvec, top_k)
            for idx, score in hits:
                text = self.topic_indexes[topic].texts[idx]
                if unique_by == "text" and text in seen:
                    continue
                seen.add(text)
                results.append({
                    "topic": topic,
                    "text": text,
                    "score": float(score),
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    def _search_topic(self, topic_index: TopicIndex, qvec: np.ndarray, top_k: int):
        if _HAS_FAISS and topic_index.index is not None:
            scores, idxs = topic_index.index.search(qvec, top_k)
            return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0])]

        emb_norm = topic_index.embeddings / np.linalg.norm(topic_index.embeddings, axis=1, keepdims=True)
        q_norm = qvec / np.linalg.norm(qvec)
        scores = np.dot(emb_norm, q_norm.T).squeeze()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]
