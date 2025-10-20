# jet/libs/bertopic/rag_bertopic.py

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from jet.adapters.bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding

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
    def __init__(self, model_name: str = "embeddinggemma", verbose: bool = False):
        self.verbose = verbose
        self.model = None
        self.topic_indexes: Dict[int, TopicIndex] = {}
        self.embedder = LlamacppEmbedding(model=model_name)

    def _log(self, msg: str, level: int = logging.INFO):
        if self.verbose:
            logger.log(level, f"[TopicRAG] {msg}")

    def _preprocess_and_filter(self, docs: List[str]) -> List[str]:
        deduplicated_docs = list(dict.fromkeys(docs))
        valid_docs = [d for d in deduplicated_docs if isinstance(d, str) and d.strip()]
        return valid_docs

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
            self._log(
                f"_safe_umap: Extremely tiny corpus detected (n_docs={n_docs}); "
                f"Setting n_neighbors={n_neighbors}, n_components={n_components}, init={init}",
                logging.DEBUG,
            )
        elif n_docs <= 10:
            # Small corpus
            n_neighbors = max(2, n_docs - 1)
            n_components = min(2, n_docs - 1)
            init = "random"
            self._log(
                f"_safe_umap: Small corpus detected (n_docs={n_docs}); "
                f"Setting n_neighbors={n_neighbors}, n_components={n_components}, init={init}",
                logging.DEBUG,
            )
        elif n_docs <= 30:
            # Medium-small corpus
            n_neighbors = min(10, n_docs - 1)
            n_components = min(5, n_docs - 1)
            init = "random"
            self._log(
                f"_safe_umap: Medium-small corpus detected (n_docs={n_docs}); "
                f"Setting n_neighbors={n_neighbors}, n_components={n_components}, init={init}",
                logging.DEBUG,
            )
        else:
            # Normal or large corpus
            n_neighbors = 15
            n_components = 5
            init = "spectral"
            self._log(
                f"_safe_umap: Normal/large corpus detected (n_docs={n_docs}); "
                f"Setting n_neighbors={n_neighbors}, n_components={n_components}, init={init}",
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

        docs = self._preprocess_and_filter(docs)
        self._log(f"Starting topic fitting on {len(docs)} docs")

        # Embeddings
        embeddings = self.embedder(docs, show_progress=True)

        umap_model = self._safe_umap(docs)
        if umap_model is None:
            self._log("Falling back to BERTopic without UMAP (document-level clustering only).", logging.WARNING)
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

        try:
            topics, _ = self.model.fit_transform(docs, embeddings)

        except ValueError:
            # For very small datasets, use fallback clustering
            self._log(
                f"Too few samples ({embeddings.shape[0]}). Using safe single-cluster fallback.",
                logging.WARNING,
            )
            topics = [0] * len(docs)
            self.model = BERTopic(
                embedding_model=None,
                calculate_probabilities=False,
                vectorizer_model=CountVectorizer(stop_words="english"),
                umap_model=None,
            )
            self._build_indexes(docs, embeddings, topics)
            return

        except TypeError as e:
            if "k >= N" in str(e):
                self._log(f"ARPACK eigsh failure detected: {e}", logging.WARNING)
                # Force random init UMAP fallback
                safe_umap = UMAP(
                    n_neighbors=max(2, len(docs) - 1),
                    n_components=min(2, len(docs) - 1),
                    metric="cosine",
                    init="random",
                    random_state=42,
                    low_memory=True,
                )
                self.model = BERTopic(umap_model=safe_umap)
                topics, _ = self.model.fit_transform(docs, embeddings)
            else:
                raise

        # 🔍 Add diagnostic logging here
        topic_info = self.model.get_topic_info()
        # Limit to key columns and top N topics
        max_topics_to_log = 5
        max_docs_per_topic = 3
        max_doc_length = 100  # Characters
        if not topic_info.empty:
            # Select key columns and limit rows
            limited_topic_info = topic_info[["Topic", "Count", "Name"]].head(max_topics_to_log)
            self._log(
                f"Topic summary (top {max_topics_to_log}):\n{limited_topic_info.to_string(index=False)}",
                logging.DEBUG,
            )
        else:
            self._log("No topic information available.", logging.DEBUG)

        # Log limited topic details
        topic_map = {}
        for doc, topic in zip(docs, topics):
            topic_map.setdefault(topic, []).append(doc)
        for tid, td in list(topic_map.items())[:max_topics_to_log]:
            # Truncate documents and limit number of documents logged
            truncated_docs = [
                (doc[:max_doc_length] + "..." if len(doc) > max_doc_length else doc)
                for doc in td[:max_docs_per_topic]
            ]
            self._log(
                f"Topic {tid}: {len(td)} docs\n - " + "\n - ".join(truncated_docs),
                logging.DEBUG,
            )
        if len(topic_map) > max_topics_to_log:
            self._log(
                f"...and {len(topic_map) - max_topics_to_log} more topics not shown.",
                logging.DEBUG,
            )

        self._build_indexes(docs, embeddings, topics)

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

        qvec = self.embedder(query, show_progress=True)

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
