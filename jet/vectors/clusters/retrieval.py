import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
import faiss
from typing import List, Tuple, Optional, Union, TypedDict
from dataclasses import dataclass
import pickle
import os
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType


class RetrievalConfigDict(TypedDict, total=False):
    """Typed dictionary for vector retrieval configuration."""
    min_cluster_size: int
    k_clusters: int
    top_k: Optional[int]
    cluster_threshold: int
    model_name: EmbedModelType
    cache_file: Optional[str]
    threshold: Optional[float]


@dataclass
class RetrievalConfig:
    """Configuration for vector retrieval parameters."""
    min_cluster_size: int = 2
    k_clusters: int = 2
    top_k: Optional[int] = None
    cluster_threshold: int = 20
    model_name: EmbedModelType = 'mxbai-embed-large'
    cache_file: Optional[str] = None
    threshold: Optional[float] = None


class VectorRetriever:
    """Class for retrieving relevant text chunks using vector search and clustering."""

    def __init__(self, config: Union[RetrievalConfig, RetrievalConfigDict]):
        if isinstance(config, dict):
            self.config = RetrievalConfig(**config)
        else:
            self.config = config
        self.model = SentenceTransformerRegistry.load_model(
            self.config.model_name)
        self.embeddings: Optional[np.ndarray] = None
        self.corpus: Optional[List[str]] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centroids: Optional[np.ndarray] = None

    def load_or_compute_embeddings(self, corpus: List[str], cache_file: Optional[str] = None) -> np.ndarray:
        """Load cached embeddings or compute new ones."""
        active_cache_file = cache_file or self.config.cache_file
        if not corpus:
            raise ValueError("Corpus cannot be empty")
        self.corpus = corpus
        if active_cache_file and os.path.exists(active_cache_file):
            with open(active_cache_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            if self.embeddings.shape[0] != len(corpus):
                raise ValueError("Cached embeddings do not match corpus size")
        else:
            self.embeddings = self.model.encode(corpus, convert_to_numpy=True)
            faiss.normalize_L2(self.embeddings)
            if active_cache_file:
                with open(active_cache_file, 'wb') as f:
                    pickle.dump(self.embeddings, f)
        return self.embeddings

    def cluster_embeddings(self, min_cluster_size: Optional[int] = None, cluster_threshold: Optional[int] = None) -> None:
        """Cluster embeddings using HDBSCAN."""
        active_min_cluster_size = min_cluster_size or self.config.min_cluster_size
        active_cluster_threshold = cluster_threshold or self.config.cluster_threshold
        if len(self.corpus) < active_cluster_threshold:
            self.cluster_labels = np.array([-1] * len(self.corpus))
            self.cluster_centroids = np.zeros((0, self.embeddings.shape[1]))
            return
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=active_min_cluster_size,
            metric='euclidean',
            cluster_selection_method='leaf',
            min_samples=1
        )
        self.cluster_labels = clusterer.fit_predict(self.embeddings)
        n_clusters = len(set(self.cluster_labels)) - \
            (1 if -1 in self.cluster_labels else 0)
        self.cluster_centroids = []
        for cluster_id in range(n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                centroid = np.mean(self.embeddings[cluster_mask], axis=0)
                self.cluster_centroids.append(centroid)
        self.cluster_centroids = np.zeros(
            (0, self.embeddings.shape[1])) if not self.cluster_centroids else np.array(self.cluster_centroids)

    def build_index(self) -> None:
        """Build Faiss index for vector search."""
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(self.embeddings)

    def retrieve_chunks(self, query: str, k_clusters: Optional[int] = None, top_k: Optional[int] = None, cluster_threshold: Optional[int] = None, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """Retrieve top-k relevant chunks for the query, optionally filtered by similarity threshold."""
        active_k_clusters = k_clusters or self.config.k_clusters
        active_top_k = top_k if top_k is not None else self.config.top_k
        active_cluster_threshold = cluster_threshold or self.config.cluster_threshold
        active_threshold = threshold or self.config.threshold
        if not query:
            raise ValueError("Query cannot be empty")
        if self.embeddings is None or self.index is None:
            raise ValueError("Retriever not initialized with corpus")
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        top_chunks: List[Tuple[str, float]] = []
        if len(self.corpus) >= active_cluster_threshold and self.cluster_centroids.shape[0] > 0:
            cluster_centroid_index = faiss.IndexFlatIP(
                self.embeddings.shape[1])
            faiss.normalize_L2(self.cluster_centroids)
            cluster_centroid_index.add(self.cluster_centroids)
            _, top_cluster_indices = cluster_centroid_index.search(
                query_embedding, active_k_clusters)
            for cluster_id in top_cluster_indices[0]:
                cluster_mask = self.cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                if len(cluster_indices) == 0:
                    continue
                cluster_embeddings = self.embeddings[cluster_mask]
                temp_index = faiss.IndexFlatIP(self.embeddings.shape[1])
                temp_index.add(cluster_embeddings)
                search_k = len(cluster_indices) if active_top_k is None else min(
                    active_top_k, len(cluster_indices))
                distances, indices = temp_index.search(
                    query_embedding, search_k)
                global_indices = cluster_indices[indices[0]]
                top_chunks.extend([(self.corpus[idx], float(distances[0][i]))
                                  for i, idx in enumerate(global_indices)])
        if not top_chunks:
            search_k = len(
                self.corpus) if active_top_k is None else active_top_k
            distances, indices = self.index.search(query_embedding, search_k)
            top_chunks = [(self.corpus[idx], float(distances[0][i]))
                          for i, idx in enumerate(indices[0])]
        if active_threshold is not None:
            top_chunks = [(chunk, score) for chunk,
                          score in top_chunks if score >= active_threshold]
        sorted_chunks = sorted(top_chunks, key=lambda x: x[1], reverse=True)
        return sorted_chunks[:active_top_k] if active_top_k is not None else sorted_chunks
