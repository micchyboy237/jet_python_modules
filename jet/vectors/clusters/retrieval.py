import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
import faiss
from typing import Dict, List, Tuple, Optional, Union, TypedDict
from dataclasses import dataclass
import pickle
import os
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from jet.models.tokenizer.base import get_tokenizer_fn


class ChunkSearchResult(TypedDict):
    """Typed dictionary for search result."""
    id: str
    rank: int
    score: float
    num_tokens: int
    text: str
    cluster_label: int
    metadata: Dict


class ClusterInfo(TypedDict):
    """Typed dictionary for cluster information."""
    id: str
    cluster_label: int
    texts: List[str]
    metadata: Dict


class CentroidInfo(TypedDict):
    """Typed dictionary for centroid information."""
    id: str
    cluster_label: int
    count: int
    centroid: List[float]  # List for JSON serialization
    texts: List[str]


class CentroidSearchResult(TypedDict):
    """Typed dictionary for centroid search results."""
    id: str
    rank: int
    cluster_label: int
    score: float
    text: str
    diverse_texts: List[str]
    metadata: Dict


class RetrievalConfigDict(TypedDict, total=False):
    """Typed dictionary for vector retrieval configuration."""
    embed_model: EmbedModelType
    min_cluster_size: int
    k_clusters: int
    top_k: Optional[int]
    cluster_threshold: int
    cache_file: Optional[str]
    threshold: Optional[float]


@dataclass
class RetrievalConfig:
    """Configuration for vector retrieval parameters."""
    embed_model: EmbedModelType = 'mxbai-embed-large'
    min_cluster_size: int = 2
    k_clusters: int = 2
    top_k: Optional[int] = None
    cluster_threshold: int = 20
    cache_file: Optional[str] = None
    threshold: Optional[float] = None


class VectorRetriever:
    """Class for retrieving relevant text chunks using vector search and clustering."""

    def __init__(self, config: Union[RetrievalConfig, RetrievalConfigDict]):
        if isinstance(config, dict):
            self.config = RetrievalConfig(**config)
        else:
            self.config = config
        self.model_name: EmbedModelType = self.config.embed_model
        self.model = SentenceTransformerRegistry.load_model(
            self.config.embed_model)
        self.tokenizer = get_tokenizer_fn(self.config.embed_model)
        self.embeddings: Optional[np.ndarray] = None
        self.corpus: Optional[List[str]] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centroids: Optional[np.ndarray] = None

    def load_or_compute_embeddings(self, corpus: List[str], cache_file: Optional[str] = None, ids: Optional[List[str]] = None, metadatas: Optional[List[Dict]] = None) -> np.ndarray:
        """Load cached embeddings or compute new ones."""
        active_cache_file = cache_file or self.config.cache_file
        if not corpus:
            raise ValueError("Corpus cannot be empty")
        self.corpus = corpus
        self.ids = ids if ids is not None else [
            str(uuid.uuid4()) for _ in corpus]
        self.metadatas = metadatas if metadatas is not None else [
            {} for _ in corpus]
        if len(self.ids) != len(corpus):
            raise ValueError("Number of IDs must match corpus size")
        if len(self.metadatas) != len(corpus):
            raise ValueError("Number of metadatas must match corpus size")
        if active_cache_file and os.path.exists(active_cache_file):
            with open(active_cache_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            if self.embeddings.shape[0] != len(corpus):
                raise ValueError("Cached embeddings do not match corpus size")
        else:
            self.embeddings = self.model.encode(corpus, convert_to_numpy=True)
            faiss.normalize_L2(self.embeddings)
            if active_cache_file:
                with open(active_cache_file, 'wb') as ATS:
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

    def get_clusters(self) -> List[ClusterInfo]:
        """Return a list of clusters with their labels and associated text chunks."""
        if self.corpus is None or self.cluster_labels is None:
            raise ValueError(
                "Retriever not initialized with corpus or clusters")

        # Initialize dictionary to group texts by cluster label
        cluster_dict = {}
        for idx, cluster_label in enumerate(self.cluster_labels):
            cluster_label = int(cluster_label)  # Ensure integer type
            if cluster_label not in cluster_dict:
                cluster_dict[cluster_label] = {
                    "texts": [], "id": str(uuid.uuid4()), "metadata": {}}
            cluster_dict[cluster_label]["texts"].append(self.corpus[idx])
            # Aggregate metadata if needed; for now, use empty dict or first metadata
            if not cluster_dict[cluster_label]["metadata"]:
                cluster_dict[cluster_label]["metadata"] = self.metadatas[idx]

        # Convert to list of ClusterInfo
        clusters: List[ClusterInfo] = [
            {"id": info["id"], "cluster_label": cluster_label,
                "texts": info["texts"], "metadata": info["metadata"]}
            for cluster_label, info in sorted(cluster_dict.items())
        ]
        return clusters

    def get_centroids(self) -> List[CentroidInfo]:
        """Return a list of centroids with their labels, associated texts, and count of texts."""
        if self.cluster_centroids is None or self.cluster_labels is None or self.corpus is None:
            raise ValueError(
                "Retriever not initialized with clusters or corpus")

        centroids: List[CentroidInfo] = []
        for cluster_label in range(len(self.cluster_centroids)):
            # Get texts for the current cluster
            cluster_mask = self.cluster_labels == cluster_label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_texts = [self.corpus[idx] for idx in cluster_indices]

            centroids.append({
                "id": str(uuid.uuid4()),
                "cluster_label": int(cluster_label),
                "count": len(cluster_texts),
                "texts": cluster_texts,
                "centroid": self.cluster_centroids[cluster_label].tolist(),
            })
        return centroids

    def search_centroids(self, query: str, diversity_threshold: float = 0.8) -> List[CentroidSearchResult]:
        """Return similarity scores for each centroid given a query, sorted by similarity in descending order, with most similar and diverse texts."""
        if not query:
            raise ValueError("Query cannot be empty")
        if self.cluster_centroids is None or self.cluster_centroids.shape[0] == 0 or self.corpus is None or self.cluster_labels is None:
            return []

        # Encode query and compute cosine similarities
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        faiss.normalize_L2(self.cluster_centroids)
        similarities = np.dot(self.cluster_centroids,
                              query_embedding.T).flatten()

        results: List[CentroidSearchResult] = []
        for cluster_label in range(len(self.cluster_centroids)):
            # Get texts and embeddings for the current cluster
            cluster_mask = self.cluster_labels == cluster_label
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_texts = [self.corpus[idx] for idx in cluster_indices]
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_metadatas = [self.metadatas[idx]
                                 for idx in cluster_indices]

            # Find most similar text to the centroid
            centroid = self.cluster_centroids[cluster_label].reshape(1, -1)
            text_similarities = np.dot(
                cluster_embeddings, centroid.T).flatten()
            most_similar_idx = np.argmax(text_similarities)
            most_similar_text = cluster_texts[most_similar_idx]
            most_similar_metadata = cluster_metadatas[most_similar_idx]

            # Find diverse texts
            diverse_texts = []
            if len(cluster_texts) > 0:
                # Start with most similar text
                diverse_texts.append(cluster_texts[most_similar_idx])
                remaining_indices = [i for i in range(
                    len(cluster_texts)) if i != most_similar_idx]
                for i in remaining_indices:
                    is_diverse = True
                    curr_embedding = cluster_embeddings[i].reshape(1, -1)
                    for selected_text in diverse_texts:
                        selected_idx = cluster_texts.index(selected_text)
                        selected_embedding = cluster_embeddings[selected_idx].reshape(
                            1, -1)
                        similarity = np.dot(
                            curr_embedding, selected_embedding.T).flatten()[0]
                        if similarity > diversity_threshold:
                            is_diverse = False
                            break
                    if is_diverse:
                        diverse_texts.append(cluster_texts[i])
                    if len(diverse_texts) >= 3:  # Limit to 3 diverse texts for efficiency
                        break

            results.append({
                "id": str(uuid.uuid4()),
                "rank": len(results) + 1,
                "cluster_label": int(cluster_label),
                "score": float(similarities[cluster_label]),
                "text": most_similar_text,
                "diverse_texts": diverse_texts,
                "metadata": most_similar_metadata
            })

        sorted_results = sorted(
            results, key=lambda x: x["score"], reverse=True)
        # Reassign ranks starting from 1
        for i, result in enumerate(sorted_results):
            result["rank"] = i + 1
        return sorted_results

    def search_chunks(self, query: str, k_clusters: Optional[int] = None, top_k: Optional[int] = None, cluster_threshold: Optional[int] = None, threshold: Optional[float] = None) -> List[ChunkSearchResult]:
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
        top_chunks: List[ChunkSearchResult] = []
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
                top_chunks.extend([{
                    "id": self.ids[idx],
                    "rank": i + 1,
                    "score": float(distances[0][i]),
                    "num_tokens": len(self.tokenizer(self.corpus[idx])),
                    "text": self.corpus[idx],
                    "cluster_label": int(self.cluster_labels[idx]),
                    "metadata": self.metadatas[idx]
                } for i, idx in enumerate(global_indices)])
        if not top_chunks:
            search_k = len(
                self.corpus) if active_top_k is None else active_top_k
            distances, indices = self.index.search(query_embedding, search_k)
            top_chunks = [{
                "id": self.ids[idx],
                "rank": i + 1,
                "score": float(distances[0][i]),
                "num_tokens": len(self.tokenizer(self.corpus[idx])),
                "text": self.corpus[idx],
                "cluster_label": int(self.cluster_labels[idx]) if self.cluster_labels is not None else -1,
                "metadata": self.metadatas[idx]
            } for i, idx in enumerate(indices[0])]
        if active_threshold is not None:
            top_chunks = [
                chunk for chunk in top_chunks if chunk["score"] >= active_threshold]
        sorted_chunks = sorted(
            top_chunks, key=lambda x: x["score"], reverse=True)
        # Reassign ranks after sorting
        for i, chunk in enumerate(sorted_chunks):
            chunk["rank"] = i + 1
        return sorted_chunks[:active_top_k] if active_top_k is not None else sorted_chunks
