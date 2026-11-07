"""
Clustering for mixed-length texts (words, sentences, and documents).
Uses configured Llama.cpp embedding models for semantic understanding.
Handles varying lengths by normalizing embeddings and chunking long texts.
Dynamically configures dimensions and safety thresholds based on model metadata.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
from jet.adapters.llama_cpp.config import (
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from sklearn.cluster import HDBSCAN, SpectralClustering
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


class MixedLengthClusterer:
    """Clusters texts of varying lengths using configured Llama.cpp embeddings."""

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        min_cluster_size: int = 2,
        max_tokens_safe: Optional[int] = None,
        query_prefix: Optional[str] = None,
        doc_prefix: Optional[str] = None,
    ):
        """
        Initialize the mixed-length clusterer.

        Args:
            model_name: Model identifier (default: from config EMBED_MODEL)
            min_cluster_size: Minimum cluster size for HDBSCAN
            max_tokens_safe: Word count threshold to trigger chunking.
                             If None, calculated dynamically from model context window.
            query_prefix: Prefix for query texts. Defaults to config EMBED_QUERY_PREFIX.
            doc_prefix: Prefix for document texts. Defaults to config EMBED_DOC_PREFIX.
        """
        self.model_name = model_name
        self.min_cluster_size = min_cluster_size

        # Resolve prefixes from config if not explicitly provided
        self.query_prefix = (
            query_prefix if query_prefix is not None else EMBED_QUERY_PREFIX
        )
        self.doc_prefix = doc_prefix if doc_prefix is not None else EMBED_DOC_PREFIX

        # Dynamically determine embedding dimension and max tokens from model metadata
        self.embedding_dim = None
        self.max_tokens_safe = max_tokens_safe
        self._initialize_model_params()

        self.embeddings = None
        self.labels = None
        self.text_lengths = None

    def _initialize_model_params(self):
        """Fetch model metadata to set embedding dim and safe token limits."""
        try:
            print(f"Fetching metadata for model: {self.model_name}...")
            model_info = get_model_ctx_embd_size(self.model_name)

            # Set embedding dimension
            self.embedding_dim = model_info["embd_dims"]
            if not self.embedding_dim or self.embedding_dim == 0:
                # Fallback if metadata doesn't provide it, though rare for valid models
                print(
                    "Warning: Could not determine embedding dimension from metadata. Defaulting to 768."
                )
                self.embedding_dim = 768

            print(f"Detected Embedding Dimension: {self.embedding_dim}")

            # Calculate max_tokens_safe if not provided
            if self.max_tokens_safe is None:
                n_ctx = model_info["ctx"]
                if n_ctx and n_ctx > 0:
                    # Estimate words per token (~1.3 words/token is a common conservative estimate)
                    # Reserve some context for prefixes and overhead
                    estimated_words = int(n_ctx / 1.3)
                    # Set safe limit to 80% of estimated capacity to allow for chunking/prefixes
                    self.max_tokens_safe = max(50, int(estimated_words * 0.8))
                    print(
                        f"Calculated max_tokens_safe: {self.max_tokens_safe} (based on ctx={n_ctx})"
                    )
                else:
                    print(
                        "Warning: Could not determine context window. Defaulting max_tokens_safe to 450."
                    )
                    self.max_tokens_safe = 450
            else:
                print(f"Using provided max_tokens_safe: {self.max_tokens_safe}")

        except Exception as e:
            print(f"Error fetching model metadata: {e}. Using defaults.")
            self.embedding_dim = 768
            if self.max_tokens_safe is None:
                self.max_tokens_safe = 450

    def _get_text_stats(self, texts: List[str]) -> Dict:
        """Calculate text length statistics."""
        lengths = [len(text.split()) for text in texts]
        return {
            "lengths": np.array(lengths),
            "mean": np.mean(lengths),
            "median": np.median(lengths),
            "max": np.max(lengths),
        }

    def _encode_with_prefix(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode text with appropriate prefix using jet.embed_utils.
        """
        prefix = self.query_prefix if is_query else self.doc_prefix

        word_count = len(text.split())

        # If text is short enough, encode directly
        if word_count <= self.max_tokens_safe:
            full_text = f"{prefix}{text}" if prefix else text
            emb = embed(full_text, model=self.model_name, return_format="numpy")
            # Ensure 1D array
            if emb.ndim > 1:
                emb = emb.flatten()
            return emb
        else:
            return self._encode_chunked(text, prefix)

    def _encode_chunked(self, text: str, prefix: str) -> np.ndarray:
        """
        Encode long text by splitting into chunks, embedding each,
        and averaging the vectors to preserve global meaning.
        """
        words = text.split()
        # Use max_tokens_safe as chunk size, but ensure it's not too large for a single API call
        # Typically llama.cpp handles batches well, but for chunking logic we want manageable pieces
        chunk_size = min(self.max_tokens_safe, 400)

        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if prefix:
                chunks.append(f"{prefix}{chunk_text}")
            else:
                chunks.append(chunk_text)

        if not chunks:
            return np.zeros(self.embedding_dim)

        # embed_batch handles parallelization and batching internally
        chunk_embeddings = embed(
            chunks,
            model=self.model_name,
            return_format="numpy",
            show_progress=False,  # Disable progress bar for internal chunking
        )

        # Average the embeddings
        avg_embedding = np.mean(chunk_embeddings, axis=0)

        # Normalize the averaged vector
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        return avg_embedding

    def fit_predict(self, texts: List[str]) -> Dict[str, any]:
        """
        Fit and predict clusters for mixed-length texts.

        Args:
            texts: List of text strings of varying lengths

        Returns:
            Dictionary with labels and cluster info
        """
        if not texts:
            raise ValueError("Input texts list is empty")

        stats = self._get_text_stats(texts)
        self.text_lengths = stats["lengths"]
        print(
            f"Text analysis: Max length {stats['max']} words, Mean {stats['mean']:.1f}"
        )
        print(
            f"Generating embeddings for {len(texts)} texts using {self.model_name}..."
        )

        embeddings_list = []
        for i, text in enumerate(texts):
            # Heuristic: very short texts are likely queries/keywords
            is_query = len(text.split()) < 5
            emb = self._encode_with_prefix(text, is_query=is_query)
            embeddings_list.append(emb)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(texts)} texts...")

        self.embeddings = np.array(embeddings_list)

        # Normalize all embeddings for cosine similarity based clustering
        self.embeddings = normalize(self.embeddings)

        n_texts = len(texts)
        if n_texts < 50:
            print("Using Spectral Clustering for small dataset...")
            n_clusters_est = max(2, min(5, n_texts // 3))
            try:
                self.clusterer = SpectralClustering(
                    n_clusters=n_clusters_est,
                    affinity="nearest_neighbors",
                    n_neighbors=min(10, n_texts - 1),
                    assign_labels="kmeans",
                    random_state=42,
                )
                self.labels = self.clusterer.fit_predict(self.embeddings)
            except Exception as e:
                print(f"Spectral failed ({e}), falling back to HDBSCAN")
                self._run_hdbscan()
        else:
            self._run_hdbscan()

        unique_labels = set(self.labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_noise = sum(1 for l in self.labels if l == -1)
        print(f"\nFound {n_clusters} clusters with {n_noise} noise points")

        return {
            "labels": self.labels,
            "embeddings": self.embeddings,
            "text_lengths": self.text_lengths,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }

    def _run_hdbscan(self):
        """Helper to run HDBSCAN configuration."""
        print("Using HDBSCAN for density-based clustering...")
        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=max(1, self.min_cluster_size // 2),
            metric="cosine",
            cluster_selection_epsilon=0.25,
        )
        self.labels = self.clusterer.fit_predict(self.embeddings)

    def get_cluster_texts(self, texts: List[str]) -> Dict[int, List[str]]:
        """Group texts by cluster labels."""
        if self.labels is None:
            raise ValueError("Must call fit_predict first")
        clusters = {}
        for text, label in zip(texts, self.labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text)
        return clusters

    def get_detailed_analysis(self, texts: List[str]) -> Dict:
        """Get detailed cluster analysis including length distribution."""
        clusters = self.get_cluster_texts(texts)
        analysis = {}
        for cluster_id, cluster_texts in clusters.items():
            lengths = [len(t.split()) for t in cluster_texts]
            label_name = (
                f"Cluster {cluster_id}" if cluster_id != -1 else "Noise/Outliers"
            )
            analysis[label_name] = {
                "count": len(cluster_texts),
                "avg_length": np.mean(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "sample_texts": cluster_texts[:2],
            }
        return analysis


if __name__ == "__main__":
    mixed_texts = [
        "machine learning",
        "AI technology",
        "deep learning",
        "neural networks",
        "The product quality exceeded my expectations significantly.",
        "Customer service was unresponsive and quite disappointing.",
        "Fast delivery and excellent packaging throughout.",
        """Cloud computing provides scalable resources on-demand. Major providers like AWS and Azure offer comprehensive services
for businesses of all sizes. This technology enables cost-effective infrastructure management.""",
        """Web development combines frontend and backend technologies. Modern frameworks enable rapid development of responsive
applications. Full-stack developers work across the entire technology stack.""",
        """Artificial intelligence is transforming industries worldwide. Machine learning algorithms can now diagnose diseases,
drive cars, and translate languages with remarkable accuracy. Deep learning, a subset of AI, uses neural networks with
many layers to process complex patterns in data.
The ethical implications of AI development are increasingly important. Questions about bias in algorithms, job displacement,
and privacy concerns need careful consideration. Governments and organizations are developing guidelines for responsible AI use.
Future advancements in quantum computing may accelerate AI capabilities exponentially. Researchers are exploring new architectures
that could solve previously intractable problems in optimization and simulation.""",
        """Data science combines statistical analysis, programming, and domain expertise. Data scientists extract meaningful insights
from large datasets to inform business decisions. They use tools like Python, R, and SQL to clean, analyze, and visualize data.
The field continues to evolve with new techniques in deep learning and automated machine learning. Organizations are investing
heavily in data infrastructure and talent to remain competitive in the data-driven economy.""",
        "cybersecurity",
        "blockchain technology",
        "The app crashes frequently during peak hours.",
        "Great user interface but needs better documentation.",
        """Internet of Things connects devices globally. Smart homes automate lighting, temperature, and security.
Industrial IoT optimizes manufacturing processes through real-time monitoring.""",
        "Python programming",
        "JavaScript development",
        """DevOps bridges development and operations teams. Continuous integration and deployment automate software releases.
Containerization with Docker ensures consistent environments across development and production.""",
        "The battery life is terrible, lasting only 3 hours.",
        "Excellent value for money with premium features included.",
    ]

    print("=" * 60)
    print("Mixed-Length Text Clustering Demo (Llama.cpp)")
    print("=" * 60)

    # Initialize with defaults from config
    clusterer = MixedLengthClusterer(
        min_cluster_size=2,
        # max_tokens_safe=None will trigger dynamic calculation
    )

    results = clusterer.fit_predict(mixed_texts)
    clusters = clusterer.get_cluster_texts(mixed_texts)
    analysis = clusterer.get_detailed_analysis(mixed_texts)

    print("\n" + "=" * 60)
    print("DETAILED CLUSTER ANALYSIS:")
    print("=" * 60)
    for cluster_name, info in sorted(analysis.items()):
        print(f"\n{cluster_name}:")
        print(f"  Count: {info['count']}")
        print(f"  Length range: {info['min_length']} - {info['max_length']} words")
        print(f"  Average length: {info['avg_length']:.1f} words")
        print(f"  Sample texts:")
        for text in info["sample_texts"]:
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"    • {preview}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {results['n_clusters']} clusters, {results['n_noise']} outliers")
    print(f"{'=' * 60}")
