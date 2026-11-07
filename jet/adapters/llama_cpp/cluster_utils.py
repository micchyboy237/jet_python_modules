"""
Clustering for mixed-length texts (words, sentences, and documents).
Uses configured Llama.cpp embedding models for semantic understanding.
Handles varying lengths by normalizing embeddings and chunking long texts.
Dynamically configures dimensions and safety thresholds based on model metadata.
"""

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import hdbscan
import numpy as np
from jet.adapters.llama_cpp.config import (
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# Define output directory relative to this script
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class TextsClusterer:
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
        self.probabilities = None
        self.centroids = {}

    def _initialize_model_params(self):
        """Fetch model metadata to set embedding dim and safe token limits."""
        try:
            print(f"Fetching metadata for model: {self.model_name}...")
            model_info = get_model_ctx_embd_size(self.model_name)

            # Set embedding dimension
            self.embedding_dim = model_info["embd_dims"]
            if not self.embedding_dim or self.embedding_dim == 0:
                print(
                    "Warning: Could not determine embedding dimension from metadata. Defaulting to 768."
                )
                self.embedding_dim = 768

            print(f"Detected Embedding Dimension: {self.embedding_dim}")

            # Calculate max_tokens_safe if not provided
            if self.max_tokens_safe is None:
                n_ctx = model_info["ctx"]
                if n_ctx and n_ctx > 0:
                    estimated_words = int(n_ctx / 1.3)
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

        chunk_embeddings = embed(
            chunks,
            model=self.model_name,
            return_format="numpy",
            show_progress=False,
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
        if n_texts < 10:
            print("Using Spectral Clustering for very small dataset...")
            n_clusters_est = max(2, min(3, n_texts // 2))
            try:
                self.clusterer = SpectralClustering(
                    n_clusters=n_clusters_est,
                    affinity="nearest_neighbors",
                    n_neighbors=min(5, n_texts - 1),
                    assign_labels="kmeans",
                    random_state=42,
                )
                self.labels = self.clusterer.fit_predict(self.embeddings)
                self.probabilities = self._calculate_custom_confidence(
                    self.embeddings, self.labels
                )
                print("Calculated custom confidence scores for SpectralClustering")
            except Exception as e:
                print(f"Spectral failed ({e}), falling back to HDBSCAN")
                self._run_hdbscan()
        else:
            self._run_hdbscan()

        unique_labels = set(self.labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_noise = sum(1 for l in self.labels if l == -1)
        print(f"\nFound {n_clusters} clusters with {n_noise} noise points")

        # Calculate centroids for all clusters
        self._calculate_centroids()

        return {
            "labels": self.labels,
            "embeddings": self.embeddings,
            "text_lengths": self.text_lengths,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "probabilities": self.probabilities,
        }

    def _run_hdbscan(self):
        """Helper to run HDBSCAN configuration using standalone hdbscan library."""
        print("Using HDBSCAN for density-based clustering...")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=max(1, self.min_cluster_size // 2),
            metric="euclidean",
            cluster_selection_epsilon=0.25,
        )
        self.labels = self.clusterer.fit_predict(self.embeddings)

        # Extract membership probabilities
        try:
            self.probabilities = self.clusterer.probabilities_
            print(
                f"Extracted membership probabilities for {len(self.probabilities)} points"
            )
        except Exception as e:
            print(f"Warning: Could not extract probabilities: {e}")
            self.probabilities = np.zeros(len(self.labels))

    def _calculate_custom_confidence(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Calculate custom confidence scores for SpectralClustering results.
        Uses distance to cluster centroid as a proxy for membership strength.
        """
        unique_labels = np.unique(labels)
        confidence_scores = np.zeros(len(labels))

        for label in unique_labels:
            if label == -1:
                continue

            cluster_mask = labels == label
            cluster_embeddings = embeddings[cluster_mask]

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm

            # Calculate cosine similarity to centroid
            similarities = cosine_similarity(
                cluster_embeddings, centroid.reshape(1, -1)
            ).flatten()

            confidence_scores[cluster_mask] = np.maximum(similarities, 0)

        return confidence_scores

    def _calculate_centroids(self):
        """Calculate centroid vector for each cluster."""
        unique_labels = np.unique(self.labels)
        self.centroids = {}

        for label in unique_labels:
            if label == -1:
                continue

            cluster_mask = self.labels == label
            cluster_embeddings = self.embeddings[cluster_mask]

            # Mean vector
            centroid = np.mean(cluster_embeddings, axis=0)

            # Normalize centroid for consistency
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm

            self.centroids[label] = centroid

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
        """Get detailed cluster analysis including length distribution, outliers, and centroids."""
        clusters = self.get_cluster_texts(texts)
        analysis = {}

        # Sort keys so Noise (-1) is handled last
        sorted_keys = sorted(clusters.keys(), key=lambda x: (x == -1, x))

        for cluster_id in sorted_keys:
            cluster_texts = clusters[cluster_id]
            lengths = [len(t.split()) for t in cluster_texts]

            # Get indices of texts in this cluster
            cluster_indices = [
                i for i, label in enumerate(self.labels) if label == cluster_id
            ]

            # Get probabilities for this cluster
            if self.probabilities is not None:
                cluster_probs = [self.probabilities[i] for i in cluster_indices]
            else:
                cluster_probs = [0.0] * len(cluster_texts)

            # Calculate distance to centroid for each text
            cluster_distances = []
            if cluster_id in self.centroids:
                centroid = self.centroids[cluster_id]
                cluster_embeddings = self.embeddings[cluster_indices]
                # Cosine distance = 1 - cosine_similarity
                sims = cosine_similarity(
                    cluster_embeddings, centroid.reshape(1, -1)
                ).flatten()
                cluster_distances = [float(1 - s) for s in sims]
            else:
                cluster_distances = [0.0] * len(cluster_texts)

            # Create list of tuples (text, prob, distance) and sort by prob descending
            combined = list(zip(cluster_texts, cluster_probs, cluster_distances))
            combined.sort(key=lambda x: x[1], reverse=True)

            sorted_texts, sorted_probs, sorted_dists = (
                zip(*combined) if combined else ([], [], [])
            )

            if cluster_id == -1:
                label_name = "Noise/Outliers"
            else:
                label_name = f"Cluster {cluster_id}"

            analysis[label_name] = {
                "count": len(cluster_texts),
                "avg_length": float(np.mean(lengths)),
                "min_length": int(min(lengths)),
                "max_length": int(max(lengths)),
                "all_texts": list(sorted_texts),
                "probabilities": [float(p) for p in sorted_probs],
                "distances_to_centroid": [float(d) for d in sorted_dists],
            }

            # Add centroid info if available
            if cluster_id in self.centroids:
                analysis[label_name]["centroid_vector_sample"] = self.centroids[
                    cluster_id
                ][:5].tolist()  # First 5 dims for brevity

        return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Cluster mixed-length texts using Llama.cpp embeddings."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to JSON file containing an array of strings.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Directory to save results. Defaults to {OUTPUT_DIR}",
    )
    parser.add_argument(
        "-m",
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size for HDBSCAN.",
    )
    parser.add_argument(
        "-n",
        "--model",
        type=str,
        default=EMBED_MODEL,
        help="Model name to use for embeddings.",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load input data
    try:
        with open(args.input, "r") as f:
            texts = json.load(f)
        if not isinstance(texts, list):
            raise ValueError("JSON root must be an array of strings.")
        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All items in the JSON array must be strings.")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    print("=" * 60)
    print("Mixed-Length Text Clustering Demo (Llama.cpp)")
    print("=" * 60)

    clusterer = TextsClusterer(
        model_name=args.model,
        min_cluster_size=args.min_cluster_size,
    )

    results = clusterer.fit_predict(texts)
    analysis = clusterer.get_detailed_analysis(texts)

    print("\n" + "=" * 60)
    print("DETAILED CLUSTER ANALYSIS:")
    print("=" * 60)

    regular_clusters = {k: v for k, v in analysis.items() if k != "Noise/Outliers"}
    noise_cluster = {k: v for k, v in analysis.items() if k == "Noise/Outliers"}

    for cluster_name, info in sorted(regular_clusters.items()):
        print(f"\n{cluster_name}:")
        print(f"  Count: {info['count']}")
        print(f"  Length range: {info['min_length']} - {info['max_length']} words")
        print(f"  Average length: {info['avg_length']:.1f} words")
        print(f"  Top Members (sorted by probability):")
        for idx, (text, prob, dist) in enumerate(
            zip(
                info["all_texts"], info["probabilities"], info["distances_to_centroid"]
            ),
            1,
        ):
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"    {idx}. [Prob: {prob:.3f}] [Dist: {dist:.3f}] {preview}")

    if noise_cluster:
        for cluster_name, info in noise_cluster.items():
            print(f"\n{cluster_name}:")
            print(f"  Count: {info['count']}")
            print(f"  All texts:")
            for idx, text in enumerate(info["all_texts"], 1):
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"    {idx}. {preview}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {results['n_clusters']} clusters, {results['n_noise']} outliers")
    print(f"{'=' * 60}")

    # Save results to JSON files
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # 1. Save Detailed Analysis
    analysis_file = out_path / "analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)
    print(f"\nDetailed analysis saved to {analysis_file}")

    # 2. Save Summary
    summary_data = {
        "summary": {
            "n_clusters": results["n_clusters"],
            "n_noise": results["n_noise"],
            "total_texts": len(texts),
        },
        "cluster_sizes": {k: v["count"] for k, v in analysis.items()},
    }
    summary_file = out_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, cls=NumpyEncoder)
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()
