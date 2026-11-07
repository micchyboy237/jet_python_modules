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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jet.adapters.llama_cpp.config import (
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
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
        self.model_name = model_name
        self.min_cluster_size = min_cluster_size
        self.query_prefix = (
            query_prefix if query_prefix is not None else EMBED_QUERY_PREFIX
        )
        self.doc_prefix = doc_prefix if doc_prefix is not None else EMBED_DOC_PREFIX

        self.embedding_dim = None
        self.max_tokens_safe = max_tokens_safe
        self._initialize_model_params()

        self.embeddings = None
        self.labels = None
        self.text_lengths = None
        self.probabilities = None
        self.centroids = {}
        self.cluster_consistency = {}
        self.coordinates_2d = None
        self.silhouette_scores = None

    def _initialize_model_params(self):
        """Fetch model metadata to set embedding dim and safe token limits."""
        try:
            print(f"Fetching metadata for model: {self.model_name}...")
            model_info = get_model_ctx_embd_size(self.model_name)
            self.embedding_dim = model_info["embd_dims"]
            if not self.embedding_dim or self.embedding_dim == 0:
                self.embedding_dim = 768
            print(f"Detected Embedding Dimension: {self.embedding_dim}")

            if self.max_tokens_safe is None:
                n_ctx = model_info["ctx"]
                if n_ctx and n_ctx > 0:
                    estimated_words = int(n_ctx / 1.3)
                    self.max_tokens_safe = max(50, int(estimated_words * 0.8))
                    print(
                        f"Calculated max_tokens_safe: {self.max_tokens_safe} (based on ctx={n_ctx})"
                    )
                else:
                    self.max_tokens_safe = 450
            else:
                print(f"Using provided max_tokens_safe: {self.max_tokens_safe}")
        except Exception as e:
            print(f"Error fetching model metadata: {e}. Using defaults.")
            self.embedding_dim = 768
            if self.max_tokens_safe is None:
                self.max_tokens_safe = 450

    def _get_text_stats(self, texts: List[str]) -> Dict:
        lengths = [len(text.split()) for text in texts]
        return {
            "lengths": np.array(lengths),
            "mean": np.mean(lengths),
            "median": np.median(lengths),
            "max": np.max(lengths),
        }

    def _encode_with_prefix(self, text: str, is_query: bool = False) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.doc_prefix
        word_count = len(text.split())
        if word_count <= self.max_tokens_safe:
            full_text = f"{prefix}{text}" if prefix else text
            emb = embed(full_text, model=self.model_name, return_format="numpy")
            if emb.ndim > 1:
                emb = emb.flatten()
            return emb
        else:
            return self._encode_chunked(text, prefix)

    def _encode_chunked(self, text: str, prefix: str) -> np.ndarray:
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
            chunks, model=self.model_name, return_format="numpy", show_progress=False
        )
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        return avg_embedding

    def fit_predict(self, texts: List[str]) -> Dict[str, any]:
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
            is_query = len(text.split()) < 5
            emb = self._encode_with_prefix(text, is_query=is_query)
            embeddings_list.append(emb)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(texts)} texts...")

        self.embeddings = np.array(embeddings_list)
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
            except Exception as e:
                print(f"Spectral failed ({e}), falling back to HDBSCAN")
                self._run_hdbscan()
        else:
            self._run_hdbscan()

        unique_labels = set(self.labels)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_noise = sum(1 for l in self.labels if l == -1)
        print(f"\nFound {n_clusters} clusters with {n_noise} noise points")

        self._calculate_centroids()
        self._generate_2d_coordinates()
        self._calculate_silhouette_scores()

        return {
            "labels": self.labels,
            "embeddings": self.embeddings,
            "text_lengths": self.text_lengths,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "probabilities": self.probabilities,
        }

    def _run_hdbscan(self):
        print("Using HDBSCAN for density-based clustering...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=max(1, self.min_cluster_size // 2),
            metric="euclidean",
            cluster_selection_epsilon=0.25,
        )
        self.labels = self.clusterer.fit_predict(self.embeddings)
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
        unique_labels = np.unique(labels)
        confidence_scores = np.zeros(len(labels))
        for label in unique_labels:
            if label == -1:
                continue
            cluster_mask = labels == label
            cluster_embeddings = embeddings[cluster_mask]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            similarities = cosine_similarity(
                cluster_embeddings, centroid.reshape(1, -1)
            ).flatten()
            confidence_scores[cluster_mask] = np.maximum(similarities, 0)
        return confidence_scores

    def _calculate_centroids(self):
        unique_labels = np.unique(self.labels)
        self.centroids = {}
        self.cluster_consistency = {}
        for label in unique_labels:
            if label == -1:
                continue
            cluster_mask = self.labels == label
            cluster_embeddings = self.embeddings[cluster_mask]
            raw_centroid = np.mean(cluster_embeddings, axis=0)
            magnitude = np.linalg.norm(raw_centroid)
            self.cluster_consistency[label] = float(magnitude)
            if magnitude > 0:
                self.centroids[label] = raw_centroid / magnitude
            else:
                self.centroids[label] = raw_centroid

    def _generate_2d_coordinates(self):
        """Generate 2D coordinates using t-SNE for visualization."""
        if self.embeddings is None or len(self.embeddings) < 2:
            self.coordinates_2d = np.zeros(
                (len(self.embeddings) if self.embeddings is not None else 0, 2)
            )
            return

        print("Generating 2D projection for visualization...")
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(5, len(self.embeddings) - 1),
            max_iter=1000,
        )
        self.coordinates_2d = tsne.fit_transform(self.embeddings)

    def _calculate_silhouette_scores(self):
        """Calculate Silhouette Score for each sample."""
        # Filter out noise points (-1) for silhouette calculation
        mask = self.labels != -1
        if np.sum(mask) < 2:
            self.silhouette_scores = np.zeros(len(self.labels))
            return

        # Silhouette score requires at least 2 clusters
        unique_labels = np.unique(self.labels[mask])
        if len(unique_labels) < 2:
            self.silhouette_scores = np.zeros(len(self.labels))
            return

        try:
            # Calculate scores only for clustered points
            sample_scores = silhouette_samples(
                self.embeddings[mask], self.labels[mask], metric="cosine"
            )

            # Create full array with 0 for noise points
            self.silhouette_scores = np.zeros(len(self.labels))
            self.silhouette_scores[mask] = sample_scores
        except Exception as e:
            print(f"Warning: Could not calculate silhouette scores: {e}")
            self.silhouette_scores = np.zeros(len(self.labels))

    def get_cluster_texts(self, texts: List[str]) -> Dict[int, List[str]]:
        if self.labels is None:
            raise ValueError("Must call fit_predict first")
        clusters = {}
        for text, label in zip(texts, self.labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(text)
        return clusters

    def get_detailed_analysis(self, texts: List[str]) -> Dict:
        clusters = self.get_cluster_texts(texts)
        analysis = {}
        sorted_keys = sorted(clusters.keys(), key=lambda x: (x == -1, x))

        for cluster_id in sorted_keys:
            cluster_texts = clusters[cluster_id]
            lengths = [len(t.split()) for t in cluster_texts]
            cluster_indices = [
                i for i, label in enumerate(self.labels) if label == cluster_id
            ]

            if self.probabilities is not None:
                cluster_probs = [self.probabilities[i] for i in cluster_indices]
            else:
                cluster_probs = [0.0] * len(cluster_texts)

            cluster_distances = []
            if cluster_id in self.centroids:
                centroid = self.centroids[cluster_id]
                cluster_embeddings = self.embeddings[cluster_indices]
                sims = cosine_similarity(
                    cluster_embeddings, centroid.reshape(1, -1)
                ).flatten()
                cluster_distances = [float(1 - s) for s in sims]
            else:
                cluster_distances = [0.0] * len(cluster_texts)

            cluster_coords = []
            if self.coordinates_2d is not None:
                cluster_coords = [
                    self.coordinates_2d[i].tolist() for i in cluster_indices
                ]

            cluster_silhouettes = []
            if self.silhouette_scores is not None:
                cluster_silhouettes = [
                    float(self.silhouette_scores[i]) for i in cluster_indices
                ]

            combined = list(
                zip(
                    cluster_texts,
                    cluster_probs,
                    cluster_distances,
                    cluster_coords,
                    cluster_silhouettes,
                )
            )
            # Sort by probability descending
            combined.sort(key=lambda x: x[1], reverse=True)

            (
                sorted_texts,
                sorted_probs,
                sorted_dists,
                sorted_coords,
                sorted_silhouettes,
            ) = zip(*combined) if combined else ([], [], [], [], [])

            label_name = (
                "Noise/Outliers" if cluster_id == -1 else f"Cluster {cluster_id}"
            )
            analysis[label_name] = {
                "count": len(cluster_texts),
                "avg_length": float(np.mean(lengths)),
                "min_length": int(min(lengths)),
                "max_length": int(max(lengths)),
                "all_texts": list(sorted_texts),
                "probabilities": [float(p) for p in sorted_probs],
                "distances_to_centroid": [float(d) for d in sorted_dists],
                "coordinates_2d": [list(c) for c in sorted_coords],
                "silhouette_scores": [float(s) for s in sorted_silhouettes],
            }
            if cluster_id in self.centroids:
                analysis[label_name]["centroid_vector_sample"] = self.centroids[
                    cluster_id
                ][:5].tolist()
                analysis[label_name]["consistency_score"] = (
                    self.cluster_consistency.get(cluster_id, 0.0)
                )
        return analysis

    def plot_clusters(self, texts: List[str], output_path: Path):
        """Generate a 2D scatter plot of the clusters using stored coordinates."""
        if self.coordinates_2d is None:
            return

        plt.figure(figsize=(12, 8))
        unique_labels = np.unique(self.labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            x = self.coordinates_2d[mask, 0]
            y = self.coordinates_2d[mask, 1]

            label_name = "Noise" if label == -1 else f"Cluster {label}"
            plt.scatter(x, y, c=[colors[i]], label=label_name, alpha=0.7, s=100)

            for j, idx in enumerate(np.where(mask)[0]):
                preview = (
                    texts[idx][:20] + "..." if len(texts[idx]) > 20 else texts[idx]
                )
                plt.annotate(preview, (x[j], y[j]), fontsize=8, alpha=0.8)

        plt.title("Text Clusters Visualization (t-SNE)")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")


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
        help=f"Directory to save results.",
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

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        with open(args.input, "r") as f:
            texts = json.load(f)
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("JSON root must be an array of strings.")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    print("=" * 60)
    print("Mixed-Length Text Clustering Demo (Llama.cpp)")
    print("=" * 60)

    clusterer = TextsClusterer(
        model_name=args.model, min_cluster_size=args.min_cluster_size
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
        print(
            f"  Count: {info['count']} | Consistency: {info.get('consistency_score', 0):.3f}"
        )
        print(f"  Top Members:")
        for idx, (text, prob, dist, sil) in enumerate(
            zip(
                info["all_texts"],
                info["probabilities"],
                info["distances_to_centroid"],
                info["silhouette_scores"],
            ),
            1,
        ):
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"    {idx}. [P:{prob:.2f}] [S:{sil:.2f}] {preview}")

    if noise_cluster:
        print(f"\nNoise/Outliers ({noise_cluster['Noise/Outliers']['count']} items):")
        for text in noise_cluster["Noise/Outliers"]["all_texts"]:
            print(f"  - {text[:80]}...")

    plot_file = out_path / "clusters_plot.png"
    clusterer.plot_clusters(texts, plot_file)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)

    summary_data = {
        "summary": {
            "n_clusters": results["n_clusters"],
            "n_noise": results["n_noise"],
            "total_texts": len(texts),
        },
        "cluster_sizes": {k: v["count"] for k, v in analysis.items()},
    }
    with open(out_path / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
