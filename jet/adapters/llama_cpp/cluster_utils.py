"""
Clustering for mixed-length texts (words, sentences, and documents).
Features:
    - Semantic Understanding: Uses configured Llama.cpp embedding models.
    - Mixed-Length Handling: Normalizes embeddings and chunks long texts dynamically.
    - Dynamic Configuration: Adjusts dimensions and safety thresholds based on model metadata.
    - Advanced Metrics: Provides membership probabilities, silhouette scores, and consistency scores.
    - Visualization: Generates 2D t-SNE plots and saves detailed JSON analysis.
    - Traceability: Tracks original indices and handles duplicates efficiently.
Usage Examples:
    1. Basic usage with default settings:
       python cluster_utils.py -i input_texts.json
    2. Strict clustering (tighter groups, more noise):
       python cluster_utils.py -i input_texts.json -s 5 -e 0.1
    3. Loose clustering (broader topics, less noise):
       python cluster_utils.py -i input_texts.json -s 2 -e 0.5
    4. Custom output directory and model:
       python cluster_utils.py -i input_texts.json -o ./my_results -n nomic-embed:2-moe
"""

import argparse
import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

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

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ClustersType = Dict[int, List[str]]
CentroidsType = Dict[int, np.ndarray]
ConsistencyType = Dict[int, float]


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------
class TextStats(TypedDict):
    lengths: np.ndarray
    mean: float
    median: float
    max: int


class FitPredictResult(TypedDict):
    labels: np.ndarray
    embeddings: np.ndarray
    text_lengths: np.ndarray
    n_clusters: int
    n_noise: int
    probabilities: Optional[np.ndarray]
    original_indices: np.ndarray
    n_original_texts: int
    n_unique_texts: int


class ClusterAnalysisItem(TypedDict, total=False):
    count: int
    original_indices: List[int]
    avg_length: float
    min_length: int
    max_length: int
    all_texts: List[str]
    probabilities: List[float]
    distances_to_centroid: List[float]
    coordinates_2d: List[List[float]]
    silhouette_scores: List[float]
    centroid_vector_sample: List[float]
    consistency_score: float


DetailedAnalysis = Dict[str, ClusterAnalysisItem]


class ClusterDetailSummary(TypedDict):
    count: int
    consistency_score: float
    avg_silhouette_score: float
    misplaced_items_count: int
    is_noise: bool


class SummaryData(TypedDict):
    summary: Dict[str, int]
    cluster_details: Dict[str, ClusterDetailSummary]


# ---------------------------------------------------------------------------
# Config / JSON
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingConfig:
    """Constructor-time encoding configuration, passed to standalone helpers."""

    model_name: str = EMBED_MODEL
    max_tokens_safe: int = 450
    embedding_dim: int = 768
    query_prefix: str = EMBED_QUERY_PREFIX
    doc_prefix: str = EMBED_DOC_PREFIX


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Standalone: deduplication
# ---------------------------------------------------------------------------
def deduplicate_texts(texts: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    Deduplicates texts while maintaining a map of original indices.

    Returns:
        unique_texts: List of unique text strings (first-occurrence order).
        index_map: Dictionary mapping each unique text to the list of its
            ORIGINAL POSITIONAL INDICES in the input list. Note: these are
            positions in the input, not indices into ``unique_texts``.
    """
    unique_texts = []
    index_map: Dict[str, List[int]] = {}
    seen = set()
    for i, text in enumerate(texts):
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
            index_map[text] = []
        index_map[text].append(i)
    return unique_texts, index_map


# ---------------------------------------------------------------------------
# Standalone: model param resolution
# ---------------------------------------------------------------------------
def resolve_model_params(
    model_name: str,
    max_tokens_safe: Optional[int],
) -> Tuple[int, int]:
    """
    Resolve (embedding_dim, max_tokens_safe) for a model, falling back to
    defaults when metadata cannot be fetched.
    """
    try:
        print(f"Fetching metadata for model: {model_name}...")
        model_info = get_model_ctx_embd_size(model_name)
        embedding_dim = model_info["embd_dims"] or 0
        if not embedding_dim:
            embedding_dim = 768
        print(f"Detected Embedding Dimension: {embedding_dim}")
        if max_tokens_safe is None:
            n_ctx = model_info["ctx"]
            if n_ctx and n_ctx > 0:
                estimated_words = int(n_ctx / 1.3)
                max_tokens_safe = max(50, int(estimated_words * 0.8))
                print(
                    f"Calculated max_tokens_safe: {max_tokens_safe} "
                    f"(based on ctx={n_ctx})"
                )
            else:
                max_tokens_safe = 450
        else:
            print(f"Using provided max_tokens_safe: {max_tokens_safe}")
        return embedding_dim, max_tokens_safe
    except Exception as e:
        print(f"Error fetching model metadata: {e}. Using defaults.")
        return 768, max_tokens_safe if max_tokens_safe is not None else 450


# ---------------------------------------------------------------------------
# Standalone: text stats
# ---------------------------------------------------------------------------
def compute_text_stats(texts: List[str]) -> TextStats:
    """Compute word-count statistics for a list of texts."""
    lengths = [len(t.split()) for t in texts]
    return {
        "lengths": np.array(lengths),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "max": int(np.max(lengths)),
    }


# ---------------------------------------------------------------------------
# Standalone: encoding
# ---------------------------------------------------------------------------
def encode_chunked(
    text: str,
    *,
    prefix: str,
    model_name: str,
    max_tokens_safe: int,
    embedding_dim: int,
) -> np.ndarray:
    """
    Chunk a long text by word window, embed each chunk, and return the
    L2-normalized mean embedding. Returns a zero vector if ``text`` is empty.
    """
    words = text.split()
    chunk_size = min(max_tokens_safe, 400)
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i : i + chunk_size])
        chunks.append(f"{prefix}{chunk_text}" if prefix else chunk_text)
    if not chunks:
        return np.zeros(embedding_dim)
    chunk_embeddings = embed(
        chunks, model=model_name, return_format="numpy", show_progress=False
    )
    avg = chunk_embeddings.mean(axis=0)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg


def encode_with_prefix(
    text: str,
    *,
    config: EmbeddingConfig,
    is_query: bool = False,
) -> np.ndarray:
    """
    Embed a single text, applying the query/doc prefix and dispatching to
    chunked encoding for long inputs. Pure function of its arguments.
    """
    prefix = config.query_prefix if is_query else config.doc_prefix
    word_count = len(text.split())
    if word_count <= config.max_tokens_safe:
        full_text = f"{prefix}{text}" if prefix else text
        emb = embed(full_text, model=config.model_name, return_format="numpy")
        if emb.ndim > 1:
            emb = emb.flatten()
        return emb
    return encode_chunked(
        text,
        prefix=prefix,
        model_name=config.model_name,
        max_tokens_safe=config.max_tokens_safe,
        embedding_dim=config.embedding_dim,
    )


# ---------------------------------------------------------------------------
# Standalone: metrics
# ---------------------------------------------------------------------------
def calculate_custom_confidence(
    embeddings: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """
    Confidence = max(cosine similarity to own cluster centroid, 0).
    Noise points (-1) receive 0.
    """
    unique_labels = np.unique(labels)
    scores = np.zeros(len(labels))
    for label in unique_labels:
        if label == -1:
            continue
        mask = labels == label
        centroid = embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        sims = cosine_similarity(embeddings[mask], centroid.reshape(1, -1)).flatten()
        scores[mask] = np.maximum(sims, 0)
    return scores


# ---------------------------------------------------------------------------
# Standalone: plotting
# ---------------------------------------------------------------------------
def plot_cluster_scatter(
    coordinates_2d: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    output_path: Path,
) -> None:
    """Render a 2D scatter plot of clusters using precomputed coordinates."""
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        x = coordinates_2d[mask, 0]
        y = coordinates_2d[mask, 1]
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        plt.scatter(x, y, c=[colors[i]], label=label_name, alpha=0.7, s=100)
        for j, idx in enumerate(np.where(mask)[0]):
            preview = texts[idx][:20] + "..." if len(texts[idx]) > 20 else texts[idx]
            plt.annotate(preview, (x[j], y[j]), fontsize=8, alpha=0.8)
    plt.title("Text Clusters Visualization (t-SNE)")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Standalone: reporting
# ---------------------------------------------------------------------------
def _get_cluster_sort_key(name: str) -> float:
    """Sort cluster labels numerically; push non-numeric labels last."""
    try:
        return float(name.split(" ")[1])
    except (IndexError, ValueError):
        return float("inf")


def print_cluster_analysis(
    analysis: DetailedAnalysis, *, include_noise: bool = True
) -> None:
    """Pretty-print a detailed cluster analysis to stdout."""
    regular_clusters = {k: v for k, v in analysis.items() if k != "Noise/Outliers"}
    noise_cluster = {k: v for k, v in analysis.items() if k == "Noise/Outliers"}

    sorted_names = sorted(regular_clusters.keys(), key=_get_cluster_sort_key)

    for cluster_name in sorted_names:
        info = regular_clusters[cluster_name]
        print(f"\n{cluster_name}:")
        print(
            f"  Count: {info['count']} | "
            f"Consistency: {info.get('consistency_score', 0):.3f} | "
            f"Length range: {info['min_length']}-{info['max_length']} words | "
            f"Avg: {info['avg_length']:.1f}"
        )
        print("  Top Members (sorted by Silhouette Score):")
        for idx, (text, orig_idx, prob, dist, sil) in enumerate(
            zip(
                info["all_texts"],
                info["original_indices"],
                info["probabilities"],
                info["distances_to_centroid"],
                info["silhouette_scores"],
            ),
            1,
        ):
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"    {idx}. [Idx:{orig_idx}] [S:{sil:.2f}] [P:{prob:.2f}] {preview}")

    if include_noise and noise_cluster:
        noise_info = noise_cluster["Noise/Outliers"]
        print(f"\nNoise/Outliers ({noise_info['count']} items):")
        for text, orig_idx in zip(
            noise_info["all_texts"], noise_info["original_indices"]
        ):
            print(f"  - [Idx:{orig_idx}] {text[:80]}...")


def save_cluster_results(
    output_dir: Path,
    analysis: DetailedAnalysis,
    summary: SummaryData,
    *,
    coordinates_2d: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    save_plot: bool = True,
) -> Dict[str, Path]:
    """
    Persist ``clusters_plot.png`` (optional), ``analysis.json``, and
    ``summary.json`` under ``output_dir``. Returns a mapping of artifact
    name to path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    if save_plot and coordinates_2d is not None and labels is not None and texts:
        plot_path = output_dir / "clusters_plot.png"
        plot_cluster_scatter(coordinates_2d, labels, texts, plot_path)
        paths["plot"] = plot_path

    analysis_path = output_dir / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    paths["analysis"] = analysis_path

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
    paths["summary"] = summary_path

    return paths


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------
class TextsClusterer:
    """
    Clusters texts of varying lengths using configured Llama.cpp embeddings.

    Public API:
        fit_predict(texts) -> FitPredictResult
        get_detailed_analysis() -> DetailedAnalysis
        get_summary(analysis=None) -> SummaryData
        cluster(texts) -> (FitPredictResult, DetailedAnalysis)

    Reporting artifacts (plot, JSON) are produced by the module-level
    ``print_cluster_analysis`` / ``save_cluster_results`` helpers, using the
    state exposed on the instance after ``fit_predict``.
    """

    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        min_cluster_size: int = 2,
        max_tokens_safe: Optional[int] = None,
        query_prefix: Optional[str] = None,
        doc_prefix: Optional[str] = None,
        cluster_selection_epsilon: float = 0.25,
    ):
        self.model_name = model_name
        self.min_cluster_size = min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.query_prefix = (
            query_prefix if query_prefix is not None else EMBED_QUERY_PREFIX
        )
        self.doc_prefix = doc_prefix if doc_prefix is not None else EMBED_DOC_PREFIX
        self.embedding_dim: Optional[int] = None
        self.max_tokens_safe: Optional[int] = max_tokens_safe
        self._initialize_model_params()
        self._embedding_config = EmbeddingConfig(
            model_name=self.model_name,
            max_tokens_safe=self.max_tokens_safe,
            embedding_dim=self.embedding_dim,
            query_prefix=self.query_prefix,
            doc_prefix=self.doc_prefix,
        )
        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.text_lengths: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None
        self.centroids: CentroidsType = {}
        self.cluster_consistency: ConsistencyType = {}
        self.coordinates_2d: Optional[np.ndarray] = None
        self.silhouette_scores: Optional[np.ndarray] = None
        self.processed_indices: Optional[np.ndarray] = None
        # State captured by fit_predict
        self._unique_texts: Optional[List[str]] = None
        self._index_map: Optional[Dict[str, List[int]]] = None
        self._n_original: Optional[int] = None

    # -- config ------------------------------------------------------------
    def _initialize_model_params(self) -> None:
        """Populate embedding_dim / max_tokens_safe via the pure resolver."""
        self.embedding_dim, self.max_tokens_safe = resolve_model_params(
            self.model_name, self.max_tokens_safe
        )

    # -- dedup -------------------------------------------------------------
    def _deduplicate(
        self, texts: List[str], indices: List[int]
    ) -> Tuple[List[str], List[int]]:
        """
        Deduplicate texts while preserving caller-supplied original indices.
        Captures ``_unique_texts``, ``_index_map``, ``_n_original`` on self.
        """
        index_map: Dict[str, List[int]] = {}
        unique_texts: List[str] = []
        for local_i, text in enumerate(texts):
            if text not in index_map:
                unique_texts.append(text)
                index_map[text] = []
            index_map[text].append(indices[local_i])
        unique_indices = [index_map[t][0] for t in unique_texts]
        self._unique_texts = unique_texts
        self._index_map = index_map
        self._n_original = len(texts)
        return unique_texts, unique_indices

    # -- pipeline ----------------------------------------------------------
    def fit_predict(
        self,
        texts: List[str],
        indices: Optional[List[int]] = None,
        *,
        deduplicate: bool = True,
    ) -> FitPredictResult:
        """
        Fits the clusterer to the provided texts.

        Args:
            texts: List of text strings.
            indices: Optional list of original indices corresponding to the
                texts. If None, assumes indices are 0..N-1.
            deduplicate: If True (default), duplicate texts are collapsed
                before embedding; the index map is captured on the instance
                so ``get_detailed_analysis`` can expand ``original_indices``
                back to every occurrence.
        """
        if not texts:
            raise ValueError("Input texts list is empty")
        if indices is None:
            indices = list(range(len(texts)))
        if len(indices) != len(texts):
            raise ValueError(
                f"Length mismatch: {len(texts)} texts vs {len(indices)} indices"
            )

        if deduplicate:
            unique_texts, unique_indices = self._deduplicate(texts, indices)
            print(
                f"Deduplicated: {len(texts)} -> {len(unique_texts)} texts "
                f"(removed {len(texts) - len(unique_texts)})"
            )
            texts = unique_texts
            indices = unique_indices
        else:
            self._unique_texts = list(texts)
            self._index_map = {t: [indices[i]] for i, t in enumerate(texts)}
            self._n_original = len(texts)

        self.processed_indices = np.array(indices)

        stats = compute_text_stats(texts)
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
            emb = encode_with_prefix(
                text, config=self._embedding_config, is_query=is_query
            )
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
                self.probabilities = calculate_custom_confidence(
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
            "original_indices": self.processed_indices,
            "n_original_texts": (
                self._n_original if self._n_original is not None else len(texts)
            ),
            "n_unique_texts": (
                len(self._unique_texts)
                if self._unique_texts is not None
                else len(texts)
            ),
        }

    def _run_hdbscan(self) -> None:
        print("Using HDBSCAN for density-based clustering...")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=max(1, self.min_cluster_size // 2),
            metric="euclidean",
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )
        self.labels = self.clusterer.fit_predict(self.embeddings)
        try:
            self.probabilities = self.clusterer.probabilities_
            print(
                f"Extracted membership probabilities for "
                f"{len(self.probabilities)} points"
            )
        except Exception as e:
            print(f"Warning: Could not extract probabilities: {e}")
            self.probabilities = np.zeros(len(self.labels))

    def _calculate_centroids(self) -> None:
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

    def _generate_2d_coordinates(self) -> None:
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

    def _calculate_silhouette_scores(self) -> None:
        """Calculate Silhouette Score for each sample."""
        mask = self.labels != -1
        if np.sum(mask) < 2:
            self.silhouette_scores = np.zeros(len(self.labels))
            return
        unique_labels = np.unique(self.labels[mask])
        if len(unique_labels) < 2:
            self.silhouette_scores = np.zeros(len(self.labels))
            return
        try:
            sample_scores = silhouette_samples(
                self.embeddings[mask], self.labels[mask], metric="cosine"
            )
            self.silhouette_scores = np.zeros(len(self.labels))
            self.silhouette_scores[mask] = sample_scores
        except Exception as e:
            print(f"Warning: Could not calculate silhouette scores: {e}")
            self.silhouette_scores = np.zeros(len(self.labels))

    # -- analysis ----------------------------------------------------------
    def get_detailed_analysis(self) -> DetailedAnalysis:
        """
        Generates detailed analysis including original indices, using the
        texts and index map captured by the last ``fit_predict`` call.
        """
        if self.labels is None:
            raise ValueError("Must call fit_predict first")
        if self._unique_texts is None or self._index_map is None:
            raise ValueError(
                "No texts/index_map available. Call fit_predict(texts) first."
            )

        texts = self._unique_texts
        index_map = self._index_map

        analysis: DetailedAnalysis = {}
        cluster_indices_map: Dict[int, List[int]] = {}
        for i, label in enumerate(self.labels):
            if label not in cluster_indices_map:
                cluster_indices_map[label] = []
            cluster_indices_map[label].append(i)

        all_keys = list(cluster_indices_map.keys())
        regular_keys = [k for k in all_keys if k != -1]
        regular_keys.sort()
        sorted_keys = regular_keys + ([-1] if -1 in all_keys else [])

        for cluster_id in sorted_keys:
            local_indices = cluster_indices_map[cluster_id]
            cluster_texts = [texts[i] for i in local_indices]
            cluster_original_indices = []
            for i in local_indices:
                cluster_original_indices.extend(index_map[texts[i]])
            lengths = [len(t.split()) for t in cluster_texts]

            if self.probabilities is not None:
                cluster_probs = [self.probabilities[i] for i in local_indices]
            else:
                cluster_probs = [0.0] * len(cluster_texts)

            if cluster_id in self.centroids:
                centroid = self.centroids[cluster_id]
                cluster_embeddings = self.embeddings[local_indices]
                sims = cosine_similarity(
                    cluster_embeddings, centroid.reshape(1, -1)
                ).flatten()
                cluster_distances = [float(1 - s) for s in sims]
            else:
                cluster_distances = [0.0] * len(cluster_texts)

            cluster_coords = []
            if self.coordinates_2d is not None:
                cluster_coords = [
                    self.coordinates_2d[i].tolist() for i in local_indices
                ]

            cluster_silhouettes = []
            if self.silhouette_scores is not None:
                cluster_silhouettes = [
                    float(self.silhouette_scores[i]) for i in local_indices
                ]

            combined = list(
                zip(
                    cluster_texts,
                    cluster_original_indices,
                    cluster_probs,
                    cluster_distances,
                    cluster_coords,
                    cluster_silhouettes,
                )
            )
            combined.sort(key=lambda x: (x[5], x[2]), reverse=True)
            (
                sorted_texts,
                sorted_orig_idxs,
                sorted_probs,
                sorted_dists,
                sorted_coords,
                sorted_silhouettes,
            ) = zip(*combined) if combined else ([], [], [], [], [], [])

            label_name = (
                "Noise/Outliers" if cluster_id == -1 else f"Cluster {cluster_id}"
            )

            item: ClusterAnalysisItem = {
                "count": len(cluster_texts),
                "original_indices": list(sorted_orig_idxs),
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
                item["centroid_vector_sample"] = self.centroids[cluster_id][:5].tolist()
                item["consistency_score"] = self.cluster_consistency.get(
                    cluster_id, 0.0
                )
            analysis[label_name] = item

        return analysis

    def get_summary(self, analysis: Optional[DetailedAnalysis] = None) -> SummaryData:
        """
        Build the summary report from a detailed analysis. If ``analysis`` is
        omitted, it is recomputed via ``get_detailed_analysis``.
        """
        if analysis is None:
            analysis = self.get_detailed_analysis()

        n_clusters = len(self.centroids) if self.centroids else 0
        n_noise = int(np.sum(self.labels == -1)) if self.labels is not None else 0

        cluster_details: Dict[str, ClusterDetailSummary] = {}
        total_misplaced = 0
        for cluster_name, info in analysis.items():
            is_noise = cluster_name == "Noise/Outliers"
            sil_scores = info.get("silhouette_scores", [])
            misplaced_count = sum(1 for s in sil_scores if s < 0)
            avg_silhouette = float(np.mean(sil_scores)) if sil_scores else 0.0
            total_misplaced += misplaced_count
            cluster_details[cluster_name] = {
                "count": info["count"],
                "consistency_score": info.get("consistency_score", 0.0),
                "avg_silhouette_score": round(avg_silhouette, 4),
                "misplaced_items_count": misplaced_count,
                "is_noise": is_noise,
            }

        n_original = self._n_original
        if n_original is None:
            n_original = (
                sum(len(v) for v in self._index_map.values())
                if self._index_map is not None
                else (len(self._unique_texts) if self._unique_texts else 0)
            )
        n_unique = len(self._unique_texts) if self._unique_texts else n_original

        return {
            "summary": {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "total_texts": n_original,
                "unique_texts_processed": n_unique,
                "total_misplaced_items": total_misplaced,
            },
            "cluster_details": cluster_details,
        }

    # -- convenience -------------------------------------------------------
    def cluster(
        self, texts: List[str], *, deduplicate: bool = True
    ) -> Tuple[FitPredictResult, DetailedAnalysis]:
        """One-shot: fit_predict + get_detailed_analysis."""
        results = self.fit_predict(texts, deduplicate=deduplicate)
        analysis = self.get_detailed_analysis()
        return results, analysis


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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
        help="Directory to save results.",
    )
    parser.add_argument(
        "-s",
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size for HDBSCAN.",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=0.25,
        help="Cluster selection epsilon (lower=strict, higher=loose).",
    )
    parser.add_argument(
        "-m",
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
            original_texts = json.load(f)
        if not isinstance(original_texts, list) or not all(
            isinstance(t, str) for t in original_texts
        ):
            raise ValueError("JSON root must be an array of strings.")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    print("=" * 60)
    print("Mixed-Length Text Clustering Demo (Llama.cpp)")
    print("=" * 60)

    clusterer = TextsClusterer(
        model_name=args.model,
        min_cluster_size=args.min_cluster_size,
        cluster_selection_epsilon=args.epsilon,
    )

    results, analysis = clusterer.cluster(original_texts)
    print(
        f"Input: {results['n_original_texts']} texts -> "
        f"Unique: {results['n_unique_texts']} texts"
    )

    print("\n" + "=" * 60)
    print("DETAILED CLUSTER ANALYSIS:")
    print("=" * 60)
    print_cluster_analysis(analysis)

    summary = clusterer.get_summary(analysis)
    saved = save_cluster_results(
        out_path,
        analysis,
        summary,
        coordinates_2d=clusterer.coordinates_2d,
        labels=clusterer.labels,
        texts=clusterer._unique_texts or [],
    )
    print(f"\nResults saved to {out_path}")
    for name, path in saved.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
