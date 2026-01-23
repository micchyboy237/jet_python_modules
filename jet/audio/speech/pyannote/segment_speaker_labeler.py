# segment_speaker_labeler.py
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import List, Literal, TypedDict

import numpy as np
import torch
from pyannote.audio import Inference, Model
from pyannote.audio.pipelines.clustering import AgglomerativeClustering as PyannoteAgglomerativeClustering
from pyannote.audio.pipelines.clustering import KMeansClustering as PyannoteKMeansClustering
from tqdm import tqdm


class SegmentResult(TypedDict):
    path: str
    parent_dir: str
    speaker_label: int
    centroid_cosine_similarity: float
    nearest_neighbor_cosine_similarity: float


class SegmentSpeakerLabeler:
    """
    A reusable class for clustering short speech segments using pyannote speaker embeddings
    and pyannote's clustering implementations (Agglomerative or KMeans).

    Designed for cases where each segment is assumed to contain a single speaker
    (e.g., extracted speech clips named 'sound.wav' in subdirectories).

    Features:
    - Configurable embedding model and clustering strategy
    - Progress bars via tqdm
    - Normalized embeddings for cosine similarity
    - Returns structured results with speaker labels and min similarity to centroid
    - Generic and reusable – no hardcoded paths or business logic
    """

    def __init__(
        self,
        embedding_model_name: str = "pyannote/embedding",
        hf_token: str | None = None,
        distance_threshold: float = 0.7,
        clustering_strategy: Literal["agglomerative", "kmeans"] = "agglomerative",
        n_clusters: int | None = None,
        # ── New parameters for reference-based assignment ─────────────────────
        reference_embeddings_by_speaker: dict[int | str, np.ndarray] | None = None,
        reference_paths_by_speaker: dict[int | str, list[str | Path]] | None = None,
        assignment_threshold: float = 0.68,
        assignment_strategy: Literal["centroid", "max"] = "centroid",
        # ──────────────────────────────────────────────────────────────────────
        use_accelerator: bool = True,  # renamed for clarity (MPS/CUDA)
    ) -> None:
        """
        Initialize the clusterer.

        Parameters
        ----------
        embedding_model_name : str, optional
            Hugging Face model name for speaker embedding.
        hf_token : str | None, optional
            Hugging Face authentication token (required for gated models).
        distance_threshold : float, optional
            Distance threshold for agglomerative clustering (lower → more clusters).
        clustering_strategy : Literal["agglomerative", "kmeans"], optional
            Clustering algorithm to use. Defaults to "agglomerative".
        n_clusters : int | None, optional
            Required when using "kmeans". Ignored for "agglomerative".
        use_accelerator : bool, optional
            Use MPS (Apple), CUDA or CPU as available.
        """

        # ── Prefer MPS on Apple Silicon, then CUDA, then CPU ─────────────────────
        if torch.backends.mps.is_available():
            device_str = "mps"
        elif use_accelerator and torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"

        self.device = torch.device(device_str)

        # Optional: log once at init so you know what's being used
        print(f"Using device: {self.device} ({'MPS acceleration' if device_str == 'mps' else 'CUDA' if device_str == 'cuda' else 'CPU'})")

        # ── Suppress safetensors "open file:" prints during model load ─────────
        @contextlib.contextmanager
        def suppress_safetensors_output():
            with contextlib.redirect_stdout(io.StringIO()):
                yield

        with suppress_safetensors_output():
            self.model = Model.from_pretrained(
                embedding_model_name,
                use_auth_token=hf_token,
                map_location=self.device,   # ← load directly to target device
                strict=False,
            )

        with suppress_safetensors_output():
            self.inference = Inference(
                model=self.model,
                duration=3.0,      # seconds
                step=0.5,          # seconds – controls overlap, smaller = more robust
                window="sliding",  # explicit for clarity
            )

        # No need for .to() anymore — model already on correct device
        # But keep this line harmless if Inference needs explicit move in future
        self.inference.to(self.device)

        self.reference_embeddings_by_speaker = reference_embeddings_by_speaker or {}
        self.reference_centroids: dict[int | str, np.ndarray] = {}
        self.assignment_threshold = assignment_threshold
        self.assignment_strategy = assignment_strategy

        if reference_paths_by_speaker:
            self._load_references_from_paths(reference_paths_by_speaker)

        self.distance_threshold = distance_threshold
        self.clustering_strategy = clustering_strategy
        self.n_clusters = n_clusters

    def _load_references_from_paths(
        self,
        reference_paths_by_speaker: dict[int | str, list[str | Path]],
    ) -> None:
        """Pre-compute embeddings and centroids from reference audio paths."""
        for speaker_id, paths in reference_paths_by_speaker.items():
            path_objs = [Path(p) for p in paths]
            embs = self._extract_embeddings(path_objs)
            if len(embs) == 0:
                continue
            if self.assignment_strategy == "centroid":
                centroid = embs.mean(axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-12
                self.reference_centroids[speaker_id] = centroid
            else:
                # For "max" strategy: store all embeddings
                self.reference_embeddings_by_speaker[speaker_id] = embs

    def _assign_to_references(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign each embedding to the closest reference speaker or new label."""
        labels = np.full(len(embeddings), -1, dtype=int)
        next_new_label = max(self.reference_centroids.keys(), default=-1)
        if isinstance(next_new_label, (str, np.generic)):
            # Only int ids can serve as a starting max for new labels.
            next_new_label = -1
        next_new_label = next_new_label + 1 if next_new_label >= 0 else 0

        for i, emb in enumerate(embeddings):
            best_sim = -1.0
            best_speaker = -1

            for spk_id, ref in self.reference_centroids.items():
                sim = float(np.dot(emb, ref))
                if sim > best_sim:
                    best_sim = sim
                    best_speaker = spk_id

            if best_sim >= self.assignment_threshold:
                labels[i] = best_speaker
            else:
                labels[i] = next_new_label
                next_new_label += 1

        return labels

    def _extract_embeddings(self, segment_paths: List[Path]) -> np.ndarray:
        """Extract and L2-normalize speaker embeddings for all segments with progress bar."""
        embeddings: List[np.ndarray] = []
        for path in tqdm(segment_paths, desc="Extracting embeddings"):
            result = self.inference(str(path))
            # When using sliding window, result is SlidingWindowFeature
            if hasattr(result, "data"):
                emb_array = result.data  # shape: (n_windows, dim)
                if emb_array.shape[0] == 0:
                    raise ValueError(f"No windows extracted for very short segment: {path}")
                # Average-pool over sliding windows to get one embedding per segment
                emb = np.mean(emb_array, axis=0)
            else:
                # Fallback for whole-window mode (numpy array)
                emb = result
                if emb.ndim == 2:
                    emb = emb.squeeze(0)
                elif emb.ndim > 2:
                    raise ValueError(f"Unexpected embedding shape {emb.shape} for {path}")

            # L2 normalize
            norm = np.linalg.norm(emb)
            if norm == 0:
                raise ValueError(f"Zero-norm embedding for {path} – silent or invalid audio?")
            emb = emb / norm
            embeddings.append(emb)
        return np.stack(embeddings)

    def _nearest_neighbor_similarity(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute nearest-neighbor cosine similarity for each embedding.
        Assumes embeddings are L2-normalized.
        """
        sim_matrix = embeddings @ embeddings.T
        np.fill_diagonal(sim_matrix, -1.0)
        return sim_matrix.max(axis=1)

    def cluster_segments(
        self,
        segment_paths: List[Path] | List[str],
    ) -> List[SegmentResult]:
        """
        Cluster speaker embeddings from a provided list of segment file paths.

        Parameters
        ----------
        segment_paths : List[Path] | List[str]
            List of paths to segment audio files to be clustered.

        Returns
        -------
        List[SegmentResult]
            List of dictionaries with keys: "path", "parent_dir", "speaker_label", "centroid_cosine_similarity", "nearest_neighbor_cosine_similarity".
        """
        if not segment_paths:
            raise ValueError("No segment file paths were provided to cluster_segments.")

        use_references = bool(self.reference_centroids or self.reference_embeddings_by_speaker)

        if use_references:
            print(f"Found {len(segment_paths)} segments. Assigning using {len(self.reference_centroids)} reference speakers...")
        else:
            print(f"Found {len(segment_paths)} segments. Extracting embeddings...")

        embeddings = self._extract_embeddings(segment_paths)

        if use_references:
            labels = self._assign_to_references(embeddings)
            unique_labels = np.unique(labels)
            print(f"Assignment complete → {len(unique_labels)} speakers detected (including possible new).")
        else:
            print("Clustering embeddings...")
            if self.clustering_strategy == "agglomerative":
                if self.n_clusters is not None:
                    raise ValueError("n_clusters cannot be used with agglomerative strategy.")
                clusterer = PyannoteAgglomerativeClustering(
                    metric="cosine",
                ).instantiate({
                    "threshold": self.distance_threshold,
                    "method": "average",
                    "min_cluster_size": 1,
                })
                labels = clusterer.cluster(
                    embeddings,
                    min_clusters=1,
                    max_clusters=9999,
                )

            elif self.clustering_strategy == "kmeans":
                if self.n_clusters is None:
                    raise ValueError("n_clusters must be provided for kmeans strategy.")
                clusterer = PyannoteKMeansClustering(metric="cosine").instantiate({})
                labels = clusterer.cluster(
                    embeddings,
                    num_clusters=self.n_clusters,
                )

            else:
                raise ValueError(f"Unsupported clustering_strategy: {self.clustering_strategy}")

            unique_labels = np.unique(labels)
            # (existing majority remapping code remains here)
            cluster_sizes = [(l, np.sum(labels == l)) for l in unique_labels]
            cluster_sizes.sort(key=lambda x: -x[1])
            old_label_to_priority = {old: idx for idx, (old, _) in enumerate(cluster_sizes)}
            labels = np.array([old_label_to_priority[l] for l in labels])  # overwrite with remapped
            unique_labels = np.unique(labels)

        # ── The rest stays almost the same ──
        # Compute centroids (needed for output even in reference mode)
        cluster_centroids = []
        for l in np.unique(labels):
            if np.sum(labels == l) == 0:
                continue
            centroid = embeddings[labels == l].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            cluster_centroids.append(centroid)
        cluster_centroids = np.stack(cluster_centroids) if len(cluster_centroids) > 0 else np.array([])

        # nearest neighbor sim (same logic)
        nearest_neighbor_sim = np.zeros(len(embeddings), dtype=np.float32)
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            if len(idx) <= 1:
                nearest_neighbor_sim[idx] = 1.0
                continue
            cluster_embs = embeddings[idx]
            nn_sim = self._nearest_neighbor_similarity(cluster_embs)
            nearest_neighbor_sim[idx] = nn_sim

        results: List[SegmentResult] = []
        centroid_map = {i: c for i, c in enumerate(cluster_centroids)}

        for i, (path, label) in enumerate(zip(segment_paths, labels)):
            path_obj = Path(path)
            centroid_sim = float(np.dot(embeddings[i], centroid_map.get(int(label), np.zeros_like(embeddings[i]))))

            results.append(
                {
                    "path": str(path_obj),
                    "parent_dir": path_obj.parent.name,
                    "speaker_label": int(label),
                    "centroid_cosine_similarity": centroid_sim,
                    "nearest_neighbor_cosine_similarity": float(nearest_neighbor_sim[i]),
                }
            )

        print(f"Processing complete → {len(np.unique(labels))} speakers detected.")
        return results