# jet_python_modules/jet/audio/speech/pyannote/segment_speaker_labeler.py
from __future__ import annotations

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
    min_cosine_similarity: float


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
        use_gpu: bool = True,
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
        use_gpu : bool, optional
            Use CUDA if available.
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = Model.from_pretrained(embedding_model_name, use_auth_token=hf_token)
        # Use sliding-window inference (3s duration, 0.5s step) – this is the recommended
        # way for pyannote/embedding and gracefully handles very short segments
        # (prevents InstanceNorm1d crash when waveform collapses to 1 time step).
        self.inference = Inference(
            model=self.model,
            duration=3.0,      # seconds
            step=0.5,          # seconds – controls overlap, smaller = more robust
            window="sliding",  # explicit for clarity
        )
        self.inference.to(self.device)  # move inference pipeline to GPU/CPU
        
        self.distance_threshold = distance_threshold
        self.clustering_strategy = clustering_strategy
        self.n_clusters = n_clusters

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
            List of dictionaries with keys: "path", "parent_dir", "speaker_label", "min_cosine_similarity".
        """
        if not segment_paths:
            raise ValueError("No segment file paths were provided to cluster_segments.")

        print(f"Found {len(segment_paths)} segments. Extracting embeddings...")
        embeddings = self._extract_embeddings(segment_paths)  # shape: (n_segments, dim), already L2-normalized

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
            # Explicitly pass min_clusters and max_clusters to avoid None comparisons
            labels = clusterer.cluster(
                embeddings,
                min_clusters=1,
                max_clusters=9999,  # Large number effectively disables upper bound
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
        # Compute centroids manually (consistent with pyannote pipelines)
        cluster_centroids = np.stack([embeddings[labels == l].mean(axis=0) for l in unique_labels])

        # Remap labels to 0-based contiguous integers (in case pyannote returns non-contiguous)
        label_to_new = {old: new for new, old in enumerate(unique_labels)}
        contiguous_labels = np.array([label_to_new[l] for l in labels])

        results: List[SegmentResult] = []
        for path, label, emb in zip(segment_paths, contiguous_labels, embeddings):
            centroid = cluster_centroids[label]
            similarity = float(np.dot(emb, centroid))  # cosine similarity (both normalized)

            path_obj = Path(path)
            results.append(
                {
                    "path": str(path_obj),
                    "parent_dir": path_obj.parent.name,
                    "speaker_label": int(label),
                    "min_cosine_similarity": similarity,
                }
            )

        print(f"Clustering complete → {len(unique_labels)} speakers detected.")
        return results