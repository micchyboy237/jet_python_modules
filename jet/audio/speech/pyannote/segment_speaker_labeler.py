from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from pyannote.audio import Inference, Model
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class SegmentSpeakerLabeler:
    """
    A reusable class for clustering short speech segments using pyannote speaker embeddings
    and agglomerative clustering with cosine distance.

    Designed for cases where each segment is assumed to contain a single speaker
    (e.g., extracted speech clips named 'sound.wav' in subdirectories).

    Features:
    - Configurable embedding model and clustering threshold
    - Progress bars via tqdm
    - Normalized embeddings for cosine similarity
    - Returns structured results with speaker labels
    - Generic and reusable – no hardcoded paths or business logic
    """

    def __init__(
        self,
        embedding_model_name: str = "pyannote/embedding",
        hf_token: str | None = None,
        distance_threshold: float = 0.7,
        use_gpu: bool = True,
    ) -> None:
        """
        Initialize the clusterer.

        Parameters
        ----------
        embedding_model_name : str, optional
            Hugging Face model name for speaker embedding, by default "pyannote/embedding".
        hf_token : str | None, optional
            Hugging Face authentication token (required for gated models).
        distance_threshold : float, optional
            Agglomerative clustering distance threshold on cosine distance.
            Lower values → more speakers/clusters.
        use_gpu : bool, optional
            Use CUDA if available, by default True.
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self.model = Model.from_pretrained(embedding_model_name, use_auth_token=hf_token)
        self.inference = Inference(self.model, window="whole")
        self.inference.to(self.device)

        self.distance_threshold = distance_threshold

    def _extract_embeddings(self, segment_paths: List[Path]) -> np.ndarray:
        """Extract and normalize speaker embeddings for all segments with progress bar."""
        embeddings: List[np.ndarray] = []

        for path in tqdm(segment_paths, desc="Extracting embeddings"):
            emb: np.ndarray = self.inference(str(path))

            # Ensure (D,) shape
            if emb.ndim == 2:
                emb = emb.squeeze(0)
            elif emb.ndim > 2:
                raise ValueError(f"Unexpected embedding shape {emb.shape} for {path}")

            # L2 normalize
            emb = emb / np.linalg.norm(emb)

            embeddings.append(emb)

        return np.stack(embeddings)

    def cluster_segments(
        self,
        segment_paths: List[Path] | List[str],
    ) -> List[Dict[str, Any]]:
        """
        Cluster speaker embeddings from a provided list of segment file paths.

        Parameters
        ----------
        segment_paths : List[Path] | List[str]
            List of paths to segment audio files to be clustered.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with keys: "path" (str), "parent_dir" (str), "speaker_label" (int).
        """

        if not segment_paths:
            raise ValueError(
                "No segment file paths were provided to cluster_segments (segment_paths is empty)."
            )

        print(f"Found {len(segment_paths)} segments. Extracting embeddings...")
        embeddings = self._extract_embeddings(segment_paths)

        print("Computing cosine distance matrix and clustering...")
        distance_matrix = 1 - cosine_similarity(embeddings)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=self.distance_threshold,
        )
        labels = clustering.fit_predict(distance_matrix)

        results: List[Dict[str, Any]] = []
        for path, label in zip(segment_paths, labels):
            path = Path(path)
            results.append(
                {
                    "path": str(path),
                    "parent_dir": path.parent.name,
                    "speaker_label": int(label),
                }
            )

        print(f"Clustering complete → {len(set(labels))} speakers detected.")
        return results
