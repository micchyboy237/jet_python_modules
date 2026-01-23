from typing import Dict, Optional, List, Tuple, Union
import numpy as np
from pyannote.audio import Model, Inference
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
import logging

# New imports for clustering mode
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # sklearn sometimes warns about convergence

# Setup rich logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("speaker_id")
console = Console()

class SpeakerIdentifier:
    """
    Reusable class for speaker identification using pyannote/embedding.
    
    - Stores reference embeddings (averaged per speaker)
    - Compares new audio segments via cosine similarity
    - Supports open-set identification with 'UNKNOWN' fallback
    """

    def __init__(
        self,
        model_name: str = "pyannote/embedding",
        similarity_threshold: float = 0.7,   # Tune: 0.7-0.8 common starting point
        unknown_threshold: float = 0.6,      # Below this → definitely unknown
        min_duration: float = 1.0,            # seconds, skip very short segments (lowered from 2.0)
        inference_duration: float = 5.0,      # NEW: configurable inference window duration
        inference_step: float = 0.5,          # NEW: configurable inference step size
        cluster_when_no_refs: bool = True,
        # Clustering parameters (tunable)
        max_speakers: Optional[int] = None,           # None = auto-detect
        distance_threshold: float = 0.35,             # ~1 - similarity_threshold
    ):
        """
        Args:
            model_name: Hugging Face model (needs acceptance + token if gated)
            similarity_threshold: Min cosine sim for 'same speaker' match
            unknown_threshold: Max sim to be considered unknown (below this)
            min_duration: Minimum audio duration to process (seconds)
            inference_duration: Duration of each embedding window (seconds)
            inference_step: Step size for sliding window (seconds)
            cluster_when_no_refs: If True, fall back to clustering when no reference speakers are enrolled.
            max_speakers: For clustering, maximum number of speakers to segment (if known; else None).
            distance_threshold: Clustering stopping criterion (higher=more clusters, lower=fewer).
        """
        self.model = Model.from_pretrained(model_name)
        self.inference = Inference(self.model, duration=inference_duration, step=inference_step)
        self.references: Dict[str, np.ndarray] = {}  # speaker_label -> averaged embedding
        self.similarity_threshold = similarity_threshold
        self.unknown_threshold = unknown_threshold
        self.min_duration = min_duration
        self.cluster_when_no_refs = cluster_when_no_refs
        self.max_speakers = max_speakers
        self.distance_threshold = distance_threshold

    def _extract_embedding(self, audio_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Extract averaged embedding from an audio file."""
        path = Path(audio_path)
        if not path.is_file():
            logger.error(f"File not found: {path}")
            return None
        
        try:
            emb = self.inference(str(path))
            # Handle SlidingWindowFeature or direct ndarray
            if hasattr(emb, "data"):
                emb = emb.data
            # Allow single window for very short clips
            if emb.shape[0] == 0:
                logger.warning(f"Audio is too short even for one window: {path.name}")
                return None
            avg_emb = np.mean(emb, axis=0)
            norm = np.linalg.norm(avg_emb)
            if norm > 0:
                avg_emb /= norm
            return avg_emb
        except Exception as e:
            logger.exception(f"Failed to extract embedding from {path.name}: {e}")
            return None

    def add_reference(
        self,
        speaker_label: str,
        audio_paths: List[Union[str, Path]],
        force_overwrite: bool = False
    ) -> bool:
        """
        Add or update reference embedding for a speaker by averaging multiple files.
        
        Returns True if successful.
        """
        if speaker_label in self.references and not force_overwrite:
            logger.warning(f"Speaker '{speaker_label}' already exists. Use force_overwrite=True.")
            return False

        embeddings: List[np.ndarray] = []
        for path in audio_paths:
            emb = self._extract_embedding(path)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            logger.error(f"No valid embeddings extracted for '{speaker_label}'")
            return False

        # Average all reference embeddings
        avg_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb /= norm

        self.references[speaker_label] = avg_emb
        logger.info(f"Added/updated reference for '{speaker_label}' ({len(embeddings)} files)")
        return True

    def identify(
        self,
        audio_path: Union[str, Path],
        # New optional param (can override init setting)
        cluster_if_no_refs: Optional[bool] = None,
    ) -> Tuple[str, float]:
        """
        Identify speaker in a new audio file.

        Returns (label, max_similarity_or_confidence)
        - With references: label is speaker name or 'UNKNOWN'/'UNCERTAIN_XXX'
        - Without references + clustering: label is 'SPEAKER_XX'
        """
        emb = self._extract_embedding(audio_path)
        if emb is None:
            return "ERROR", 0.0

        do_clustering = (
            cluster_if_no_refs if cluster_if_no_refs is not None
            else self.cluster_when_no_refs
        )

        if not self.references:
            if not do_clustering:
                logger.warning("No reference speakers enrolled and clustering disabled → returning UNKNOWN")
                return "UNKNOWN", 0.0

            # Clustering mode: treat whole file as multi-speaker
            return self._identify_with_clustering(audio_path)

        # Existing closed-set identification (unchanged)
        similarities = {}
        for label, ref_emb in self.references.items():
            # Since both normalized → sim = dot product
            sim = float(np.dot(emb, ref_emb))
            similarities[label] = sim

        best_label = max(similarities, key=similarities.get)
        best_sim = similarities[best_label]

        if best_sim >= self.similarity_threshold:
            return best_label, best_sim
        elif best_sim >= self.unknown_threshold:
            return f"UNCERTAIN_{best_label}", best_sim
        else:
            return "UNKNOWN", best_sim

    def _identify_with_clustering(self, audio_path: Union[str, Path]) -> Tuple[str, float]:
        """Fallback: cluster embeddings from the whole file and assign pseudo-labels"""
        path = str(Path(audio_path))
        try:
            # Get embeddings over sliding windows
            emb_array = self.inference(path)  # shape: (num_windows, embedding_dim)
            if hasattr(emb_array, 'data'):
                arr = emb_array.data
            else:
                arr = emb_array
            if arr.shape[0] <= 1:
                logger.debug("Single embedding window → treating as single speaker (normal for short clips)")
                return "SPEAKER_00", 1.0

            # only try real clustering when we have 2+ windows
            if arr.shape[0] < 3:
                logger.info(
                    f"Only {arr.shape[0]} windows — limited clustering possible, "
                    "but proceeding anyway (may produce 1 cluster)"
                )

            # Normalize just in case
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            arr = arr / norms

            # Clustering
            if self.max_speakers is not None:
                clustering = AgglomerativeClustering(
                    n_clusters=self.max_speakers,
                    metric="cosine",
                    linkage="average"
                )
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=self.distance_threshold,
                    metric="cosine",
                    linkage="average"
                )

            labels = clustering.fit_predict(arr)

            # Most common label → dominant speaker of the file
            unique, counts = np.unique(labels, return_counts=True)
            dominant_idx = np.argmax(counts)
            dominant_label_id = unique[dominant_idx]

            # Format label
            label = f"SPEAKER_{dominant_label_id:02d}"

            # Rough confidence: proportion of dominant cluster
            confidence = counts[dominant_idx] / len(labels)

            logger.info(f"Clustered file into {len(unique)} speakers → dominant: {label}")
            return label, confidence

        except Exception as e:
            logger.exception(f"Clustering failed for {audio_path}: {e}")
            return "UNKNOWN", 0.0

    def print_references(self) -> None:
        """Pretty-print enrolled speakers."""
        if not self.references:
            console.print("[yellow]No reference speakers enrolled yet.[/yellow]")
            return

        table = Table(title="Enrolled Speakers")
        table.add_column("Label", style="cyan")
        table.add_column("Embedding Dim", justify="right")
        for label, emb in self.references.items():
            table.add_row(label, str(emb.shape))
        console.print(table)
