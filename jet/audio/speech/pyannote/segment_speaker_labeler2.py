from __future__ import annotations

from typing import Dict, List, Optional, TypedDict, Literal
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


# ---------------------------
# Types
# ---------------------------

SpeakerId = str
AssignmentStrategy = Literal["centroid", "max"]


class SegmentResult(TypedDict):
    start: float
    end: float
    speaker_label: SpeakerId
    confidence: float


@dataclass
class SpeakerModel:
    speaker_id: SpeakerId
    centroid: np.ndarray
    embeddings: np.ndarray
    count: int


# ---------------------------
# Utilities
# ---------------------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---------------------------
# Labeler
# ---------------------------

class SegmentSpeakerLabeler:
    """
    Persistent speaker labeler.

    When use_references=True, the labeler maintains speaker identity
    across multiple calls to `cluster_segments`.
    """

    def __init__(
        self,
        *,
        use_references: bool = False,
        assignment_strategy: AssignmentStrategy = "centroid",
        similarity_threshold: float = 0.75,
    ) -> None:
        self.use_references = use_references
        self.assignment_strategy = assignment_strategy
        self.similarity_threshold = similarity_threshold

        # Persistent speaker registry
        self._speakers: Dict[SpeakerId, SpeakerModel] = {}
        self._speaker_counter = 0

    # ---------------------------
    # Public API
    # ---------------------------

    def cluster_segments(
        self,
        segments: List[dict],
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> List[SegmentResult]:
        """
        Cluster segments and assign persistent speaker labels.
        """

        embeddings = l2_normalize(embeddings)

        # group embeddings by cluster
        clusters: Dict[int, np.ndarray] = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_labels):
            clusters[int(cluster_id)].append(embeddings[idx])

        clusters = {
            cid: np.vstack(embs)
            for cid, embs in clusters.items()
        }

        cluster_to_speaker: Dict[int, SpeakerId] = {}

        for cluster_id, cluster_embs in clusters.items():
            speaker_id, confidence = self._assign_cluster(cluster_embs)
            cluster_to_speaker[cluster_id] = speaker_id

            if self.use_references:
                self._update_speaker(speaker_id, cluster_embs)

        # build results
        results: List[SegmentResult] = []
        for idx, segment in enumerate(segments):
            cid = int(cluster_labels[idx])
            sid = cluster_to_speaker[cid]

            results.append(
                SegmentResult(
                    start=segment["start"],
                    end=segment["end"],
                    speaker_label=sid,
                    confidence=1.0,
                )
            )

        return results

    # ---------------------------
    # Assignment
    # ---------------------------

    def _assign_cluster(
        self,
        cluster_embeddings: np.ndarray,
    ) -> tuple[SpeakerId, float]:
        """
        Assign a cluster to an existing speaker or create a new one.
        """

        if not self.use_references or not self._speakers:
            return self._create_new_speaker(cluster_embeddings), 1.0

        cluster_centroid = l2_normalize(cluster_embeddings.mean(axis=0))

        best_speaker: Optional[SpeakerId] = None
        best_score = -1.0

        for speaker_id, model in self._speakers.items():
            score = self._compute_similarity(cluster_embeddings, cluster_centroid, model)
            if score > best_score:
                best_score = score
                best_speaker = speaker_id

        if best_score >= self.similarity_threshold:
            return best_speaker, best_score  # type: ignore

        return self._create_new_speaker(cluster_embeddings), 1.0

    def _compute_similarity(
        self,
        cluster_embeddings: np.ndarray,
        cluster_centroid: np.ndarray,
        model: SpeakerModel,
    ) -> float:
        """
        Compute similarity between a cluster and a speaker model.
        """

        if self.assignment_strategy == "centroid":
            return cosine_similarity(cluster_centroid, model.centroid)

        if self.assignment_strategy == "max":
            sims = cluster_embeddings @ model.embeddings.T
            return float(np.max(sims))

        raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")

    # ---------------------------
    # Speaker lifecycle
    # ---------------------------

    def _create_new_speaker(self, cluster_embeddings: np.ndarray) -> SpeakerId:
        speaker_id = f"speaker_{self._speaker_counter}"
        self._speaker_counter += 1

        centroid = l2_normalize(cluster_embeddings.mean(axis=0))
        self._speakers[speaker_id] = SpeakerModel(
            speaker_id=speaker_id,
            centroid=centroid,
            embeddings=cluster_embeddings.copy(),
            count=len(cluster_embeddings),
        )

        return speaker_id

    def _update_speaker(self, speaker_id: SpeakerId, new_embeddings: np.ndarray) -> None:
        model = self._speakers[speaker_id]

        all_embeddings = np.vstack([model.embeddings, new_embeddings])
        centroid = l2_normalize(all_embeddings.mean(axis=0))

        model.embeddings = all_embeddings
        model.centroid = centroid
        model.count = all_embeddings.shape[0]
