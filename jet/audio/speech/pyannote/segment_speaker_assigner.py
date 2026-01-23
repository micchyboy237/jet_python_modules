# online_speaker_assigner.py
"""
Incremental / online speaker assignment for live segments using pyannote embeddings.

Compares incoming speech segments against known speaker centroids.
If similar enough → assign known speaker
If not → create new speaker ID and store its centroid

Supports:
- Pre-loaded reference speakers (optional)
- Learning new speakers on-the-fly
- EMA centroid update when same speaker appears again
- Multiple input formats (file, bytes, numpy array)
"""

from __future__ import annotations

from typing import Optional, Set, Tuple, TypedDict

import torch

from jet.audio.speech.pyannote.segment_speaker_labeler import AudioInput, SegmentSpeakerLabeler, SegmentResult


class AssignmentMeta(TypedDict):
    """Metadata returned for each processed segment — matches original return shape"""
    is_same_speaker: bool
    """Whether this segment was assigned the same speaker as the immediately previous segment"""
    was_new_speaker: bool
    prev_speaker_label: Optional[int | str]
    """Speaker label of the previous segment (None on first segment)"""
    label: int | str
    segment_num: int


class SegmentSpeakerAssigner:
    """
    Assigns speaker labels to live audio utterance segments incrementally.

    Main entry point: .add_segment(audio, ...)

    Features:
    - Online assignment (no need to wait for all segments)
    - Cosine similarity with L2-normalized embeddings
    - Optional EMA update of centroids when same speaker speaks again
    - Supports file paths, bytes/BytesIO, or numpy arrays (mono float32 @ 16kHz)
    - Rich console logging option for debugging
    """

    def __init__(
        self,
        embedding_model: str = "pyannote/embedding",
        hf_token: str | None = None,
        device: str | torch.device | None = None,
        verbose: bool = False,
    ) -> None:
        self.clusterer = SegmentSpeakerLabeler(
            embedding_model=embedding_model,
            hf_token=hf_token,
            device=device,
            verbose=verbose,
            use_references=True,
        )

        self.speaker_segments: dict[int | str, int] = {}
        self.prev_speaker_audio: Optional[AudioInput] = None
        self.prev_speaker_label: Optional[int | str] = None   # widened type
        self.prev_segment_num: int = 0
        self.speaker_labels: Set[int] = set()

    def _add_next_speaker(self) -> int:
        """
        Adds the next available speaker index starting from 0
        and returns the newly assigned index.
        """
        if not self.speaker_labels:
            next_id = 0
        else:
            next_id = max(self.speaker_labels) + 1

        self.speaker_labels.add(next_id)
        return next_id

    def add_segment(
        self,
        audio: AudioInput,
    ) -> Tuple[SegmentResult, AssignmentMeta]:
        """
        Assign speaker to one speech segment.

        Args:
            audio: waveform (np.float32 mono), file path, or bytes/BytesIO (wav)

        Returns:
            Tuple[SegmentResult, AssignmentMeta]
        """
        if self.prev_speaker_audio:
            results = self.clusterer.cluster_segments([self.prev_speaker_audio, audio])
            segment_result = results[-1]
        else:
            results = self.clusterer.cluster_segments(audio)
            segment_result = results[0]   # was missing indexing for first segment

        centroid_sim = segment_result["centroid_cosine_similarity"]
        should_create_new_speaker_label = centroid_sim >= 1.0

        if should_create_new_speaker_label:
            self._add_next_speaker()

        new_speaker_label = segment_result["speaker_label"]
        new_segment_num = self.prev_segment_num + 1
        is_same_speaker = (new_speaker_label == self.prev_speaker_label)


        was_new_speaker = new_speaker_label not in self.speaker_segments

        if new_speaker_label in self.speaker_segments:
            self.speaker_segments[new_speaker_label] += 1
        else:
            self.speaker_segments[new_speaker_label] = 1

        self.prev_speaker_audio = audio
        self.prev_speaker_label = new_speaker_label
        self.prev_segment_num = new_segment_num

        meta: AssignmentMeta = {
            "is_same_speaker": is_same_speaker,
            "was_new_speaker": was_new_speaker,
            "prev_speaker_label": self.prev_speaker_label,
            "label": new_speaker_label,
            "segment_num": new_segment_num,
        }

        return segment_result, meta

    def get_speaker_counts(self) -> dict[int | str, int]:
        """Return how many segments each speaker has contributed"""
        return dict(self.speaker_segments)

    def reset(self) -> None:
        """Forget all learned speakers (e.g. new call/session)"""
        self.speaker_segments.clear()
        self.prev_speaker_audio = None
        self.prev_speaker_label = None
        self.prev_segment_num = 0

# Quick smoke test / usage example (comment out in production)
if __name__ == "__main__":
    assigner = SegmentSpeakerAssigner()

    # Simulate segments (replace with real websocket audio)
    print("Simulating first segment...")
    assigner.add_segment("sample_speaker_A_001.wav")  # ← replace with real path

    print("\nSimulating second segment (same speaker)...")
    assigner.add_segment("sample_speaker_A_002.wav")

    print("\nSimulating third segment (different speaker)...")
    assigner.add_segment("sample_speaker_B_001.wav")

    print("\nCurrent speaker counts:", assigner.get_speaker_counts())
