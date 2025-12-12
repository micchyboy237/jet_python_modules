"""
Reusable speaker embedding & similarity utilities using pyannote.audio.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Literal
import numpy as np
from scipy.spatial.distance import cdist
from pyannote.audio import Model, Inference
from pyannote.core import Segment


class EmbeddingResult(TypedDict):
    """Represents a single embedding vector."""
    vector: np.ndarray


class SpeakerEmbedding:
    """
    Loads a pyannote speaker embedding model and provides utilities
    for extracting embeddings and computing similarity.
    """

    def __init__(self, model_id: str, token: Optional[str] = None):
        """
        Initialize the model.

        Args:
            model_id: HuggingFace model identifier.
            token: HF token used for authentication.
        """
        self.model = Model.from_pretrained(model_id, use_auth_token=token)

    def create_inference(
        self,
        window: Literal["whole", "sliding"] = "whole",
        duration: Optional[float] = None,
        step: Optional[float] = None,
    ) -> Inference:
        """
        Create an inference pipeline.

        Args:
            window: "whole" or "sliding".
            duration: Duration for sliding window.
            step: Step size for sliding window.
        """
        return Inference(
            self.model,
            window=window,
            duration=duration,
            step=step,
        )

    def embed_file(
        self,
        inference: Inference,
        file_path: str,
    ) -> EmbeddingResult:
        """
        Extract embedding for the whole audio file.
        """
        vec = inference(file_path)

        # Ensure shape is (1, D)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        return EmbeddingResult(vector=vec)

    def embed_segment(
        self,
        inference: Inference,
        file_path: str,
        segment: Segment,
    ) -> EmbeddingResult:
        """
        Extract embedding for a portion of the audio file.
        """
        vec = inference.crop(file_path, segment)

        # Ensure shape is (1, D)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        return EmbeddingResult(vector=vec)

    def cosine_distance(
        self,
        e1: EmbeddingResult,
        e2: EmbeddingResult,
    ) -> float:
        """
        Compute cosine distance between two embeddings.
        """
        v1 = e1["vector"]
        v2 = e2["vector"]
        return float(cdist(v1, v2, metric="cosine")[0, 0])

    # --- New private helpers ---

    def _ensure_2d(self, vec: np.ndarray) -> np.ndarray:
        return vec.reshape(1, -1) if vec.ndim == 1 else vec

    def _get_inference(self) -> Inference:
        return Inference(self.model, window="whole")

    def _embed_via_inference(
        self,
        file_path: str,
        inference: Inference,
        start: float | None,
        end: float | None,
    ) -> np.ndarray:
        if start is not None and end is not None:
            vec = inference.crop(file_path, Segment(start, end))
        else:
            vec = inference(file_path)
        return self._ensure_2d(vec)

    # --- New public simpler API ---

    def embed(
        self,
        file_path: str,
        start: float | None = None,
        end: float | None = None,
    ) -> EmbeddingResult:
        """
        High-level embedding function that hides pyannote internals.
        Developers only pass a file and optional segment times.
        """
        inference = self._get_inference()
        vec = self._embed_via_inference(file_path, inference, start, end)
        return EmbeddingResult(vector=vec)

    def distance(
        self,
        file1: str,
        file2: str,
        *,
        start1: float | None = None,
        end1: float | None = None,
        start2: float | None = None,
        end2: float | None = None,
    ) -> float:
        """
        Compute cosine distance between two audio files or two audio segments,
        with all internal steps hidden.
        """
        e1 = self.embed(file1, start1, end1)
        e2 = self.embed(file2, start2, end2)
        return float(cdist(e1["vector"], e2["vector"], metric="cosine")[0, 0])

    def similarity(
        self,
        file1: str,
        file2: str,
        *,
        start1: float | None = None,
        end1: float | None = None,
        start2: float | None = None,
        end2: float | None = None,
    ) -> float:
        """
        Compute cosine similarity between two audio files or segments.
        Returns a value between -1 and 1, where 1 means identical.

        Args:
            file1: Path to the first audio file.
            file2: Path to the second audio file.
            start1, end1: Optional start and end times for the first file.
            start2, end2: Optional start and end times for the second file.
        """
        dist = self.distance(file1, file2, start1=start1, end1=end1, start2=start2, end2=end2)
        return 1.0 - dist
