from collections import deque
from typing import Any, List, Literal, TypedDict

import numpy as np


class SubtitleHistoryItem(TypedDict):
    """Structured history item for LLM chat completion."""

    role: Literal["user", "assistant"]
    content: str


class CircularAudioBuffer:
    """Pure sample-based circular buffer.

    Keeps only the most recent max_duration_sec of audio.
    Chunks are treated as contiguous live audio — no silence gaps are inserted.
    Extremely memory efficient and guaranteed to never exceed the limit.
    """

    def __init__(
        self,
        max_duration_sec: float = 30.0,
        sample_rate: int = 16000,
    ):
        self.max_duration_sec = max_duration_sec
        self.sample_rate = sample_rate
        self.max_samples: int = int(max_duration_sec * sample_rate)
        self.segments: deque[tuple[np.ndarray, dict]] = deque()
        self.total_samples: int = 0

    def add_audio_segment(
        self,
        audio_np: np.ndarray,
        **meta: Any,
    ) -> None:
        """Add a new audio chunk. Timestamps are completely ignored."""
        if len(audio_np) == 0:
            return
        if audio_np.ndim != 1:
            raise ValueError("Audio must be a 1D (mono) array")
        if audio_np.dtype != np.dtype(np.int16):
            raise TypeError("add_audio_segment expects np.int16 array")

        audio_np = audio_np.astype(np.int16, copy=True)
        chunk_samples = len(audio_np)

        seg_duration = chunk_samples / self.sample_rate
        if seg_duration - self.max_duration_sec > 1e-6:
            raise ValueError(
                f"Segment duration ({seg_duration:.2f}s) exceeds "
                f"max_duration_sec ({self.max_duration_sec:.2f}s). "
                "Split the audio before adding."
            )

        self.segments.append((audio_np, meta))
        self.total_samples += chunk_samples

        self._prune_old_segments()

    # ────────────────────────────────────────────────
    def _prune_old_segments(self) -> None:
        while self.segments and self.total_samples > self.max_samples:
            oldest_audio, _ = self.segments.popleft()
            self.total_samples -= len(oldest_audio)

    def get_full_audio(self) -> np.ndarray:
        """Return the last N seconds as int16 PCM (exactly what FunASR expects)."""
        if not self.segments:
            return np.array([], dtype=np.int16)

        # All segments are already int16 → just concatenate
        return np.concatenate([audio for audio, _ in self.segments]).astype(np.int16)

    def get_total_duration(self) -> float:
        """Returns current buffered duration (will always be ≤ max_duration_sec)."""
        return self.total_samples / self.sample_rate

    def get_history(
        self,
        max_segments: int = 5,
    ) -> List[SubtitleHistoryItem]:
        """
        Build translation history from buffered segments.

        Returns:
            List[SubtitleHistoryItem]:
                Alternating user (JA) and assistant (EN) messages.

        Notes:
            - Uses only segments with both JA and EN text.
            - Returns last N segments only (bounded context).
            - Preserves chronological order.
        """
        if not self.segments:
            return []

        history: List[SubtitleHistoryItem] = []

        # Take last N segments
        selected_segments = list(self.segments)[-max_segments:]

        for _, meta in selected_segments:
            ja = (meta.get("ja_text") or "").strip()
            en = (meta.get("en_text") or "").strip()

            if not ja or not en:
                continue

            history.append(
                {
                    "role": "user",
                    "content": ja,
                }
            )
            history.append(
                {
                    "role": "assistant",
                    "content": en,
                }
            )

        return history

    def reset(self) -> None:
        """Clear all buffered audio and metadata, reset total sample count to 0."""
        self.segments.clear()
        self.total_samples = 0
