from collections import deque
from collections.abc import Sequence
from typing import Deque, List, Optional

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.helpers.config import FRAME_SHIFT_MS, SAMPLE_RATE
from numpy.typing import NDArray


class SpeechSegmentsBuffer:
    """
    Fixed-size rolling buffer for audio + metadata.
    Keeps the most recent audio (up to `max_duration_sec`) for context.
    """

    def __init__(
        self,
        max_duration_sec: float = 30.0,
        sample_rate: int = SAMPLE_RATE,
        frame_shift_ms: float = FRAME_SHIFT_MS,
    ):
        self.max_duration_sec = max_duration_sec
        self.sample_rate = sample_rate
        self.frame_shift_ms = frame_shift_ms
        self.frame_duration_s = frame_shift_ms / 1000.0
        self.max_samples: int = int(max_duration_sec * sample_rate)

        # Deque of (audio: int16 ndarray, metadata)
        self.segments: Deque[tuple[NDArray[np.int16], SpeechSegment]] = deque()
        self.total_samples: int = 0

    def _compute_times(self, frame_idx: int) -> tuple[float, float]:
        """Convert frame index to start/end time in seconds."""
        start_s = frame_idx * self.frame_duration_s
        end_s = (frame_idx + 1) * self.frame_duration_s
        return start_s, end_s

    def add_audio_segment(
        self,
        audio_np: NDArray[np.int16],
        meta: SpeechSegment,
    ) -> None:
        """
        Add a new audio chunk. Timestamps in meta are ignored.

        Raises:
            TypeError: if input is not int16
            ValueError: if a single segment exceeds max_duration_sec
        """
        if audio_np.dtype != np.int16:
            raise TypeError("add_audio_segment expects np.int16 array")

        # Ensure we own a contiguous int16 copy
        audio_np = np.asarray(audio_np, dtype=np.int16).copy()

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

    def _prune_old_segments(self) -> None:
        """Remove oldest segments until we are back under max_samples."""
        while self.segments and self.total_samples > self.max_samples:
            oldest_audio, _ = self.segments.popleft()
            self.total_samples -= len(oldest_audio)

    def get_context_audio(
        self, max_segments: Optional[int] = None
    ) -> NDArray[np.int16]:
        """
        Return concatenated audio as contiguous int16 PCM (ready for FunASR etc.).

        Args:
            max_segments: If provided, return only the last N segments.
                         If None (default), return all currently buffered audio.
        """
        if not self.segments:
            return np.array([], dtype=np.int16)

        if max_segments is not None:
            segments_to_use = list(self.segments)[-max_segments:]
        else:
            segments_to_use = self.segments

        arrays = [audio for audio, _ in segments_to_use]
        return np.concatenate(arrays).astype(
            np.int16
        )  # ensures contiguous + correct dtype

    def get_total_duration(self) -> float:
        """Current buffered duration in seconds (≤ max_duration_sec)."""
        return self.total_samples / self.sample_rate

    def get_segments(self) -> Sequence[tuple[NDArray[np.int16], SpeechSegment]]:
        """Return list of (audio, meta) tuples currently in the buffer."""
        return list(self.segments)

    def get_list_metadata(self) -> List[SpeechSegment]:
        """Return ordered list of metadata for all buffered segments."""
        return [meta for _, meta in self.segments]

    def reset(self) -> None:
        """Clear everything."""
        self.segments.clear()
        self.total_samples = 0
