# jet_python_modules/jet/audio/circular_audio_buffer.py
from collections import deque
from typing import Iterator, List, Optional

import numpy as np
from jet.audio.helpers.config import SAMPLE_RATE


class CircularAudioBuffer:
    """Fixed-duration sliding window over a live microphone stream.

    Internally keeps a deque of raw numpy chunks.  Once the total number of
    samples exceeds ``max_samples``, the oldest chunks are evicted from the
    left so that the window always covers at most ``max_sec`` seconds of audio.

    The buffer exposes:
    - ``append(chunk)``        — add a new mic chunk, evict if over budget.
    - ``to_numpy()``           — return the full window as one contiguous array.
    - ``slice_seconds(s, e)``  — extract a time range (in seconds, relative to
                                  the start of the current window).
    - ``total_samples``        — cumulative samples seen since construction
                                  (never decreases); used to map VAD timestamps
                                  back to absolute positions when needed.

    It also satisfies the ``list[np.ndarray]`` duck-type that ``speech_detector``
    and ``extract_segment_data`` rely on:
    - ``len(buf)``             — number of chunks currently in the window.
    - ``buf[i]``               — individual chunk by index.
    - iteration                — yields chunks in order.
    """

    def __init__(
        self,
        max_sec: float,
        sample_rate: int = SAMPLE_RATE,
        dtype: str = "int16",
    ) -> None:
        self.max_sec = max_sec
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.max_samples: int = int(max_sec * sample_rate)

        self._chunks: deque[np.ndarray] = deque()
        self._window_samples: int = 0  # samples currently in the deque
        self._total_samples: int = 0  # cumulative samples ever appended

    # ------------------------------------------------------------------
    # Core mutation
    # ------------------------------------------------------------------

    def append(self, chunk: np.ndarray) -> None:
        """Add *chunk* to the right of the window, evicting from the left as needed."""
        self._chunks.append(chunk)
        n = len(chunk)
        self._window_samples += n
        self._total_samples += n

        # Evict oldest chunks until we're within budget.
        while self._window_samples > self.max_samples and self._chunks:
            evicted = self._chunks.popleft()
            self._window_samples -= len(evicted)

    def clear(self) -> None:
        """Reset the window (does not reset total_samples)."""
        self._chunks.clear()
        self._window_samples = 0

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def total_samples(self) -> int:
        """Cumulative samples appended since construction (monotonically increasing)."""
        return self._total_samples

    @property
    def window_samples(self) -> int:
        """Samples currently in the sliding window."""
        return self._window_samples

    @property
    def window_sec(self) -> float:
        return self._window_samples / self.sample_rate

    def to_numpy(self) -> np.ndarray:
        """Return the current window as one contiguous numpy array.

        If the buffer is empty, returns a zero-length array with the
        correct dtype.
        """
        if not self._chunks:
            return np.array([], dtype=self.dtype)
        return np.concatenate(list(self._chunks), axis=0)

    def slice_seconds(
        self,
        start_sec: float,
        end_sec: float,
        trim_silent: bool = False,
        silence_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Extract ``[start_sec, end_sec)`` relative to the window start.

        This replaces the ``extract_segment_data`` slice logic so callers
        can work in seconds (exactly as VAD reports them) rather than
        computing chunk indices.

        Args:
            start_sec: Seconds from the beginning of the current window.
            end_sec:   Seconds from the beginning of the current window.
            trim_silent: When True, strip leading/trailing silent 100 ms
                         sub-chunks (mirrors the old ``trim_silent`` flag).
            silence_threshold: RMS threshold for silence detection.
                               Auto-calibrated when None.
        """
        full = self.to_numpy()
        start_sample = int(round(start_sec * self.sample_rate))
        end_sample = int(round(end_sec * self.sample_rate))

        # Clamp to valid range.
        start_sample = max(0, min(start_sample, len(full)))
        end_sample = max(start_sample, min(end_sample, len(full)))

        seg = full[start_sample:end_sample]

        if trim_silent and len(seg) > 0:
            from jet.audio.helpers.silence import (
                calibrate_silence_threshold,
                trim_silent_chunks,
            )

            if silence_threshold is None:
                silence_threshold = calibrate_silence_threshold()
            chunk_size = int(0.1 * self.sample_rate)
            chunks: List[np.ndarray] = [
                seg[i : i + chunk_size] for i in range(0, len(seg), chunk_size)
            ]
            trimmed = trim_silent_chunks(chunks, silence_threshold)
            seg = (
                np.concatenate(trimmed, axis=0)
                if trimmed
                else np.array([], dtype=self.dtype)
            )

        return seg

    # ------------------------------------------------------------------
    # list[np.ndarray] duck-type interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> np.ndarray:
        # deque doesn't support slicing but does support integer index.
        return self._chunks[index]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._chunks)

    def __bool__(self) -> bool:
        return bool(self._chunks)

    def __repr__(self) -> str:
        return (
            f"CircularAudioBuffer("
            f"max_sec={self.max_sec}, "
            f"window={self.window_sec:.2f}s / {self._window_samples} samples, "
            f"chunks={len(self._chunks)}, "
            f"total_samples={self._total_samples})"
        )
