"""CircularAudioBuffer — fixed-duration sliding window with timestamp support."""

from collections import deque
from datetime import datetime, timedelta
from typing import Iterator, List, Optional

import numpy as np
from jet.audio.helpers.config import SAMPLE_RATE


class CircularAudioBuffer:
    """Fixed-duration sliding window over a live microphone stream.

    Internally keeps a deque of raw numpy chunks.  Once the total number of
    samples exceeds ``max_samples``, the oldest chunks are evicted from the
    left so that the window always covers at most ``max_sec`` seconds of audio.

    The buffer exposes:
    - ``append(chunk, capture_time=None)`` — add a new mic chunk, evict if over budget.
    - ``to_numpy()``                      — return the full window as one contiguous array.
    - ``slice_seconds(s, e)``             — extract a time range (in seconds, relative to
                                             the start of the current window).
    - ``get_absolute_time(rel_sec)``      — convert window-relative seconds to UTC datetime.
    - ``total_samples``                   — cumulative samples seen since construction
                                             (never decreases).

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
        self._trimmed_samples: int = 0  # samples removed via trim_to_sec

        # ── Timestamp tracking ──────────────────────────────────────────
        # Absolute UTC time of the FIRST sample currently in the window.
        # Set when the first chunk with a capture_time is appended.
        # Advanced by trim_to_sec to stay accurate.
        self._window_start_time: Optional[datetime] = None
        # Whether we've received at least one timestamped chunk.
        self._has_timestamps: bool = False

    # ------------------------------------------------------------------
    # Core mutation
    # ------------------------------------------------------------------

    def append(
        self,
        chunk: np.ndarray,
        capture_time: Optional[datetime] = None,
    ) -> None:
        """Add *chunk* to the right of the window, evicting from the left as needed.

        Args:
            chunk: Audio samples as numpy array.
            capture_time: UTC datetime when the **first sample** of this chunk
                         was captured. Only used for the very first chunk to
                         establish the window's time anchor.
        """
        self._chunks.append(chunk)
        n = len(chunk)
        self._window_samples += n
        self._total_samples += n

        # ── Timestamp anchor: set on first timestamped chunk ──────────
        if not self._has_timestamps and capture_time is not None:
            self._window_start_time = capture_time
            self._has_timestamps = True

        # Evict oldest chunks until we're within budget.
        # If we evict, advance the timestamp anchor accordingly.
        while self._window_samples > self.max_samples and self._chunks:
            evicted = self._chunks.popleft()
            evicted_len = len(evicted)
            self._window_samples -= evicted_len

            # Advance anchor by the duration of evicted audio.
            if self._has_timestamps and self._window_start_time is not None:
                evicted_sec = evicted_len / self.sample_rate
                self._window_start_time += timedelta(seconds=evicted_sec)

    def clear(self) -> None:
        """Reset the window (does not reset total_samples or timestamp anchor)."""
        self._chunks.clear()
        self._window_samples = 0

    def trim_to_sec(self, sec: float) -> None:
        """Drop the first *sec* seconds from the front of the window.

        After this call all timestamps returned by VAD (which are relative
        to the window start) remain correct — the window simply starts later.
        Use ``trimmed_sec`` to convert between absolute recording time and
        window-relative time when needed.

        Clamped to the current window length; asking to trim more than the
        buffer holds trims everything.

        Also advances the timestamp anchor so ``get_absolute_time`` stays accurate.
        """
        samples_to_drop = min(
            int(sec * self.sample_rate),
            self._window_samples,
        )
        dropped = 0
        while self._chunks and dropped < samples_to_drop:
            chunk = self._chunks[0]
            remaining = samples_to_drop - dropped
            if len(chunk) <= remaining:
                evicted = self._chunks.popleft()
                dropped += len(evicted)
            else:
                # Partial trim: slice off the consumed prefix.
                partial_len = remaining
                self._chunks[0] = chunk[remaining:]
                dropped = samples_to_drop

        self._window_samples -= dropped
        self._trimmed_samples += dropped

        # ── Advance timestamp anchor ───────────────────────────────────
        if self._has_timestamps and self._window_start_time is not None and dropped > 0:
            dropped_sec = dropped / self.sample_rate
            self._window_start_time += timedelta(seconds=dropped_sec)

    @property
    def trimmed_sec(self) -> float:
        """Total seconds trimmed from the front of the buffer since construction."""
        return self._trimmed_samples / self.sample_rate

    # ------------------------------------------------------------------
    # Timestamp conversion
    # ------------------------------------------------------------------

    def get_absolute_time(self, relative_seconds: float) -> Optional[str]:
        """Convert window-relative seconds to ISO 8601 UTC datetime string.

        Args:
            relative_seconds: Seconds from the start of the **current window**
                             (the same values VAD returns in segment["start"]).

        Returns:
            ISO 8601 string like ``"2026-05-25T14:32:17.123456+00:00"``,
            or None if no timestamp anchor has been set.

        Example:
            >>> buf.get_absolute_time(2.3)
            "2026-05-25T14:32:17.123456+00:00"
        """
        if self._window_start_time is None:
            return None
        absolute_dt = self._window_start_time + timedelta(seconds=relative_seconds)
        return absolute_dt.isoformat()

    @property
    def window_start_time(self) -> Optional[datetime]:
        """Absolute UTC time of the first sample currently in the window."""
        return self._window_start_time

    @property
    def has_timestamps(self) -> bool:
        """Whether the buffer has a valid timestamp anchor."""
        return self._has_timestamps

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
        ts_info = ""
        if self._has_timestamps and self._window_start_time is not None:
            ts_info = f", window_start={self._window_start_time.isoformat()}"
        return (
            f"CircularAudioBuffer("
            f"max_sec={self.max_sec}, "
            f"window={self.window_sec:.2f}s / {self._window_samples} samples, "
            f"chunks={len(self._chunks)}, "
            f"total_samples={self._total_samples}"
            f"{ts_info})"
        )
