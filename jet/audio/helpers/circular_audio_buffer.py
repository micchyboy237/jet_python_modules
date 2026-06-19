"""CircularAudioBuffer — fixed-duration sliding window with timestamp support."""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Iterator, List, Optional

import numpy as np
from jet.audio.helpers.config import SAMPLE_RATE

logger = logging.getLogger(__name__)


class CircularAudioBuffer:
    """Fixed-duration sliding window over a live microphone stream.

    Internally keeps a deque of (chunk, capture_time) tuples. Once the total
    number of samples exceeds ``max_samples``, the oldest chunks are evicted
    so the window always covers at most ``max_sec`` seconds of audio.

    Tracks per-chunk capture timestamps to detect gaps and maintain accurate
    time mapping even with stream jitter.
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

        # Store (chunk, capture_time) tuples
        self._chunks: deque[tuple[np.ndarray, Optional[datetime]]] = deque()
        self._window_samples: int = 0
        self._total_samples: int = 0
        self._trimmed_samples: int = 0

        # Track gaps
        self._total_gap_samples: int = 0
        self._gap_events: List[dict] = []

        # Timestamp anchor
        self._window_start_time: Optional[datetime] = None
        self._has_timestamps: bool = False

        # Track last chunk time for gap detection
        self._last_chunk_end_time: Optional[datetime] = None

    def append(
        self,
        chunk: np.ndarray,
        capture_time: Optional[datetime] = None,
    ) -> None:
        """Add *chunk* to the right of the window, evicting from the left as needed.

        Args:
            chunk: Audio samples as numpy array.
            capture_time: UTC datetime when the **first sample** of this chunk
                         was captured. Used for gap detection and timestamp accuracy.
        """
        n = len(chunk)
        chunk_duration = n / self.sample_rate

        # ── Gap detection ──────────────────────────────────────────
        if capture_time is not None and self._last_chunk_end_time is not None:
            expected_time = self._last_chunk_end_time
            actual_time = capture_time
            gap_sec = (actual_time - expected_time).total_seconds()

            if gap_sec > 0.001:  # More than 1ms gap
                gap_samples = int(gap_sec * self.sample_rate)
                logger.warning(
                    f"Audio gap detected: {gap_sec:.3f}s ({gap_samples} samples). "
                    f"Expected chunk at {expected_time.isoformat()}, "
                    f"got chunk at {actual_time.isoformat()}"
                )
                self._total_gap_samples += gap_samples
                self._gap_events.append(
                    {
                        "gap_sec": gap_sec,
                        "gap_samples": gap_samples,
                        "expected_time": expected_time.isoformat(),
                        "actual_time": actual_time.isoformat(),
                    }
                )

        # ── Timestamp anchor ───────────────────────────────────────
        if not self._has_timestamps and capture_time is not None:
            self._window_start_time = capture_time
            self._has_timestamps = True
            logger.debug(f"Timestamp anchor set: {capture_time.isoformat()}")

        # Track chunk end time for gap detection
        if capture_time is not None:
            self._last_chunk_end_time = capture_time + timedelta(seconds=chunk_duration)

        # ── Store chunk ────────────────────────────────────────────
        self._chunks.append((chunk, capture_time))
        self._window_samples += n
        self._total_samples += n

        # ── Evict overflow ─────────────────────────────────────────
        while self._window_samples > self.max_samples and self._chunks:
            evicted_chunk, evicted_time = self._chunks.popleft()
            evicted_len = len(evicted_chunk)
            self._window_samples -= evicted_len

            # Advance anchor by the duration of evicted audio
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
        """
        samples_to_drop = min(
            int(sec * self.sample_rate),
            self._window_samples,
        )

        logger.debug(
            f"trim_to_sec({sec:.3f}): dropping {samples_to_drop} samples, "
            f"window has {self._window_samples} samples ({self.window_sec:.3f}s)"
        )

        dropped = 0
        while self._chunks and dropped < samples_to_drop:
            chunk, _ = self._chunks[0]
            remaining = samples_to_drop - dropped
            if len(chunk) <= remaining:
                self._chunks.popleft()
                dropped += len(chunk)
            else:
                # Partial trim: slice off the consumed prefix
                partial_len = remaining
                self._chunks[0] = (chunk[remaining:], self._chunks[0][1])
                dropped = samples_to_drop

        self._window_samples -= dropped
        self._trimmed_samples += dropped

        # ── Advance timestamp anchor ───────────────────────────────────
        if self._has_timestamps and self._window_start_time is not None and dropped > 0:
            dropped_sec = dropped / self.sample_rate
            self._window_start_time += timedelta(seconds=dropped_sec)
            logger.debug(
                f"Timestamp anchor advanced by {dropped_sec:.3f}s to "
                f"{self._window_start_time.isoformat()}"
            )

    @property
    def trimmed_sec(self) -> float:
        """Total seconds trimmed from the front of the buffer since construction."""
        return self._trimmed_samples / self.sample_rate

    @property
    def total_gap_sec(self) -> float:
        """Total gap seconds detected."""
        return self._total_gap_samples / self.sample_rate

    @property
    def gap_events(self) -> List[dict]:
        """List of detected gap events."""
        return self._gap_events.copy()

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
        """Return the current window as one contiguous numpy array."""
        if not self._chunks:
            return np.array([], dtype=self.dtype)
        chunks_only = [chunk for chunk, _ in self._chunks]
        return np.concatenate(chunks_only, axis=0)

    def slice_seconds(
        self,
        start_sec: float,
        end_sec: float,
        trim_silent: bool = False,
        silence_threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Extract ``[start_sec, end_sec)`` relative to the window start."""
        full = self.to_numpy()
        start_sample = int(round(start_sec * self.sample_rate))
        end_sample = int(round(end_sec * self.sample_rate))

        # Clamp to valid range
        start_sample = max(0, min(start_sample, len(full)))
        end_sample = max(start_sample, min(end_sample, len(full)))

        seg = full[start_sample:end_sample]

        logger.debug(
            f"slice_seconds({start_sec:.3f}, {end_sec:.3f}): "
            f"full={len(full)} samples, "
            f"slice=[{start_sample}:{end_sample}], "
            f"result={len(seg)} samples ({len(seg) / self.sample_rate:.3f}s)"
        )

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
        # Return just the chunk (without timestamp) for backward compatibility
        return self._chunks[index][0]

    def __iter__(self) -> Iterator[np.ndarray]:
        # Return just the chunks (without timestamps) for backward compatibility
        for chunk, _ in self._chunks:
            yield chunk

    def __bool__(self) -> bool:
        return bool(self._chunks)

    def __repr__(self) -> str:
        ts_info = ""
        if self._has_timestamps and self._window_start_time is not None:
            ts_info = f", window_start={self._window_start_time.isoformat()}"
        gap_info = ""
        if self._total_gap_samples > 0:
            gap_info = (
                f", gaps={self.total_gap_sec:.3f}s ({len(self._gap_events)} events)"
            )
        return (
            f"CircularAudioBuffer("
            f"max_sec={self.max_sec}, "
            f"window={self.window_sec:.2f}s / {self._window_samples} samples, "
            f"chunks={len(self._chunks)}, "
            f"total_samples={self._total_samples}"
            f"{ts_info}"
            f"{gap_info})"
        )
