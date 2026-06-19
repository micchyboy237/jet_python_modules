# jet_python_modules/jet/audio/helpers/circular_audio_buffer.py

"""CircularAudioBuffer — fixed-duration sliding window with timestamp support."""

import logging
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Iterator, List, Optional

import numpy as np
from jet.audio.helpers.config import SAMPLE_RATE

logger = logging.getLogger(__name__)

_MIN_REAL_GAP_SEC: float = 0.050
# Gaps below this threshold are likely clock jitter, not real data loss
_GAP_WARNING_THRESHOLD_SEC: float = 0.100


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
        self._chunks: deque[tuple[np.ndarray, Optional[datetime]]] = deque()
        self._window_samples: int = 0
        self._total_samples: int = 0
        self._trimmed_samples: int = 0
        self._total_gap_samples: int = 0
        self._gap_events: List[dict] = []
        self._window_start_time: Optional[datetime] = None
        self._has_timestamps: bool = False
        self._last_chunk_end_time: Optional[datetime] = None
        self._dropped_blocks: int = 0
        self._dropped_lock = threading.Lock()

    def record_dropped_block(self) -> None:
        """Called from the real-time audio callback when a queue-full drop occurs.

        Thread-safe; uses a simple lock. Called from the callback thread only
        when the consumer is too slow to drain blocks, which means a lock
        contention here is evidence of real overload — acceptable cost.
        """
        with self._dropped_lock:
            self._dropped_blocks += 1

    @property
    def dropped_blocks(self) -> int:
        """Total blocks dropped because the consumer was too slow."""
        with self._dropped_lock:
            return self._dropped_blocks

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

        if capture_time is not None and self._last_chunk_end_time is not None:
            expected_time = self._last_chunk_end_time
            actual_time = capture_time
            gap_sec = (actual_time - expected_time).total_seconds()

            if gap_sec > _MIN_REAL_GAP_SEC:
                gap_samples = int(gap_sec * self.sample_rate)

                # Use appropriate log level based on gap severity
                if gap_sec >= _GAP_WARNING_THRESHOLD_SEC:
                    logger.warning(
                        "Audio gap detected: %.3fs (%d samples). "
                        "Expected chunk at %s, got chunk at %s",
                        gap_sec,
                        gap_samples,
                        expected_time.isoformat(),
                        actual_time.isoformat(),
                    )
                else:
                    logger.debug(
                        "Minor audio gap: %.3fs (%d samples). "
                        "Expected chunk at %s, got chunk at %s",
                        gap_sec,
                        gap_samples,
                        expected_time.isoformat(),
                        actual_time.isoformat(),
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
            elif gap_sec < -0.010:
                logger.debug(
                    "Timestamp went backwards by %.3fs — clock jitter, ignoring",
                    -gap_sec,
                )

        if not self._has_timestamps and capture_time is not None:
            self._window_start_time = capture_time
            self._has_timestamps = True
            logger.debug("Timestamp anchor set: %s", capture_time.isoformat())

        if capture_time is not None:
            self._last_chunk_end_time = capture_time + timedelta(seconds=chunk_duration)

        self._chunks.append((chunk, capture_time))
        self._window_samples += n
        self._total_samples += n

        while self._window_samples > self.max_samples and self._chunks:
            evicted_chunk, evicted_time = self._chunks.popleft()
            evicted_len = len(evicted_chunk)
            self._window_samples -= evicted_len
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
            "trim_to_sec(%.3f): dropping %d samples, window has %d samples (%.3fs)",
            sec,
            samples_to_drop,
            self._window_samples,
            self.window_sec,
        )
        dropped = 0
        while self._chunks and dropped < samples_to_drop:
            chunk, _ = self._chunks[0]
            remaining = samples_to_drop - dropped
            if len(chunk) <= remaining:
                self._chunks.popleft()
                dropped += len(chunk)
            else:
                self._chunks[0] = (chunk[remaining:], self._chunks[0][1])
                dropped = samples_to_drop
        self._window_samples -= dropped
        self._trimmed_samples += dropped
        if self._has_timestamps and self._window_start_time is not None and dropped > 0:
            dropped_sec = dropped / self.sample_rate
            self._window_start_time += timedelta(seconds=dropped_sec)
            logger.debug(
                "Timestamp anchor advanced by %.3fs to %s",
                dropped_sec,
                self._window_start_time.isoformat(),
            )

    @property
    def trimmed_sec(self) -> float:
        """Total seconds trimmed from the front of the buffer since construction."""
        return self._trimmed_samples / self.sample_rate

    @property
    def total_gap_sec(self) -> float:
        """Total gap seconds detected (real gaps only, above 50ms threshold)."""
        return self._total_gap_samples / self.sample_rate

    @property
    def gap_events(self) -> List[dict]:
        """List of detected gap events."""
        return self._gap_events.copy()

    def get_absolute_time(self, relative_seconds: float) -> Optional[str]:
        """Convert window-relative seconds to ISO 8601 UTC datetime string."""
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
        start_sample = max(0, min(start_sample, len(full)))
        end_sample = max(start_sample, min(end_sample, len(full)))
        seg = full[start_sample:end_sample]
        logger.debug(
            "slice_seconds(%.3f, %.3f): full=%d samples, "
            "slice=[%d:%d], result=%d samples (%.3fs)",
            start_sec,
            end_sec,
            len(full),
            start_sample,
            end_sample,
            len(seg),
            len(seg) / self.sample_rate,
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

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int) -> np.ndarray:
        return self._chunks[index][0]

    def __iter__(self) -> Iterator[np.ndarray]:
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
        drop_info = ""
        if self._dropped_blocks > 0:
            drop_info = f", dropped_blocks={self._dropped_blocks}"
        return (
            f"CircularAudioBuffer("
            f"max_sec={self.max_sec}, "
            f"window={self.window_sec:.2f}s / {self._window_samples} samples, "
            f"chunks={len(self._chunks)}, "
            f"total_samples={self._total_samples}"
            f"{ts_info}"
            f"{gap_info}"
            f"{drop_info})"
        )
