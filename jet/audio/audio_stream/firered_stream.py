"""FireredStream — context-managed sounddevice InputStream for VAD pipelines.
Yields (chunk, capture_time) tuples with precise wall-clock timestamps."""

from __future__ import annotations

from datetime import datetime, timezone
from types import TracebackType
from typing import Generator, Optional, Tuple, Type

import numpy as np
import sounddevice as sd
from jet.audio.helpers.config import FRAME_SHIFT_SAMPLE, SAMPLE_RATE
from jet.audio.helpers.silence import CHANNELS, DTYPE

DEFAULT_HOPS_PER_READ: int = 50

# Type alias for clarity
AudioChunk = np.ndarray
TimestampedChunk = Tuple[AudioChunk, datetime]


class FireredStream:
    """Thin wrapper around ``sd.InputStream`` tuned for Firered VAD.

    Yields ``(chunk, capture_time)`` tuples where ``capture_time`` is the
    wall-clock UTC time when the **first sample** of the chunk was captured.

    Usage::

        with FireredStream() as stream:
            for chunk, capture_time in stream:
                # chunk is np.ndarray of shape (N, CHANNELS), dtype=DTYPE
                # capture_time is datetime (UTC)
                ...

    Parameters
    ----------
    hops_per_read : int
        How many 10-ms VAD frames to batch. Default 50 → 500 ms chunks.
    sample_rate : int
        Sample rate in Hz. Defaults to ``fireredvad``'s 16 kHz.
    channels : int
        Audio channels. Defaults to the project-wide ``CHANNELS`` constant.
    dtype : str
        NumPy dtype string. Defaults to ``DTYPE`` (float32).
    blocksize : int
        Low-level PortAudio block size. Defaults to ``FRAME_SHIFT_SAMPLE``.
    """

    def __init__(
        self,
        *,
        hops_per_read: int = DEFAULT_HOPS_PER_READ,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        dtype: str = DTYPE,
        blocksize: int = FRAME_SHIFT_SAMPLE,
    ) -> None:
        self._hops_per_read = hops_per_read
        self._sample_rate = sample_rate
        self._channels = channels
        self._dtype = dtype
        self._blocksize = blocksize
        self._chunk_size: int = blocksize * hops_per_read

        # Set when entering the context manager
        self._stream: Optional[sd.InputStream] = None

        # Timestamp tracking
        self._stream_started_at: Optional[datetime] = None
        self._samples_read: int = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "FireredStream":
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
            blocksize=self._blocksize,
        )
        self._stream.start()
        # Capture the exact moment the stream starts
        self._stream_started_at = datetime.now(timezone.utc)
        self._samples_read = 0
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._stream_started_at = None

    # ------------------------------------------------------------------
    # Iteration — yields timestamped chunks
    # ------------------------------------------------------------------
    def __iter__(self) -> Generator[TimestampedChunk, None, None]:
        """Yield ``(audio_chunk, capture_time_utc)`` tuples."""
        if self._stream is None or self._stream_started_at is None:
            raise RuntimeError("FireredStream must be used as a context manager")

        while self._stream.active:
            # Calculate timestamp for the FIRST sample of this chunk
            chunk_capture_time = self._compute_timestamp(self._samples_read)

            chunk, _ = self._stream.read(self._chunk_size)
            self._samples_read += len(chunk)

            yield chunk, chunk_capture_time

    # ------------------------------------------------------------------
    # Timestamp calculation
    # ------------------------------------------------------------------
    def _compute_timestamp(self, sample_offset: int) -> datetime:
        """Return the UTC datetime of the sample at *sample_offset*.

        Uses the stream start time as anchor and sample-accurate offsets.
        """
        offset_seconds = sample_offset / self._sample_rate
        return self._stream_started_at + _timedelta_seconds(offset_seconds)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def chunk_size(self) -> int:
        """Number of samples yielded per iteration."""
        return self._chunk_size

    @property
    def chunk_duration_sec(self) -> float:
        """Duration of one chunk in seconds."""
        return self._chunk_size / self._sample_rate

    @property
    def stream_started_at(self) -> Optional[datetime]:
        """UTC timestamp when the stream was started (entered)."""
        return self._stream_started_at

    @property
    def total_samples_read(self) -> int:
        """Total samples read since stream start."""
        return self._samples_read


def _timedelta_seconds(seconds: float) -> "datetime.timedelta":
    """Create a timedelta from float seconds, avoiding import in signature."""
    from datetime import timedelta

    return timedelta(seconds=seconds)
