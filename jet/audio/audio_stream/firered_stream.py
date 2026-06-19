"""FireredStream — context-managed sounddevice InputStream for VAD pipelines.
Yields (chunk, capture_time) tuples with precise wall-clock timestamps."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import TracebackType
from typing import Generator, Optional, Tuple, Type

import numpy as np
import sounddevice as sd
from jet.audio.helpers.config import FRAME_SHIFT_SAMPLE, SAMPLE_RATE
from jet.audio.helpers.silence import CHANNELS, DTYPE

logger = logging.getLogger(__name__)

DEFAULT_HOPS_PER_READ: int = 50
AudioChunk = np.ndarray
TimestampedChunk = Tuple[AudioChunk, datetime, bool]  # Added overflow flag


class FireredStream:
    """Thin wrapper around ``sd.InputStream`` tuned for Firered VAD.

    Yields ``(chunk, capture_time, overflow)`` tuples where:
    - ``capture_time`` is the wall-clock UTC time when the **first sample**
      of the chunk was captured
    - ``overflow`` is True if audio was lost since the last read

    Usage::
        with FireredStream() as stream:
            for chunk, capture_time, overflow in stream:
                if overflow:
                    logger.warning("Audio overflow detected!")
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
    latency : str | float
        PortAudio latency setting. 'low' for minimal latency (higher CPU/overflow risk),
        'high' for stability. Default 'high' for M1 reliability.
    """

    def __init__(
        self,
        *,
        hops_per_read: int = DEFAULT_HOPS_PER_READ,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        dtype: str = DTYPE,
        blocksize: int = FRAME_SHIFT_SAMPLE,
        latency: str | float = "high",  # Changed default for M1 stability
    ) -> None:
        self._hops_per_read = hops_per_read
        self._sample_rate = sample_rate
        self._channels = channels
        self._dtype = dtype
        self._blocksize = blocksize
        self._latency = latency
        self._chunk_size: int = blocksize * hops_per_read
        self._stream: Optional[sd.InputStream] = None
        self._stream_started_at: Optional[datetime] = None
        self._samples_read: int = 0
        self._total_overflows: int = 0
        self._total_samples_lost: int = 0

    def __enter__(self) -> "FireredStream":
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
            blocksize=self._blocksize,
            latency=self._latency,
            callback=None,  # Use blocking reads, not callback
        )
        self._stream.start()
        self._stream_started_at = datetime.now(timezone.utc)
        self._samples_read = 0
        self._total_overflows = 0
        self._total_samples_lost = 0
        logger.info(
            f"FireredStream started: sr={self._sample_rate}, "
            f"blocksize={self._blocksize}, latency={self._latency}"
        )
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
        if self._total_overflows > 0:
            logger.warning(
                f"Stream ended with {self._total_overflows} overflows, "
                f"approximately {self._total_samples_lost} samples lost "
                f"({self._total_samples_lost / self._sample_rate:.3f}s)"
            )
        self._stream_started_at = None

    def __iter__(self) -> Generator[TimestampedChunk, None, None]:
        """Yield ``(audio_chunk, capture_time, overflow)`` tuples."""
        if self._stream is None or self._stream_started_at is None:
            raise RuntimeError("FireredStream must be used as a context manager")

        while self._stream.active:
            chunk_capture_time = self._compute_timestamp(self._samples_read)

            try:
                chunk, overflow = self._stream.read(self._chunk_size)
            except sd.InputOverflow as e:
                logger.error(f"Input overflow: {e}. Audio data was lost!")
                # Try to recover by reading again
                chunk, overflow = self._stream.read(self._chunk_size)
                overflow = True

            if overflow:
                self._total_overflows += 1
                # Estimate lost samples based on time discrepancy
                expected_time = self._compute_timestamp(
                    self._samples_read + self._chunk_size
                )
                actual_time = datetime.now(timezone.utc)
                time_diff = (actual_time - expected_time).total_seconds()
                if time_diff > 0.01:  # More than 10ms drift
                    lost_samples = int(time_diff * self._sample_rate)
                    self._total_samples_lost += lost_samples
                    logger.warning(
                        f"Overflow #{self._total_overflows}: "
                        f"estimated {lost_samples} samples lost "
                        f"({time_diff:.3f}s behind). "
                        f"Total lost: {self._total_samples_lost} samples "
                        f"({self._total_samples_lost / self._sample_rate:.3f}s)"
                    )
                    # Adjust the sample counter to account for lost samples
                    self._samples_read += lost_samples

            self._samples_read += len(chunk)
            yield chunk, chunk_capture_time, overflow

    def _compute_timestamp(self, sample_offset: int) -> datetime:
        """Return the UTC datetime of the sample at *sample_offset*."""
        offset_seconds = sample_offset / self._sample_rate
        return self._stream_started_at + _timedelta_seconds(offset_seconds)

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
        """Total samples read since stream start (excluding estimated lost)."""
        return self._samples_read

    @property
    def total_overflows(self) -> int:
        """Total number of overflow events detected."""
        return self._total_overflows

    @property
    def total_samples_lost(self) -> int:
        """Estimated total samples lost due to overflows."""
        return self._total_samples_lost


def _timedelta_seconds(seconds: float) -> "datetime.timedelta":
    """Create a timedelta from float seconds, avoiding import in signature."""
    from datetime import timedelta

    return timedelta(seconds=seconds)
