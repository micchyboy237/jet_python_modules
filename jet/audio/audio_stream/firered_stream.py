# jet_python_modules/jet/audio/audio_stream/firered_stream.py

"""FireredStream — context-managed sounddevice InputStream for VAD pipelines.
Yields (chunk, capture_time, overflow) tuples with precise wall-clock timestamps.
KEY DESIGN: Uses a callback + queue.Queue instead of blocking stream.read().
The audio driver invokes the callback from a real-time OS thread the instant each
block is ready, so the PortAudio ring-buffer is drained immediately — independent
of how long the Python main loop spends on VAD / speech extraction.
"""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Generator, Optional, Tuple, Type

import numpy as np
import sounddevice as sd
from jet.audio.helpers.config import FRAME_SHIFT_SAMPLE, SAMPLE_RATE
from jet.audio.helpers.silence import CHANNELS, DTYPE

logger = logging.getLogger(__name__)

DEFAULT_HOPS_PER_READ: int = 50
_QUEUE_MAXSIZE: int = 75  # Increased from 20 to 50 (~37s buffer)
AudioChunk = np.ndarray
TimestampedChunk = Tuple[AudioChunk, datetime, bool]
_STOP = object()


class FireredStream:
    """Thin wrapper around ``sd.InputStream`` tuned for Firered VAD.

    Yields ``(chunk, capture_time, overflow)`` tuples where:
    - ``capture_time`` — UTC wall-clock time of the **first sample** in the chunk
      (derived from PortAudio's ``time.inputBufferAdcTime`` for accuracy)
    - ``overflow`` — True if PortAudio discarded audio since the last callback

    Architecture
    ------------
    A **callback** is registered with ``sd.InputStream``.  PortAudio fires it
    from a high-priority OS thread the moment each ``blocksize``-frame block is
    captured.  The callback does nothing except copy the data into a
    ``queue.Queue``; all heavy work happens in the caller's thread via
    ``__iter__``.  This prevents the driver ring-buffer from overflowing when
    the consumer thread is busy with VAD.

    Parameters
    ----------
    hops_per_read : int
        Callback blocks to batch per yielded chunk. Default 50 → 500 ms chunks.
    sample_rate : int
        Sample rate in Hz. Defaults to fireredvad's 16 kHz.
    channels : int
        Audio channels. Defaults to project-wide CHANNELS constant.
    dtype : str
        NumPy dtype string. Defaults to DTYPE (float32).
    blocksize : int
        PortAudio block size (frames per callback). Default FRAME_SHIFT_SAMPLE.
    latency : str | float
        PortAudio latency. 'high' for maximum stability.
    queue_maxsize : int
        Max pending callback blocks before back-pressure. Default 50 (~25s).
    """

    def __init__(
        self,
        *,
        hops_per_read: int = DEFAULT_HOPS_PER_READ,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        dtype: str = DTYPE,
        blocksize: int = FRAME_SHIFT_SAMPLE,
        latency: str | float = "high",
        queue_maxsize: int = _QUEUE_MAXSIZE,
    ) -> None:
        self._hops_per_read = hops_per_read
        self._sample_rate = sample_rate
        self._channels = channels
        self._dtype = dtype
        self._blocksize = blocksize
        self._latency = latency
        self._queue_maxsize = queue_maxsize
        self._chunk_size: int = blocksize * hops_per_read
        self._stream: Optional[sd.InputStream] = None
        self._stream_started_at: Optional[datetime] = None
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._dropped_blocks: int = 0
        self._dropped_lock = threading.Lock()
        self._samples_read: int = 0
        self._total_overflows: int = 0
        self._total_samples_lost: int = 0

    def __enter__(self) -> "FireredStream":
        self._q = queue.Queue(maxsize=self._queue_maxsize)
        self._samples_read = 0
        self._total_overflows = 0
        self._total_samples_lost = 0
        with self._dropped_lock:
            self._dropped_blocks = 0
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
            blocksize=self._blocksize,
            latency=self._latency,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._stream_started_at = datetime.now(timezone.utc)
        logger.info(
            "FireredStream started (callback mode): sr=%d, blocksize=%d, "
            "latency=%s, queue_maxsize=%d, chunk_duration=%.3fs",
            self._sample_rate,
            self._blocksize,
            self._latency,
            self._queue_maxsize,
            self._chunk_size / self._sample_rate,
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
        try:
            self._q.put_nowait(_STOP)
        except queue.Full:
            pass
        dropped = self.dropped_blocks
        if self._total_overflows > 0 or dropped > 0:
            logger.warning(
                "Stream ended: %d PortAudio overflow(s), ~%d samples lost (%.3fs), "
                "%d block(s) dropped due to slow consumer (queue full)",
                self._total_overflows,
                self._total_samples_lost,
                self._total_samples_lost / self._sample_rate,
                dropped,
            )
        else:
            logger.info("Stream ended cleanly — no overflows or drops detected.")
        self._stream_started_at = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """Fires for every ``blocksize`` frames.  Copy → queue, nothing else."""
        overflow = bool(status.input_overflow)
        try:
            adc_offset = float(time_info.inputBufferAdcTime)
            capture_time = self._stream_started_at + timedelta(seconds=adc_offset)
        except Exception:
            capture_time = datetime.now(timezone.utc)
        try:
            self._q.put_nowait((indata.copy(), capture_time, overflow))
        except queue.Full:
            with self._dropped_lock:
                self._dropped_blocks += 1

    def __iter__(self) -> Generator[TimestampedChunk, None, None]:
        """Yield ``(audio_chunk, capture_time, overflow)`` tuples."""
        if self._stream is None or self._stream_started_at is None:
            raise RuntimeError("FireredStream must be used as a context manager")
        pending_blocks: list[np.ndarray] = []
        chunk_start_time: Optional[datetime] = None
        chunk_overflow = False
        while True:
            try:
                item = self._q.get(timeout=0.5)  # Reduced from 1.0s for faster response
            except queue.Empty:
                if self._stream is None or not self._stream.active:
                    logger.debug("FireredStream: stream inactive, exiting iterator")
                    break
                dropped = self.dropped_blocks
                if dropped > 0:
                    logger.warning(
                        "FireredStream: %d block(s) dropped (queue full) — "
                        "VAD processing may be too slow; consider reducing "
                        "VAD window or increasing queue_maxsize",
                        dropped,
                    )
                continue
            if item is _STOP:
                logger.debug("FireredStream: stop sentinel received")
                break
            block, capture_time, overflow = item
            if chunk_start_time is None:
                chunk_start_time = capture_time
            pending_blocks.append(block)
            chunk_overflow = chunk_overflow or overflow
            if len(pending_blocks) >= self._hops_per_read:
                chunk = np.concatenate(pending_blocks, axis=0)
                self._samples_read += len(chunk)
                if chunk_overflow:
                    self._total_overflows += 1
                    expected_time = self._compute_timestamp(
                        self._samples_read - len(chunk)
                    )
                    time_diff = (chunk_start_time - expected_time).total_seconds()
                    if time_diff > 0.01:
                        lost_samples = int(time_diff * self._sample_rate)
                        self._total_samples_lost += lost_samples
                        self._samples_read += lost_samples
                        logger.warning(
                            "PortAudio overflow #%d: ~%d samples lost (%.3fs). "
                            "Total lost: %d samples (%.3fs). "
                            "Queue depth: %d/%d. Dropped blocks: %d",
                            self._total_overflows,
                            lost_samples,
                            time_diff,
                            self._total_samples_lost,
                            self._total_samples_lost / self._sample_rate,
                            self._q.qsize(),
                            self._queue_maxsize,
                            self.dropped_blocks,
                        )
                yield chunk, chunk_start_time, chunk_overflow
                pending_blocks = []
                chunk_start_time = None
                chunk_overflow = False

    def _compute_timestamp(self, sample_offset: int) -> datetime:
        offset_seconds = sample_offset / self._sample_rate
        return self._stream_started_at + timedelta(seconds=offset_seconds)

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
        """UTC timestamp when the stream was started."""
        return self._stream_started_at

    @property
    def total_samples_read(self) -> int:
        """Total samples read since stream start (including estimated lost)."""
        return self._samples_read

    @property
    def total_overflows(self) -> int:
        """Total PortAudio overflow events detected."""
        return self._total_overflows

    @property
    def total_samples_lost(self) -> int:
        """Estimated total samples lost due to PortAudio overflows."""
        return self._total_samples_lost

    @property
    def dropped_blocks(self) -> int:
        """Blocks dropped because the Python consumer queue was full."""
        with self._dropped_lock:
            return self._dropped_blocks

    @property
    def queue_size(self) -> int:
        """Current pending (unprocessed) blocks in the internal queue."""
        return self._q.qsize()
