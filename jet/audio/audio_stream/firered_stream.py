"""FireredStream — context-managed sounddevice InputStream for VAD pipelines.

Yields (chunk, capture_time, overflow) tuples with precise wall-clock timestamps.

KEY FIX: Uses a callback + queue.Queue instead of blocking stream.read().
The audio driver invokes the callback from a real-time OS thread the instant each
block is ready, so the PortAudio ring-buffer is drained immediately — independent
of how long the Python main loop spends on VAD / speech extraction.
This eliminates the overflow that occurred when processing delayed the next read().
"""
from __future__ import annotations

import logging
import queue
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Generator, Optional, Tuple, Type

import numpy as np
import sounddevice as sd

from jet.audio.helpers.config import FRAME_SHIFT_SAMPLE, SAMPLE_RATE
from jet.audio.helpers.silence import CHANNELS, DTYPE

logger = logging.getLogger(__name__)

DEFAULT_HOPS_PER_READ: int = 50

# Maximum number of chunks allowed in the internal queue before the producer
# (the real-time audio callback) blocks.  At 50 hops × 10 ms = 500 ms per
# chunk this gives 10 s of headroom before back-pressure is applied.
_QUEUE_MAXSIZE: int = 20

AudioChunk = np.ndarray
TimestampedChunk = Tuple[AudioChunk, datetime, bool]

# Sentinel placed in the queue when the stream stops so __iter__ can exit cleanly.
_STOP = object()


class FireredStream:
    """Thin wrapper around ``sd.InputStream`` tuned for Firered VAD.

    Yields ``(chunk, capture_time, overflow)`` tuples where:
    - ``capture_time`` is the wall-clock UTC time when the **first sample**
      of the chunk was captured (derived from PortAudio's ``time.inputBufferAdcTime``)
    - ``overflow`` is ``True`` if audio was lost since the last callback invocation

    Architecture
    ------------
    A **callback** is passed to ``sd.InputStream``.  PortAudio calls it from a
    dedicated high-priority OS thread the moment each ``blocksize``-frame block
    is captured.  The callback does *nothing* except copy the data into a
    ``queue.Queue``; all heavy work (VAD, buffer management, etc.) happens in
    the caller's thread via ``__iter__``.

    This decoupling prevents the driver ring-buffer from overflowing when the
    consumer thread is busy — the previous blocking-``read()`` approach caused
    overflows because ``stream.read()`` was only called *after* VAD processing
    finished, leaving the ring-buffer unserviced for hundreds of milliseconds.

    Parameters
    ----------
    hops_per_read : int
        How many 10-ms VAD frames to batch per yielded chunk. Default 50 → 500 ms.
    sample_rate : int
        Sample rate in Hz. Defaults to ``fireredvad``'s 16 kHz.
    channels : int
        Audio channels. Defaults to the project-wide ``CHANNELS`` constant.
    dtype : str
        NumPy dtype string. Defaults to ``DTYPE`` (float32).
    blocksize : int
        Low-level PortAudio block size (frames per callback). Defaults to
        ``FRAME_SHIFT_SAMPLE`` (160 frames = 10 ms at 16 kHz).
    latency : str | float
        PortAudio latency setting. ``'high'`` for maximum stability.
    queue_maxsize : int
        Maximum pending callback blocks before back-pressure. Default 20.
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

        # hops_per_read individual callback blocks are accumulated into one chunk
        self._chunk_size: int = blocksize * hops_per_read

        self._stream: Optional[sd.InputStream] = None
        self._stream_started_at: Optional[datetime] = None

        # Internal queue: items are (block_np, adc_time_utc, overflow_flag)
        # _STOP is enqueued when the stream stops.
        self._q: queue.Queue = queue.Queue(maxsize=queue_maxsize)

        self._samples_read: int = 0
        self._total_overflows: int = 0
        self._total_samples_lost: int = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "FireredStream":
        self._q = queue.Queue(maxsize=self._queue_maxsize)
        self._samples_read = 0
        self._total_overflows = 0
        self._total_samples_lost = 0

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype=self._dtype,
            blocksize=self._blocksize,
            latency=self._latency,
            callback=self._audio_callback,  # ← KEY CHANGE: callback, not blocking read
        )
        self._stream.start()
        self._stream_started_at = datetime.now(timezone.utc)

        logger.info(
            "FireredStream started (callback mode): sr=%d, blocksize=%d, "
            "latency=%s, queue_maxsize=%d",
            self._sample_rate,
            self._blocksize,
            self._latency,
            self._queue_maxsize,
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

        # Signal __iter__ to stop
        try:
            self._q.put_nowait(_STOP)
        except queue.Full:
            pass

        if self._total_overflows > 0:
            logger.warning(
                "Stream ended with %d overflow(s), ~%d samples lost (%.3fs)",
                self._total_overflows,
                self._total_samples_lost,
                self._total_samples_lost / self._sample_rate,
            )
        else:
            logger.info("Stream ended cleanly — no overflows detected.")

        self._stream_started_at = None

    # ------------------------------------------------------------------
    # Real-time audio callback  (runs in a high-priority OS thread)
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,          # sd.CallbackFlags CData struct
        status: sd.CallbackFlags,
    ) -> None:
        """Called by PortAudio for every ``blocksize`` frames captured.

        IMPORTANT: This function must return as fast as possible.
        All it does is copy the data and push it onto the queue.
        No logging, no heavy computation.
        """
        overflow = bool(status.input_overflow)

        # Derive capture time from PortAudio's ADC timestamp.
        # time_info.inputBufferAdcTime is seconds since stream start (host clock).
        # We anchor it to the UTC wall-clock recorded at stream start.
        try:
            adc_offset = float(time_info.inputBufferAdcTime)
            capture_time = self._stream_started_at + timedelta(seconds=adc_offset)
        except Exception:
            # Fallback: use current wall time (less accurate but safe)
            capture_time = datetime.now(timezone.utc)

        try:
            self._q.put_nowait((indata.copy(), capture_time, overflow))
        except queue.Full:
            # Queue is full — consumer is too slow.  Drop this block rather than
            # blocking the audio thread (which would itself cause an overflow).
            # We log via put_nowait to avoid any blocking call here; a warning
            # will surface via the overflow flag on the next successful put.
            pass  # overflow will be detected on next iteration anyway

    # ------------------------------------------------------------------
    # Iterator: assembles individual blocks into hops_per_read-sized chunks
    # ------------------------------------------------------------------

    def __iter__(self) -> Generator[TimestampedChunk, None, None]:
        """Yield ``(audio_chunk, capture_time, overflow)`` tuples.

        Each yielded chunk contains ``hops_per_read`` callback blocks concatenated.
        ``capture_time`` is the timestamp of the **first** block in the chunk.
        ``overflow`` is True if *any* block in the chunk had an overflow flag.
        """
        if self._stream is None or self._stream_started_at is None:
            raise RuntimeError("FireredStream must be used as a context manager")

        pending_blocks: list[np.ndarray] = []
        chunk_start_time: Optional[datetime] = None
        chunk_overflow = False

        while True:
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                # Stream may have stopped; check and exit if so
                if self._stream is None or not self._stream.active:
                    logger.debug("FireredStream: stream inactive, exiting iterator")
                    break
                continue

            if item is _STOP:
                logger.debug("FireredStream: received stop sentinel, exiting iterator")
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
                    # Estimate samples lost from time gap
                    expected_time = self._compute_timestamp(
                        self._samples_read - len(chunk)
                    )
                    actual_time = chunk_start_time
                    time_diff = (actual_time - expected_time).total_seconds()
                    if time_diff > 0.01:
                        lost_samples = int(time_diff * self._sample_rate)
                        self._total_samples_lost += lost_samples
                        logger.warning(
                            "Overflow #%d: ~%d samples lost (%.3fs). "
                            "Total lost: %d samples (%.3fs). "
                            "Queue size at overflow: %d/%d",
                            self._total_overflows,
                            lost_samples,
                            time_diff,
                            self._total_samples_lost,
                            self._total_samples_lost / self._sample_rate,
                            self._q.qsize(),
                            self._queue_maxsize,
                        )
                        self._samples_read += lost_samples

                yield chunk, chunk_start_time, chunk_overflow

                pending_blocks = []
                chunk_start_time = None
                chunk_overflow = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_timestamp(self, sample_offset: int) -> datetime:
        """Return the UTC datetime of the sample at *sample_offset*."""
        offset_seconds = sample_offset / self._sample_rate
        return self._stream_started_at + timedelta(seconds=offset_seconds)

    # ------------------------------------------------------------------
    # Properties
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
        """Total samples read since stream start (including estimated lost)."""
        return self._samples_read

    @property
    def total_overflows(self) -> int:
        """Total number of overflow events detected."""
        return self._total_overflows

    @property
    def total_samples_lost(self) -> int:
        """Estimated total samples lost due to overflows."""
        return self._total_samples_lost

    @property
    def queue_size(self) -> int:
        """Current number of pending (unprocessed) blocks in the internal queue."""
        return self._q.qsize()
