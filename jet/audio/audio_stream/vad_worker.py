"""VadSegmentWorker — runs the expensive VAD/segment-extraction pass on a
dedicated background thread, decoupled from the real-time audio drain loop.

PROBLEM THIS SOLVES
--------------------
`extract_current_speech_segment` re-runs VAD over the *entire* sliding
window (up to `buffer_max_sec` seconds) every time it's called. If this
happens inline in the same loop that drains `FireredStream`'s internal
queue, a slow VAD pass blocks chunk consumption — `FireredStream`'s queue
(bounded, `queue_maxsize`) fills up, and new audio blocks get silently
dropped in the real-time callback thread. That packet loss then shows up
downstream as "Audio gap detected" warnings from `CircularAudioBuffer`.

FIX
---
Same callback/queue pattern `FireredStream` already uses internally,
applied one layer up:
    - the audio-drain loop (caller) stays cheap: append to buffer, detect
      gaps/silence, signal "new audio available" — never blocks on VAD.
    - this worker thread does the expensive VAD/segment-extraction pass
      independently, picking up the latest buffer state whenever it's
      free, and pushes completed (segment, audio) tuples to a result
      queue that the caller drains without blocking.
This makes VAD cost a non-issue for real-time audio capture: however slow
VAD gets, the drain loop never waits on it, so the PortAudio callback
queue (and therefore captured audio) is never starved.
"""

from __future__ import annotations

import queue
import threading
from typing import List, Optional, Tuple

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_SOFT_LIMIT_SEC,
    DEFAULT_THRESHOLD,
)
from jet.audio.audio_waveform.vad.vad_speech_segments_extractor import (
    extract_speech_timestamps,
)
from jet.audio.helpers.base import get_audio_duration
from jet.audio.helpers.circular_audio_buffer import CircularAudioBuffer
from jet.audio.normalization.norm_speech_loudness import normalize_audio_for_vad
from jet.audio.normalization.quant import quantize_audio
from jet.audio.speech.utils import display_segments
from rich.console import Console

console = Console()

SegmentResult = Tuple[SpeechSegment, np.ndarray]

_WAKE_POLL_TIMEOUT_SEC = 0.5


def extract_segment_data(
    segment: SpeechSegment,
    audio_np: CircularAudioBuffer,
    trim_silent: bool = True,
    silence_threshold: Optional[float] = None,
) -> np.ndarray:
    """Extract the audio for *segment* from the circular buffer.

    VAD timestamps are relative to the start of ``audio_np``'s current window,
    so we can pass them directly to ``slice_seconds`` without any index math.
    """
    return audio_np.slice_seconds(
        start_sec=float(segment["start"]),
        end_sec=float(segment["end"]),
        trim_silent=trim_silent,
        silence_threshold=silence_threshold,
    )


def extract_current_speech_segment(
    audio_data: CircularAudioBuffer,
    threshold: float = DEFAULT_THRESHOLD,
    min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
    min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
    max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
    verbose: bool = False,
) -> List[SpeechSegment]:
    """Run VAD over the entire current buffer window.

    EXPENSIVE — this is exactly the call `VadSegmentWorker` keeps off the
    real-time drain thread. Uses ``to_numpy()`` (lock-protected) so it's
    safe to call concurrently with the drain thread's ``append()``.
    """
    full_audio_np = audio_data.to_numpy()

    # Early silence check - skip quantization if silent
    if full_audio_np.size == 0 or np.max(np.abs(full_audio_np)) == 0:
        return []

    # Normalize audio for better VAD detection
    full_audio_np, _ = normalize_audio_for_vad(full_audio_np, audio_data.sample_rate)
    duration = get_audio_duration(full_audio_np, audio_data.sample_rate)
    if duration >= DEFAULT_SOFT_LIMIT_SEC:
        full_audio_np, _ = quantize_audio(
            full_audio_np,
            target_dtype="float16",
            sr=audio_data.sample_rate,
            verbose=verbose,
        )

    curr_speech_segs, _speech_probs = extract_speech_timestamps(
        audio=full_audio_np,
        with_scores=True,
        return_seconds=True,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        soft_limit_sec=DEFAULT_SOFT_LIMIT_SEC,
    )
    return curr_speech_segs


def _enrich_segment_with_timestamps(
    segment: SpeechSegment,
    audio_buffer: CircularAudioBuffer,
) -> None:
    """Add absolute UTC timestamps to a speech segment in-place."""
    segment["start_time_utc"] = audio_buffer.get_absolute_time(float(segment["start"]))
    segment["end_time_utc"] = audio_buffer.get_absolute_time(float(segment["end"]))


class VadSegmentWorker:
    """Background thread that owns all VAD/segment-boundary logic.

    The caller (the real-time audio drain loop) only ever calls the
    cheap, non-blocking methods: ``notify_new_audio()``, ``notify_silence()``,
    ``notify_overflow()``, and ``poll_results()``. All expensive work —
    re-running VAD over the buffer, extracting segment audio, enriching
    timestamps, trimming the buffer — happens here, off the drain loop's
    critical path.
    """

    def __init__(
        self,
        audio_data: CircularAudioBuffer,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
        min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
        max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
        overlap_seconds: float = 0.0,
        trim_silent: bool = False,
        silence_threshold: Optional[float] = None,
        trim_overlap_sec: float = 0.3,
        verbose: bool = False,
    ) -> None:
        self._audio_data = audio_data
        self._threshold = threshold
        self._min_silence_duration_sec = min_silence_duration_sec
        self._min_speech_duration_sec = min_speech_duration_sec
        self._max_speech_duration_sec = max_speech_duration_sec
        self._overlap_seconds = overlap_seconds
        self._trim_silent = trim_silent
        self._silence_threshold = silence_threshold
        self._trim_overlap_sec = trim_overlap_sec
        self._verbose = verbose

        self._wake_event = threading.Event()
        self._flush_event = threading.Event()
        self._stop_event = threading.Event()
        self._results_q: "queue.Queue[SegmentResult]" = queue.Queue()
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vad-segment-worker"
        )

        # State formerly local to the synchronous loop in speech_detector.py —
        # now owned by the worker since it's the only thread that runs VAD /
        # decides segment boundaries.
        self._curr_segment: Optional[SpeechSegment] = None
        self._prev_segment: Optional[SpeechSegment] = None
        self._last_yielded_end_sec: float = 0.0
        self._curr_speech_segs: List[SpeechSegment] = []
        self._consecutive_empty_yields: int = 0

        self._pending_overflow_lock = threading.Lock()
        self._pending_overflow: bool = False

        # Stats for tracing.
        self._vad_passes: int = 0
        self._segments_emitted: int = 0

    # ---- public API, called from the drain loop (must stay cheap) ----

    def start(self) -> None:
        self._thread.start()

        if self._verbose:
            console.print(
                f"[green]VadSegmentWorker started (tid={self._thread.ident})[/green]"
            )

    def notify_new_audio(self) -> None:
        """Wake the worker to re-run VAD on the latest buffer state.
        Cheap: just sets an Event. Safe to call once per drained chunk —
        wakes that arrive while the worker is still busy coalesce into a
        single subsequent pass, since the worker always reads the
        *current* buffer state, not a queued snapshot."""
        self._wake_event.set()

    def notify_silence(self) -> None:
        """Ask the worker to force-flush any pending in-progress segment
        because sustained silence was detected. Cheap + idempotent."""
        self._flush_event.set()
        self._wake_event.set()

    def notify_overflow(self) -> None:
        """Mark that a PortAudio overflow occurred. The next segment the
        worker emits will be tagged ``had_overflow=True``."""
        with self._pending_overflow_lock:
            self._pending_overflow = True

    def poll_results(self) -> List[SegmentResult]:
        """Non-blocking drain of every segment completed so far."""
        results: List[SegmentResult] = []
        while True:
            try:
                results.append(self._results_q.get_nowait())
            except queue.Empty:
                break
        return results

    def flush_final_segment(self) -> None:
        """Emit any trailing in-progress segment. Only safe to call after
        the worker thread has fully stopped (see ``stop_and_drain``) —
        avoids racing with ``_handle_vad_pass``/``_handle_flush`` over
        shared state like ``_prev_segment``."""
        prev_segment = self._prev_segment
        if prev_segment is None:
            return
        if self._verbose:
            display_segments(self._curr_speech_segs)
        last_yielded_window_sec = max(
            0.0, self._last_yielded_end_sec - self._audio_data.trimmed_sec
        )
        effective_start_sec = max(
            last_yielded_window_sec,
            float(prev_segment["start"]) - self._overlap_seconds,
        )
        if effective_start_sec >= float(prev_segment["end"]):
            if self._verbose:
                console.print(
                    "[yellow]VadSegmentWorker: skipping final segment yield due to overlap causing empty audio[/yellow]"
                )
            self._prev_segment = None
            return
        prev_segment["start"] = effective_start_sec
        seg_audio_np = self._extract(prev_segment)
        if seg_audio_np is None or seg_audio_np.size == 0:
            self._on_empty_segment(prev_segment)
        else:
            self._consecutive_empty_yields = 0
            self._enrich_and_emit(prev_segment, seg_audio_np)
        self._prev_segment = None

    def stop_and_drain(self, timeout: float = 10.0) -> List[SegmentResult]:
        """Signal shutdown and wait for the thread to exit. Call this
        BEFORE ``flush_final_segment()`` so no two threads touch worker
        state concurrently. Returns any results produced so far."""
        self._stop_event.set()
        self._wake_event.set()  # unblock a pending wait()
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            if self._verbose:
                console.print(
                    f"[yellow]VadSegmentWorker did not exit within {timeout:.1f}s — abandoning thread[/yellow]"
                )
        if self._error is not None:
            if self._verbose:
                console.print(
                    f"[red]VadSegmentWorker exited with error: {self._error}[/red]"
                )
        if self._verbose:
            console.print(
                f"[green]VadSegmentWorker stopped: {self._vad_passes} VAD pass(es), {self._segments_emitted} segment(s) emitted[/green]"
            )
        return self.poll_results()

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    # ---- internal: runs entirely on the worker thread ----

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._flush_event.is_set():
                    self._flush_event.clear()
                    self._handle_flush()
                woke = self._wake_event.wait(timeout=_WAKE_POLL_TIMEOUT_SEC)
                if self._stop_event.is_set():
                    break
                if not woke:
                    continue
                self._wake_event.clear()
                self._handle_vad_pass()
        except Exception as exc:  # noqa: BLE001 - surface to caller thread
            self._error = exc

            if self._verbose:
                console.print(f"[red]VadSegmentWorker crashed: {exc}[/red]")

    def _handle_flush(self) -> None:
        """Force-yield ``prev_segment`` (extended silence detected)."""
        prev_segment = self._prev_segment
        if prev_segment is None:
            return
        seg_audio_np = self._extract(prev_segment)
        if seg_audio_np is None or seg_audio_np.size == 0:
            self._on_empty_segment(prev_segment)
        else:
            self._consecutive_empty_yields = 0
            self._enrich_and_emit(prev_segment, seg_audio_np)
            self._audio_data.trim_to_sec(float(prev_segment["end"]))

            if self._verbose:
                console.print(
                    f"[blue]VadSegmentWorker: flushed on silence, trimmed to {prev_segment['end']:.3f}s. "
                    f"Buffer now: {self._audio_data.window_sec:.3f}s, gaps: {self._audio_data.total_gap_sec:.3f}s[/blue]"
                )
        self._prev_segment = None
        if self._curr_speech_segs and self._verbose:
            display_segments(self._curr_speech_segs, done=True)

    def _handle_vad_pass(self) -> None:
        self._vad_passes += 1
        self._curr_speech_segs = extract_current_speech_segment(
            self._audio_data,
            threshold=self._threshold,
            min_silence_duration_sec=self._min_silence_duration_sec,
            min_speech_duration_sec=self._min_speech_duration_sec,
            max_speech_duration_sec=self._max_speech_duration_sec,
            verbose=self._verbose,
        )
        if self._verbose:
            display_segments(self._curr_speech_segs)

        self._curr_segment = (
            self._curr_speech_segs[-1] if self._curr_speech_segs else self._prev_segment
        )
        self._prev_segment = (
            self._curr_speech_segs[-2] if len(self._curr_speech_segs) > 1 else None
        )

        if not (
            self._curr_segment
            and self._prev_segment
            and self._curr_segment["start"] != self._prev_segment["start"]
        ):
            self._prev_segment = self._curr_segment
            return

        prev_segment = self._prev_segment
        last_yielded_window_sec = max(
            0.0, self._last_yielded_end_sec - self._audio_data.trimmed_sec
        )
        effective_start_sec = max(
            last_yielded_window_sec,
            float(prev_segment["start"]) - self._overlap_seconds,
        )
        if effective_start_sec >= float(prev_segment["end"]):
            self._prev_segment = self._curr_segment
            return

        original_start = prev_segment["start"]
        prev_segment["start"] = effective_start_sec
        seg_audio_np = self._extract(prev_segment)
        prev_segment["start"] = original_start
        prev_segment["duration"] = prev_segment["end"] - prev_segment["start"]

        if seg_audio_np is None or seg_audio_np.size == 0:
            self._on_empty_segment(prev_segment)
            self._prev_segment = self._curr_segment
            return

        self._consecutive_empty_yields = 0
        self._last_yielded_end_sec = (
            float(prev_segment["end"]) + self._audio_data.trimmed_sec
        )
        self._enrich_and_emit(prev_segment, seg_audio_np)
        trim_point = max(0.0, float(prev_segment["end"]) - self._trim_overlap_sec)
        self._audio_data.trim_to_sec(trim_point)

        if self._verbose:
            console.print(
                f"[blue]VadSegmentWorker: trimmed to {trim_point:.3f}s (segment end={prev_segment['end']:.3f}s, overlap={self._trim_overlap_sec:.3f}s). "
                f"Buffer now: {self._audio_data.window_sec:.3f}s, gaps: {self._audio_data.total_gap_sec:.3f}s[/blue]"
            )
        self._prev_segment = self._curr_segment

    def _extract(self, segment: SpeechSegment) -> np.ndarray:
        return extract_segment_data(
            segment,
            self._audio_data,
            trim_silent=self._trim_silent,
            silence_threshold=self._silence_threshold,
        )

    def _enrich_and_emit(
        self, segment: SpeechSegment, seg_audio_np: np.ndarray
    ) -> None:
        _enrich_segment_with_timestamps(segment, self._audio_data)
        with self._pending_overflow_lock:
            if self._pending_overflow:
                segment["had_overflow"] = True
                self._pending_overflow = False
        self._segments_emitted += 1
        self._results_q.put((segment, seg_audio_np))

        if self._verbose:
            console.print(
                f"[blue]VadSegmentWorker: emitted segment [{float(segment['start']):.3f}, {float(segment['end']):.3f}] had_overflow={segment.get('had_overflow', False)}[/blue]"
            )

    def _on_empty_segment(self, segment: SpeechSegment) -> None:
        self._consecutive_empty_yields += 1

        if self._verbose:
            console.print(
                f"[yellow]VadSegmentWorker: empty audio for segment at [{float(segment['start']):.3f}, {float(segment['end']):.3f}]. "
                f"Buffer: {self._audio_data.window_sec:.3f}s, Gaps: {self._audio_data.total_gap_sec:.3f}s. Consecutive empty: {self._consecutive_empty_yields}[/yellow]"
            )
        if self._consecutive_empty_yields > 3:
            if self._verbose:
                console.print(
                    "[red]VadSegmentWorker: too many empty segments - possible stream issue![/red]"
                )
