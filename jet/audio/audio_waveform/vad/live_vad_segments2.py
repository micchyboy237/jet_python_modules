# live_vad_segments.py

"""
Live VAD Segmenter — streams mic audio via sounddevice, accumulates speech
chunks, and fires on_segment() when a natural boundary is detected.
Optionally saves results to disk in the same layout as vad_firered_hybrid.py.

Segment-end triggers (in priority order):
  1. "silence"       — prob stayed below threshold for >= min_silence_sec
  2. "soft_silence"  — past soft_limit AND currently in silence
  3. "valley"        — past soft_limit AND a clean valley trough was found
  4. "hard_limit"    — past hard_limit (safety fallback, no trough needed)
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
    DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    DEFAULT_SOFT_LIMIT_SEC,
    DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    DEFAULT_THRESHOLD,
)
from jet.audio.audio_waveform.vad.vad_firered_hybrid import FireRedVAD

# ── project imports ────────────────────────────────────────────────────────────
from jet.audio.audio_waveform.vad.vad_utils import save_segments
from jet.audio.helpers.config import (
    FRAME_SHIFT_MS,
    FRAME_SHIFT_S,
    FRAME_SHIFT_SAMPLE,
    SAMPLE_RATE,
)
from jet.audio.speech.vad_extractors import get_best_valley_trough
from jet.audio.speech.vad_types import StreamVadFrame, ValleyTrough
from rich.console import Console

console = Console()

# ── types ──────────────────────────────────────────────────────────────────────

SegmentCallback = Callable[
    [np.ndarray, List[float], str, float],  # audio, probs, reason, duration_s
    None,
]


class _State(Enum):
    IDLE = auto()
    SPEECH = auto()


@dataclass
class SegmentInfo:
    """Metadata passed alongside the audio to on_segment."""

    duration_s: float
    num_frames: int
    end_reason: str  # "silence" | "soft_silence" | "valley" | "hard_limit"
    valley_time_s: Optional[float]  # set when reason == "valley"
    valley_score: Optional[float]
    speech_ratio: float  # fraction of frames above threshold


# ── constants ──────────────────────────────────────────────────────────────────

# How often (in frames) we run valley detection while past the soft limit.
# 10 frames = 100 ms between checks — avoids running scipy every chunk.
_VALLEY_CHECK_INTERVAL_FRAMES = 10


def linkify(path: Path):
    # Provide clickable file link with basename (for rich/terminal that support it)
    return f"[bold blue][link=file://{path}]{path.name}[/link][/bold blue]"


# ── disk persistence (mirrors vad_firered_hybrid.save_segments) ────────────────


def save_live_segment(
    seg_num: int,
    audio_np: np.ndarray,
    probs: List[float],
    reason: str,
    duration_s: float,
    output_dir: Path,
    trough: Optional[ValleyTrough] = None,
    show_progress: bool = True,
) -> Path:
    """
    Build a SpeechSegment from live VAD data and delegate to save_segments.
    Returns the segment directory (output_dir/segments/segment_NNN).
    """
    n_frames = len(probs)
    segment: SpeechSegment = {
        "num": seg_num,
        "start": 0.0,
        "end": duration_s,
        "prob": float(max(probs)) if probs else 0.0,
        "duration": duration_s,
        "frames_length": n_frames,
        "frame_start": 0,
        "frame_end": max(n_frames - 1, 0),
        "type": "speech",
        "segment_probs": list(probs),
    }

    save_segments(
        segments=[segment],
        audio_chunks=[audio_np],
        output_base_dir=output_dir,
        show_progress=show_progress,
    )

    seg_dir = output_dir / "segments" / f"segment_{seg_num:03d}"

    if trough is not None:
        trough_path = seg_dir / "best_trough.json"
        with open(trough_path, "w", encoding="utf-8") as fh:
            json.dump(trough, fh, ensure_ascii=False, indent=2)

        console.print(
            f"[dim magenta]  🕳 Saved trough → {linkify(trough_path)}[/dim magenta]"
        )

    return seg_dir


def save_session_summary(
    all_meta: List[dict],
    output_dir: Path,
) -> None:
    """Write all_speech_segments.json to *output_dir*, same as vad_firered_hybrid."""
    summary_path = output_dir / "all_speech_segments.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(all_meta, fh, ensure_ascii=False, indent=2)
    console.print(
        f"[bold green]✓ Session summary:[/bold green] {linkify(summary_path)}"
    )


# ── logging helpers ────────────────────────────────────────────────────────────

_LAST_LOG: dict[str, float] = {}


def _throttled(key: str, interval_s: float = 2.0) -> bool:
    """Return True if we should emit a log for *key* (rate-limited)."""
    now = time.monotonic()
    if now - _LAST_LOG.get(key, 0.0) >= interval_s:
        _LAST_LOG[key] = now
        return True
    return False


def _log_listening() -> None:
    if _throttled("idle", 15.0):
        console.print("[dim cyan]🎙  Listening…[/dim cyan]")


def _log_speech_start(prob: float) -> None:
    console.print(
        f"[bold green]▶  Speech started[/bold green]  [dim](prob={prob:.2f})[/dim]"
    )


def _log_accumulating(duration_s: float, prob: float) -> None:
    if _throttled("accum", 1.5):
        console.print(
            f"[cyan]   ↳ accumulating[/cyan]  "
            f"[yellow]{duration_s:.1f}s[/yellow]  "
            f"[dim]prob={prob:.2f}[/dim]"
        )


def _log_soft_limit_check(duration_s: float, soft_limit_s: float) -> None:
    if _throttled("soft", 2.0):
        console.print(
            f"[magenta]   ⚠  Past soft limit[/magenta]  "
            f"[yellow]{duration_s:.1f}s[/yellow] > "
            f"[yellow]{soft_limit_s:.1f}s[/yellow]  — watching for valley…"
        )


def _log_segment_end(reason: str, duration_s: float, extra: str = "") -> None:
    color = {
        "silence": "green",
        "soft_silence": "yellow",
        "valley": "magenta",
        "hard_limit": "red",
    }.get(reason, "white")
    console.print(
        f"[bold {color}]■  Segment ended[/bold {color}]  "
        f"reason=[bold]{reason}[/bold]  "
        f"dur=[bold yellow]{duration_s:.2f}s[/bold yellow]"
        + (f"  {extra}" if extra else "")
    )


def _log_valley_found(time_s: float, score: float) -> None:
    console.print(
        f"[magenta]   🔍 Valley trough found[/magenta]  "
        f"at [bold]{time_s:.2f}s[/bold]  "
        f"score=[bold]{score:.3f}[/bold]"
    )


def _log_no_valley(duration_s: float) -> None:
    if _throttled("no_valley", 2.0):
        console.print(
            f"[dim yellow]   ⏳ No valley found yet at {duration_s:.1f}s — continuing…[/dim yellow]"
        )


# ── main class ─────────────────────────────────────────────────────────────────


class LiveVADSegmenter:
    """
    Real-time microphone segmenter using FireRedVAD.

    Parameters
    ----------
    on_segment:
        Callback fired at the end of each speech segment::

            on_segment(audio_np, probs, reason, duration_s)

        *audio_np*   – float32 array of accumulated speech samples
        *probs*      – list of per-frame VAD probabilities
        *reason*     – one of "silence", "soft_silence", "valley", "hard_limit"
        *duration_s* – segment duration in seconds

    threshold:
        VAD speech probability threshold (default 0.5).

    min_silence_sec:
        Consecutive silence required to close a segment (default 0.8 s).

    soft_limit_sec:
        After this many seconds the segmenter actively looks for a valley
        to cut early (default 6 s).

    hard_limit_sec:
        Absolute max segment length — fires even without a valley (default 15 s).

    device:
        sounddevice input device index or name.  None → system default.

    All ``valley_*`` kwargs are forwarded directly to
    ``get_best_valley_trough`` and mirror the defaults in
    ``vad_firered_hybrid.py``.
    """

    def __init__(
        self,
        on_segment: SegmentCallback,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_sec: float = DEFAULT_MIN_SILENCE_SEC,
        soft_limit_sec: float = DEFAULT_SOFT_LIMIT_SEC,
        hard_limit_sec: float = 15.0,
        # valley-detection knobs (reuse vad_firered_hybrid defaults)
        valley_min_duration_s: float = DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
        valley_smoothing_window: int = DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
        valley_trough_prominence: float = DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
        valley_min_trough_offset_s: float = DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
        # sounddevice
        device: Optional[int | str] = None,
        # optional disk output (None = don't save)
        output_dir: Optional[Path] = None,
        # VAD model
        vad: Optional[FireRedVAD] = None,
    ) -> None:
        self._cb = on_segment
        self._threshold = threshold
        self._min_silence_frames = int(min_silence_sec / FRAME_SHIFT_S)
        self._soft_limit_frames = int(soft_limit_sec / FRAME_SHIFT_S)
        self._hard_limit_frames = int(hard_limit_sec / FRAME_SHIFT_S)
        self._soft_limit_sec = soft_limit_sec

        # valley kwargs forwarded verbatim
        self._valley_kwargs = dict(
            min_valley_duration_s=valley_min_duration_s,
            smoothing_window=valley_smoothing_window,
            trough_prominence=valley_trough_prominence,
            min_trough_offset_s=valley_min_trough_offset_s,
            frame_shift_ms=float(FRAME_SHIFT_MS),
            sample_rate=SAMPLE_RATE,
        )

        self._device = device

        # output
        self._output_dir = output_dir
        self._seg_counter = 0
        self._session_meta: List[dict] = []

        # build or reuse VAD
        self._vad: FireRedVAD = vad or FireRedVAD(threshold=threshold)

        # state
        self._state = _State.IDLE
        self._segment_audio: List[np.ndarray] = []  # only current segment
        self._silence_frames: int = 0
        self._onset_frames: int = 0
        self._frames_since_valley_check: int = 0
        # Offset into vad._accumulated_probs marking the start of the current segment.
        # Advances on every emit so we never re-use probs from past segments.
        self._prob_offset: int = 0

        # thread safety — sounddevice calls the callback from a C thread
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()

        # Queue mirrors AudioStreamManager's pattern: decouples the real-time
        # SD callback from all heavy VAD / state-machine work.
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self._worker_thread: Optional[threading.Thread] = None

        self._stream: Optional[sd.InputStream] = None

    # ── public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone stream and begin processing."""
        self._vad.reset()
        self._seg_counter = 0
        self._session_meta.clear()
        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        self._reset_segment()

        self._worker_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._worker_thread.start()

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SHIFT_SAMPLE,  # one 10 ms frame per callback — matches VAD input size
            device=self._device,
            callback=self._sd_callback,
        )
        self._stream.start()
        console.print(
            f"[bold cyan]LiveVADSegmenter started[/bold cyan]  "
            f"block={FRAME_SHIFT_SAMPLE} samples (one frame)"
            f"({FRAME_SHIFT_MS} ms)  "
            f"sr={SAMPLE_RATE}"
        )

    def stop(self) -> None:
        """Stop the microphone stream gracefully."""
        # Signal worker to exit before closing the stream.
        # PATCH: Do not put None into the queue, since the type is np.ndarray
        # Instead, to truly stop, mirror how a worker thread should exit safely.
        try:
            self._audio_queue.put_nowait(
                np.zeros(FRAME_SHIFT_SAMPLE, dtype=np.float32) * np.nan
            )  # marker impossible in normal audio
        except queue.Full:
            pass
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        console.print("[bold red]LiveVADSegmenter stopped.[/bold red]")

    def run_forever(self) -> None:
        """Block until KeyboardInterrupt, then stop cleanly."""
        self.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # ── internal ───────────────────────────────────────────────────────────────

    def _reset_segment(self) -> None:
        self._segment_audio.clear()
        self._silence_frames = 0
        self._onset_frames = 0
        self._frames_since_valley_check = 0

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,  # noqa: ANN001 — CData struct, not typed
        status: sd.CallbackFlags,
    ) -> None:
        """Real-time SD callback — only enqueues; never does heavy work."""
        if status:
            console.print(f"[red]sounddevice status: {status}[/red]")

        chunk = indata[:, 0].copy()  # mono, shape (FRAME_SHIFT_SAMPLE,)

        # Mirror AudioStreamManager's drop-oldest strategy.
        if self._audio_queue.full():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            pass

    def _processing_loop(self) -> None:
        """Worker thread — drains the queue and runs the VAD state machine."""
        while True:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # PATCH: Use a nan-filled chunk as the stop marker (see stop)
            if np.isnan(chunk).all():
                break
            with self._lock:
                self._process_chunk(chunk)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Core per-frame logic."""
        # Let VAD handle normalization + hybrid prob + accumulation
        prob = self._vad.get_speech_prob(chunk)
        is_speech = prob >= self._threshold

        if self._state is _State.IDLE:
            _log_listening()
            if is_speech:
                self._state = _State.SPEECH
                _log_speech_start(prob)
                # Only seed the segment buffer when speech actually starts.
                self._segment_audio = [chunk]
            # Non-speech idle frames are discarded — they must not bleed into
            # the next segment's audio or prob slice.
            return

        # ── SPEECH state ─────────────────────────────────────────────────────
        self._segment_audio.append(chunk)
        # Use the offset so n_frames counts only this segment's frames.
        n_frames = len(self._vad._accumulated_probs) - self._prob_offset
        duration_s = n_frames * FRAME_SHIFT_S

        if is_speech:
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        _log_accumulating(duration_s, prob)

        in_silence = self._silence_frames >= self._min_silence_frames
        past_soft = n_frames >= self._soft_limit_frames
        past_hard = n_frames >= self._hard_limit_frames

        # ── 1. hard limit (safety net) ─────────────────────────────────────
        if past_hard:
            self._emit("hard_limit", valley_time_s=None, valley_score=None, trough=None)
            return

        # ── 2. plain silence (not yet at soft limit) ───────────────────────
        if in_silence and not past_soft:
            self._emit("silence", valley_time_s=None, valley_score=None, trough=None)
            return

        # ── 3. past soft limit ─────────────────────────────────────────────
        if past_soft:
            _log_soft_limit_check(duration_s, self._soft_limit_sec)

            # 3a. silence while past soft limit
            if in_silence:
                self._emit(
                    "soft_silence", valley_time_s=None, valley_score=None, trough=None
                )
                return

            # 3b. valley trough check (throttled)
            self._frames_since_valley_check += 1
            if self._frames_since_valley_check >= _VALLEY_CHECK_INTERVAL_FRAMES:
                self._frames_since_valley_check = 0

                # Use accumulated probs from VAD (already hybrid!)
                history: List[StreamVadFrame] = self._vad.get_accumulated_probs()
                # Slice from the segment start offset, not from the tail.
                current_probs = [
                    p["smoothed_prob"]
                    for p in history[self._prob_offset : self._prob_offset + n_frames]
                ]

                trough = get_best_valley_trough(
                    probs=current_probs, **self._valley_kwargs
                )

                if trough is not None:
                    cut_frame = trough["frame"]
                    self._segment_audio = self._segment_audio[:cut_frame]
                    # Advance offset by the number of frames we're consuming.
                    self._prob_offset += cut_frame
                    self._emit(
                        "valley",
                        valley_time_s=trough["time_s"],
                        valley_score=trough["valley"].get("final_score", 0.0),
                        trough=trough,
                    )
                    return
                else:
                    _log_no_valley(duration_s)

    def _emit(
        self,
        reason: str,
        *,
        valley_time_s: Optional[float],
        valley_score: Optional[float],
        trough: Optional[dict] = None,
    ) -> None:
        if not self._segment_audio:
            self._reset_segment()
            self._state = _State.IDLE
            return

        audio_np = np.concatenate(self._segment_audio, axis=0)
        history = self._vad.get_accumulated_probs()
        seg_len = len(self._segment_audio)
        probs = [
            p["smoothed_prob"]
            for p in history[self._prob_offset : self._prob_offset + seg_len]
        ]

        duration_s = len(audio_np) / SAMPLE_RATE
        above = sum(1 for p in probs if p >= self._threshold)
        speech_ratio = above / max(len(probs), 1)

        self._seg_counter += 1
        seg_num = self._seg_counter

        _log_segment_end(
            reason,
            duration_s,
            extra=(
                f"valley_t={valley_time_s:.2f}s  score={valley_score:.3f}"
                if valley_time_s is not None
                else f"frames={len(probs)}  speech_ratio={speech_ratio:.0%}"
            ),
        )

        # Capture trough before resetting state for closure
        trough_to_save = trough

        self._state = _State.IDLE
        self._reset_segment()
        # Advance the prob offset past the frames we just consumed.
        # For non-valley emits (silence, hard_limit) we advance by the full segment.
        if reason != "valley":
            self._prob_offset += seg_len

        # ── optional disk save (done outside lock, in a daemon thread) ────
        if self._output_dir is not None:
            out_dir = self._output_dir

            def _save() -> None:
                try:
                    seg_dir = save_live_segment(
                        seg_num,
                        audio_np,
                        probs,
                        reason,
                        duration_s,
                        out_dir,
                        trough=trough_to_save,
                    )

                    meta = {
                        "num": seg_num,
                        "duration_s": duration_s,
                        "end_reason": reason,
                        "output_path": str(
                            (seg_dir / "sound.wav").relative_to(out_dir)
                        ),
                    }
                    with self._lock:
                        self._session_meta.append(meta)
                    with self._save_lock:
                        save_session_summary(self._session_meta, out_dir)
                    console.print(
                        f"[dim green]  💾 Saved segment {seg_num:03d} → "
                        f"{linkify(seg_dir)}[/dim green]"
                    )
                except Exception as exc:
                    console.print(f"[red]  Save failed (seg {seg_num}): {exc}[/red]")

            threading.Thread(target=_save, daemon=True).start()

        # fire callback outside the lock to avoid deadlocks
        try:
            self._cb(audio_np, probs, reason, duration_s)
        except Exception as exc:
            console.print(f"[red]on_segment callback raised: {exc}[/red]")


# ── demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEFAULT_OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Live VAD segmenter — streams mic and saves speech segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save segments (same layout as vad_firered_hybrid)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable disk output (callback-only mode)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="VAD speech probability threshold",
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default=DEFAULT_MIN_SILENCE_SEC,
        help="Consecutive silence (s) required to close a segment",
    )
    parser.add_argument(
        "--soft-limit",
        type=float,
        default=DEFAULT_SOFT_LIMIT_SEC,
        help="Soft segment duration limit before valley search (s)",
    )
    parser.add_argument(
        "--hard-limit",
        type=float,
        default=15.0,
        help="Hard maximum segment duration (s)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="sounddevice input device index or name",
    )
    args = parser.parse_args()

    output_dir = None if args.no_save else args.output_dir
    if output_dir:
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]Output dir:[/cyan] {output_dir}")

    seg_count = 0

    def on_segment(  # noqa: F811
        audio: np.ndarray,
        probs: List[float],
        reason: str,
        duration_s: float,
    ) -> None:
        global seg_count
        seg_count += 1
        above = sum(1 for p in probs if p >= DEFAULT_THRESHOLD)
        console.rule(
            f"[bold green]Segment #{seg_count}[/bold green]  "
            f"reason=[bold]{reason}[/bold]  "
            f"{duration_s:.2f}s  "
            f"speech={above}/{len(probs)} frames",
            style="green",
        )

    segmenter = LiveVADSegmenter(
        on_segment=on_segment,
        threshold=args.threshold,
        min_silence_sec=args.min_silence,
        soft_limit_sec=args.soft_limit,
        hard_limit_sec=args.hard_limit,
        device=args.device,
        output_dir=output_dir,
    )
    console.print("[bold]Press Ctrl+C to stop.[/bold]")
    segmenter.run_forever()
