"""
live_vad_segments.py
════════════════════
Bridges a live sounddevice microphone stream into VadSpeechSegmentsTracker,
saving result files for every completed segment and calling an optional
on_segment callback.

Saved per segment  (OUTPUT_DIR / "segments" / "segment_NNN/")
─────────────────
  sound.wav           – 16-kHz PCM-16 audio
  meta.json           – SpeechSegment metadata + probs_info summary
  raw_probs.json      – per-frame raw model probabilities
  smoothed_probs.json – per-frame smoothed probabilities used for VAD
  energies.json       – per-frame RMS energy
  plot.png            – speech prob + RMS + hybrid overlay plot

Threading model
───────────────
  sounddevice callback  →  queue.Queue  →  worker thread
                                              ├─ FireRedVAD.get_speech_prob()
                                              ├─ VadSpeechSegmentsTracker.push()
                                              ├─ _save_segment()    (on result)
                                              └─ on_segment(result) (on result)

The sounddevice callback ONLY does `queue.put_nowait(chunk)` — no blocking,
no Python-level computation.  All VAD and I/O live in the worker thread.
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib
from jet.audio.audio_waveform.vad.vad_logging import log_accumulating

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    DEFAULT_POSTROLL_MAX_SEC,
    DEFAULT_PREROLL_HYBRID_THRESHOLD,
    DEFAULT_PREROLL_MAX_SEC,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RMS_WEIGHT,
    DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
    DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
    DEFAULT_SOFT_LIMIT_SEC,
    DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
    DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
    DEFAULT_THRESHOLD,
)
from jet.audio.audio_waveform.vad.vad_firered_hybrid import FireRedVAD
from jet.audio.audio_waveform.vad.vad_speech_segments_tracker import (
    SegmentResult,
    VadSpeechSegmentsTracker,
)
from jet.audio.helpers.config import FRAME_SHIFT_MS, HOP_SIZE, HOP_STEP_S, SAMPLE_RATE
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

# ── Sentinel: signals worker thread to stop ────────────────────────────────────
_STOP = object()


# ─────────────────────────────────────────────────────────────────────────────
# Segment file saver
# ─────────────────────────────────────────────────────────────────────────────


def _compute_frame_rms(audio_np: np.ndarray, frame_size: int = HOP_SIZE) -> np.ndarray:
    """Per-frame RMS energy aligned to VAD frames (10 ms each)."""
    n_frames = len(audio_np) // frame_size
    if n_frames == 0:
        return np.array([], dtype=np.float32)
    frames = audio_np[: n_frames * frame_size].reshape(n_frames, frame_size)
    return np.sqrt(np.mean(frames**2, axis=1)).astype(np.float32)


def _save_segment(
    seg_num: int,
    result: SegmentResult,
    segment: SpeechSegment,
    raw_probs: List[float],
    output_dir: Path,
) -> Path:
    """
    Persist all artefacts for one completed speech segment.

    Parameters
    ----------
    seg_num     Global segment counter (used for the directory name).
    result      The SegmentResult from VadSpeechSegmentsTracker.
    segment     The specific SpeechSegment dict to save (result may contain
                multiple when soft-limit splits occurred).
    raw_probs   Raw (un-smoothed) per-frame probabilities for this segment,
                aligned with result.probs (smoothed).
    output_dir  Root output directory; files go under
                output_dir / "segments" / "segment_NNN".

    Returns
    -------
    Path to the segment subdirectory.
    """
    seg_dir = output_dir / "segments" / f"segment_{seg_num:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    audio_np: np.ndarray = result.audio_np
    smoothed_probs: List[float] = result.probs

    # ── 1. sound.wav ──────────────────────────────────────────────────────────
    wav_path = seg_dir / "sound.wav"
    if len(audio_np) > 0:
        try:
            torchaudio.save(
                str(wav_path),
                torch.from_numpy(audio_np).unsqueeze(0),
                SAMPLE_RATE,
                encoding="PCM_S",
                bits_per_sample=16,
            )
        except Exception as exc:
            console.print(f"[red]  WAV save error: {exc}[/red]")
    else:
        console.print(
            f"[yellow]  segment {seg_num:03d}: empty audio — skipping WAV[/yellow]"
        )

    # ── 2. RMS energies ───────────────────────────────────────────────────────
    rms = _compute_frame_rms(audio_np)

    # ── 3. Probs summary ──────────────────────────────────────────────────────
    sp_arr = np.array(smoothed_probs, dtype=np.float32)
    probs_info = {
        "num_frames": int(len(sp_arr)),
        "mean": float(np.mean(sp_arr)) if len(sp_arr) else 0.0,
        "max": float(np.max(sp_arr)) if len(sp_arr) else 0.0,
        "min": float(np.min(sp_arr)) if len(sp_arr) else 0.0,
        "std": float(np.std(sp_arr)) if len(sp_arr) else 0.0,
        "median": float(np.median(sp_arr)) if len(sp_arr) else 0.0,
        "frame_rate_hz": int(1.0 / HOP_STEP_S),
    }

    # ── 4. meta.json ──────────────────────────────────────────────────────────
    meta_out = dict(segment)
    meta_out["output_path"] = str(wav_path.relative_to(output_dir))
    meta_out["end_reason"] = result.end_reason
    meta_out["end_condition_label"] = result.end_condition_label
    meta_out["probs_info"] = probs_info
    meta_out.pop("segment_probs", None)
    with open(seg_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta_out, fh, indent=2, ensure_ascii=False)

    # ── 5. raw_probs.json ─────────────────────────────────────────────────────
    with open(seg_dir / "raw_probs.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "probs": [float(p) for p in raw_probs],
                "frame_shift_sec": HOP_STEP_S,
                "frame_start": segment.get("frame_start", 0),
                "num_frames": len(raw_probs),
            },
            fh,
            indent=2,
        )

    # ── 6. smoothed_probs.json ────────────────────────────────────────────────
    with open(seg_dir / "smoothed_probs.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "probs": smoothed_probs,
                "frame_shift_sec": HOP_STEP_S,
                "frame_start": segment.get("frame_start", 0),
                "summary": probs_info,
            },
            fh,
            indent=2,
        )

    # ── 7. energies.json ──────────────────────────────────────────────────────
    with open(seg_dir / "energies.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "rms": rms.tolist(),
                "frame_shift_sec": HOP_STEP_S,
                "num_frames": int(len(rms)),
            },
            fh,
            indent=2,
        )

    # ── 8. plot.png ───────────────────────────────────────────────────────────
    _save_plot(
        seg_dir=seg_dir,
        seg_num=seg_num,
        smoothed_probs=sp_arr,
        raw_probs=np.array(raw_probs, dtype=np.float32),
        rms=rms,
        threshold=DEFAULT_THRESHOLD,
        end_reason=result.end_reason,
        end_label=result.end_condition_label,
    )

    return seg_dir


def _save_plot(
    seg_dir: Path,
    seg_num: int,
    smoothed_probs: np.ndarray,
    raw_probs: np.ndarray,
    rms: np.ndarray,
    threshold: float,
    end_reason: str,
    end_label: str,
) -> None:
    """Generate a 3-panel plot: raw probs, smoothed probs, RMS + hybrid."""
    n = len(smoothed_probs)
    if n == 0:
        return

    times = np.arange(n) * HOP_STEP_S

    # Hybrid score (prob + RMS)
    n_rms = len(rms)
    n_min = min(n, n_rms)
    if n_min > 0:
        rms_ceil = np.percentile(rms[:n_min], 99) + 1e-10
        rms_norm = np.clip(rms[:n_min] / rms_ceil, 0.0, 1.0)
        hybrid = (0.5 * smoothed_probs[:n_min] + 0.5 * rms_norm).astype(np.float32)
    else:
        hybrid = np.array([], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"Segment {seg_num:03d}  —  {end_reason} [{end_label}]  "
        f"({n * HOP_STEP_S:.2f} s)",
        fontsize=11,
    )

    # Panel 1: raw probs
    ax = axes[0]
    if len(raw_probs) == n:
        ax.fill_between(times, raw_probs, alpha=0.4, color="#7F77DD", label="raw prob")
        ax.plot(times, raw_probs, linewidth=0.8, color="#534AB7")
    ax.axhline(
        threshold,
        color="#E24B4A",
        linewidth=0.8,
        linestyle="--",
        label=f"threshold={threshold}",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Raw prob", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: smoothed probs
    ax = axes[1]
    ax.fill_between(
        times, smoothed_probs, alpha=0.4, color="#1D9E75", label="smoothed prob"
    )
    ax.plot(times, smoothed_probs, linewidth=0.8, color="#0F6E56")
    ax.axhline(threshold, color="#E24B4A", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Smoothed prob", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")

    # Panel 3: RMS + hybrid
    ax = axes[2]
    h_times = np.arange(len(hybrid)) * HOP_STEP_S
    r_times = np.arange(n_min) * HOP_STEP_S
    if n_min > 0:
        ax.fill_between(
            r_times, rms_norm, alpha=0.25, color="#BA7517", label="RMS norm"
        )
        ax.plot(r_times, rms_norm, linewidth=0.6, color="#BA7517")
    if len(hybrid) > 0:
        ax.fill_between(h_times, hybrid, alpha=0.35, color="#D85A30", label="hybrid")
        ax.plot(h_times, hybrid, linewidth=0.8, color="#993C1D")
    ax.axhline(
        DEFAULT_PREROLL_HYBRID_THRESHOLD,
        color="#378ADD",
        linewidth=0.8,
        linestyle=":",
        label=f"hybrid thr={DEFAULT_PREROLL_HYBRID_THRESHOLD}",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Energy / hybrid", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(seg_dir / "plot.png", dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class LiveVadRecorder:
    """
    Records from the microphone in real time, detects speech segments using
    FireRedVAD + VadSpeechSegmentsTracker, saves result files, and fires an
    optional callback for each completed segment.

    Parameters
    ----------
    on_segment
        Called on the worker thread after files are saved.
        Signature: ``on_segment(result: SegmentResult, seg_dir: Path) -> None``
        Keep it fast — heavy work should be dispatched elsewhere.
    output_dir
        Root directory for saved segment folders.
        Default: ``<script_dir>/generated/live_vad_segments``
    device
        sounddevice device index or name.  None = system default.
    chunk_duration_ms
        How many ms of audio per sounddevice callback.  20–40 ms is typical.
        Smaller = lower latency; larger = fewer Python thread-switch overheads.
    threshold / min_silence_sec / ...
        Passed directly to VadSpeechSegmentsTracker.
    vad_model_dir
        Path to the FireRedVAD pretrained model directory.
    verbose
        Enable per-frame rich logging inside the tracker.

    Usage
    -----
    def handle(result, seg_dir):
        print(f"Saved to {seg_dir}  ({result.duration_s:.1f}s)")

    recorder = LiveVadRecorder(on_segment=handle)
    recorder.start()
    input("Press Enter to stop...")
    recorder.stop()
    """

    def __init__(
        self,
        on_segment: Optional[Callable[[SegmentResult, Path], None]] = None,
        output_dir: Path = OUTPUT_DIR,
        device: Optional[int | str] = None,
        chunk_duration_ms: int = FRAME_SHIFT_MS,
        # VAD model
        vad_model_dir: Optional[str] = None,
        # Tracker params (forwarded)
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_sec: float = DEFAULT_MIN_SILENCE_SEC,
        min_speech_sec: float = DEFAULT_MIN_SPEECH_SEC,
        soft_limit_sec: float = DEFAULT_SOFT_LIMIT_SEC,
        hard_limit_sec: float = DEFAULT_MAX_SPEECH_SEC,
        preroll_max_sec: float = DEFAULT_PREROLL_MAX_SEC,
        preroll_hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
        preroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
        preroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
        postroll_max_sec: float = DEFAULT_POSTROLL_MAX_SEC,
        postroll_hybrid_threshold: float = DEFAULT_POSTROLL_HYBRID_THRESHOLD,
        postroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
        postroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
        soft_limit_min_valley_duration_s: float = DEFAULT_SOFT_LIMIT_MIN_VALLEY_DURATION_S,
        soft_limit_smoothing_window: int = DEFAULT_SOFT_LIMIT_SMOOTHING_WINDOW,
        soft_limit_trough_prominence: float = DEFAULT_SOFT_LIMIT_TROUGH_PROMINENCE,
        soft_limit_min_trough_offset_s: float = DEFAULT_SOFT_LIMIT_MIN_TROUGH_OFFSET_S,
        verbose: bool = False,
    ) -> None:
        self.on_segment = on_segment
        self.output_dir = Path(output_dir)
        self.device = device
        self.chunk_samples = int(SAMPLE_RATE * chunk_duration_ms / 1000)
        self.threshold = threshold
        self.verbose = verbose

        # Lazy-import default model dir from the same place FireRedVAD uses
        if vad_model_dir is None:
            from jet.audio.audio_waveform.vad.vad_firered_hybrid import SAVE_DIR

            vad_model_dir = SAVE_DIR

        # ── Build VAD model ────────────────────────────────────────────────
        self._vad = FireRedVAD(
            model_dir=vad_model_dir,
            threshold=threshold,
            min_silence_duration_sec=min_silence_sec,
            min_speech_duration_sec=min_speech_sec,
        )

        # ── Build tracker ──────────────────────────────────────────────────
        self._tracker = VadSpeechSegmentsTracker(
            threshold=threshold,
            min_silence_sec=min_silence_sec,
            min_speech_sec=min_speech_sec,
            soft_limit_sec=soft_limit_sec,
            hard_limit_sec=hard_limit_sec,
            preroll_max_sec=preroll_max_sec,
            preroll_hybrid_threshold=preroll_hybrid_threshold,
            preroll_prob_weight=preroll_prob_weight,
            preroll_rms_weight=preroll_rms_weight,
            postroll_max_sec=postroll_max_sec,
            postroll_hybrid_threshold=postroll_hybrid_threshold,
            postroll_prob_weight=postroll_prob_weight,
            postroll_rms_weight=postroll_rms_weight,
            soft_limit_min_valley_duration_s=soft_limit_min_valley_duration_s,
            soft_limit_smoothing_window=soft_limit_smoothing_window,
            soft_limit_trough_prominence=soft_limit_trough_prominence,
            soft_limit_min_trough_offset_s=soft_limit_min_trough_offset_s,
            verbose=verbose,
        )

        # ── Internal state ─────────────────────────────────────────────────
        self._audio_queue: queue.Queue[np.ndarray | object] = queue.Queue(maxsize=200)
        self._worker_thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._seg_counter = 0

        # Raw prob accumulation: we keep raw probs per segment in parallel
        # with the tracker so we can save them separately.
        self._raw_probs_buf: List[float] = []

        # Progress stats
        self._frames_processed = 0
        self._start_time: float = 0.0

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the microphone stream and start the worker thread."""
        if self._running:
            console.print("[yellow]LiveVadRecorder already running[/yellow]")
            return

        self._running = True
        self._start_time = time.monotonic()
        self._tracker.reset()
        self._vad.reset()
        self._raw_probs_buf.clear()

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="vad-worker",
            daemon=True,
        )
        self._worker_thread.start()

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_samples,
            device=self.device,
            callback=self._sd_callback,
        )
        self._stream.start()

        console.print(
            f"[bold green]LiveVadRecorder started[/bold green]  "
            f"[dim]device={self.device or 'default'}  "
            f"chunk={self.chunk_samples} samples "
            f"({self.chunk_samples / SAMPLE_RATE * 1000:.0f} ms)[/dim]"
        )

    def stop(self) -> None:
        """
        Stop recording.  Flushes any buffered audio as a final segment
        before shutting down the worker thread.
        """
        if not self._running:
            return

        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Signal worker to flush and exit
        self._audio_queue.put(_STOP)

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=10)
            self._worker_thread = None

        elapsed = time.monotonic() - self._start_time
        console.print(
            f"[bold red]LiveVadRecorder stopped[/bold red]  "
            f"[dim]wall={elapsed:.1f}s  segments={self._seg_counter}[/dim]"
        )

    def __enter__(self) -> "LiveVadRecorder":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── sounddevice callback (audio thread) ────────────────────────────────────

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Called by sounddevice on the C-level audio thread.

        RULES:
          - No blocking calls.
          - No Python I/O.
          - Only array copy + queue.put_nowait.
        """
        if status:
            # Log overflow/underflow on the next console flush — not here
            pass
        chunk = indata[:, 0].copy()  # (N,) float32, mono
        try:
            self._audio_queue.put_nowait(chunk)
        except queue.Full:
            pass  # drop frame rather than block the audio thread

    # ── Worker thread ──────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Consumes audio chunks from the queue.

        For each chunk:
          1. Run FireRedVAD → smoothed_prob, raw_prob
          2. Push into VadSpeechSegmentsTracker
          3. If result → save files → call on_segment
        """
        console.print("[dim cyan]  worker thread started[/dim cyan]")

        while True:
            try:
                item = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                if not self._running:
                    break
                continue

            if item is _STOP:
                self._flush_final()
                break

            chunk: np.ndarray = item
            self._process_chunk(chunk)

        console.print("[dim cyan]  worker thread exiting[/dim cyan]")

    def _process_chunk(self, chunk: np.ndarray) -> None:
        """Run VAD on one audio chunk and push result into tracker."""
        self._frames_processed += 1

        # ── VAD inference ──────────────────────────────────────────────────
        smoothed_prob = self._vad.get_speech_prob(chunk)

        duration_s = self._tracker.get_current_duration_s()
        log_accumulating(duration_s, smoothed_prob)

        # Raw prob: last accumulated frame from the VAD model
        acc = self._vad.get_accumulated_probs()
        raw_prob = acc[-1]["raw_prob"] if acc else smoothed_prob

        self._raw_probs_buf.append(raw_prob)

        # ── Tracker push ───────────────────────────────────────────────────
        result: Optional[SegmentResult] = self._tracker.push(smoothed_prob, chunk)

        if result is not None:
            self._handle_result(result)

    def _flush_final(self) -> None:
        """Called at stop() — drain any partial segment."""
        result = self._tracker.flush()
        if result is not None:
            self._handle_result(result)

    def _handle_result(self, result: SegmentResult) -> None:
        """Save files for every sub-segment in the result and fire callback."""
        # Raw probs accumulated so far belong to this segment
        raw_probs_for_seg = list(self._raw_probs_buf)
        self._raw_probs_buf.clear()

        for seg in result.segments:
            self._seg_counter += 1
            num = self._seg_counter

            seg_dir = _save_segment(
                seg_num=num,
                result=result,
                segment=seg,
                raw_probs=raw_probs_for_seg,
                output_dir=self.output_dir,
            )

            console.print(
                f"[bold green]  saved segment {num:03d}[/bold green]  "
                f"[dim]{result.end_reason}[{result.end_condition_label}]  "
                f"{result.duration_s:.2f}s  →  {seg_dir}[/dim]"
            )

            if self.on_segment is not None:
                try:
                    self.on_segment(result, seg_dir)
                except Exception as exc:
                    console.print(f"[red]  on_segment callback error: {exc}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────────────────────


def _demo_callback(result: SegmentResult, seg_dir: Path) -> None:
    """Simple demo callback — prints a summary table."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[cyan]reason[/cyan]", result.end_reason)
    table.add_row("[cyan]condition[/cyan]", result.end_condition_label)
    table.add_row("[cyan]duration[/cyan]", f"{result.duration_s:.2f} s")
    table.add_row("[cyan]sub-segments[/cyan]", str(len(result.segments)))
    table.add_row("[cyan]output[/cyan]", str(seg_dir))
    console.print(Panel(table, title="Segment complete", border_style="green"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live VAD segment recorder")
    parser.add_argument(
        "--device", default=None, help="sounddevice device index or name"
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-silence", type=float, default=DEFAULT_MIN_SILENCE_SEC)
    parser.add_argument("--soft-limit", type=float, default=DEFAULT_SOFT_LIMIT_SEC)
    parser.add_argument("--hard-limit", type=float, default=DEFAULT_MAX_SPEECH_SEC)
    parser.add_argument("--chunk-ms", type=int, default=FRAME_SHIFT_MS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Reset output dir
    shutil.rmtree(args.output_dir, ignore_errors=True)

    recorder = LiveVadRecorder(
        on_segment=_demo_callback,
        output_dir=args.output_dir,
        device=args.device,
        chunk_duration_ms=args.chunk_ms,
        threshold=args.threshold,
        min_silence_sec=args.min_silence,
        soft_limit_sec=args.soft_limit,
        hard_limit_sec=args.hard_limit,
        verbose=args.verbose,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )

    with progress:
        task: TaskID = progress.add_task(
            "[green]Recording…  (Ctrl-C to stop)", total=None
        )
        recorder.start()
        try:
            while True:
                time.sleep(0.1)
                progress.advance(task)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping…[/yellow]")
        finally:
            recorder.stop()

    console.print(
        f"\n[bold]Output:[/bold] [link=file://{args.output_dir.resolve()}]{args.output_dir}[/link]"
    )


if __name__ == "__main__":
    main()
