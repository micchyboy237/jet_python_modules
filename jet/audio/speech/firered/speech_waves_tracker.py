# jet.audio.speech.firered.speech_waves_tracker

from __future__ import annotations

import argparse
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd
from jet.audio.audio_types import SpeechWave
from jet.audio.helpers.config import HOP_SIZE, SAMPLE_RATE
from jet.audio.speech.firered.config import SAVE_DIR
from jet.audio.speech.firered.speech_waves import (
    WaveShapeConfig,
    _top5_reports,
    build_summary_rows,
    check_speech_waves,
    save_wave_data,
)
from jet.audio.speech.firered.vad import FireRedVAD
from jet.file.utils import save_file
from rich.console import Console

console = Console()


class SpeechWavesTracker:
    """
    Accumulates raw audio chunks from a sounddevice stream, periodically
    runs FireRedVAD, detects completed speech waves (rise → peak → fall),
    and saves each valid wave's audio + metadata to disk.

    Parameters
    ----------
    output_dir : Path
        Root folder where wave sub-directories are written.
    sample_rate : int
        Audio sample rate expected from the microphone (default 16 000 Hz).
    threshold : float
        VAD open threshold — hybrid signal must reach this to start a wave.
    close_threshold : float | None
        Hysteresis close threshold (defaults to `threshold`).
    shape_cfg : WaveShapeConfig | None
        Shape-validation settings (prominence, excursion, min frames …).
    vad_interval_sec : float
        How often (seconds of buffered audio) to trigger a VAD pass.
        Lower = more responsive but more CPU; higher = more context for VAD.
    max_buffer_sec : float
        Hard cap on rolling buffer length. Older audio is discarded once the
        buffer exceeds this length.  Prevents unbounded RAM growth.
    prob_weight : float
        Weight of VAD probability in the hybrid signal (default 0.5).
    rms_weight : float
        Weight of RMS energy in the hybrid signal (default 0.5).
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = 0.5,
        close_threshold: Optional[float] = None,
        shape_cfg: Optional[WaveShapeConfig] = None,
        vad_interval_sec: float = 1.0,
        max_buffer_sec: float = 30.0,
        prob_weight: float = 0.5,
        rms_weight: float = 0.5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.close_threshold = (
            close_threshold if close_threshold is not None else threshold
        )
        self.shape_cfg = shape_cfg or WaveShapeConfig()
        self.vad_interval_sec = vad_interval_sec
        self.max_buffer_samples = int(max_buffer_sec * sample_rate)
        self.prob_weight = prob_weight
        self.rms_weight = rms_weight

        # ── Internal state ──────────────────────────────────────────────────
        # Raw audio accumulator (float32, mono)
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)
        # How many samples were in the buffer the last time we ran VAD
        self._last_vad_sample: int = 0
        # Global wave counter (never resets within a session)
        self._wave_counter: int = 0
        # Frame index of the last wave boundary we already saved
        self._saved_up_to_frame: int = 0
        # Accumulate all detected/saved waves for summary export
        self._all_waves: List[SpeechWave] = []
        # Lock so feed() and flush() are safe to call from different threads
        self._lock = threading.Lock()

        # VAD model (lazy-loaded on first VAD call)
        self._vad: Optional[FireRedVAD] = None

    # ── Public interface ─────────────────────────────────────────────────────

    def feed(self, chunk: np.ndarray) -> None:
        """
        Ingest one audio chunk from the sounddevice callback.

        The chunk must be:
          - mono float32 (shape ``(N,)`` or ``(N, 1)``)
          - sampled at ``self.sample_rate``

        This method is intentionally lightweight — heavy work (VAD, saving)
        is triggered only when enough audio has accumulated.
        """
        # Flatten stereo / column-vector to 1-D
        chunk = np.asarray(chunk, dtype=np.float32).squeeze()
        if chunk.ndim != 1:
            raise ValueError(f"feed() expects mono audio, got shape {chunk.shape}")

        with self._lock:
            # Append to rolling buffer
            self._buffer = np.concatenate([self._buffer, chunk])

            # Trim the front if we've exceeded the hard cap
            if len(self._buffer) > self.max_buffer_samples:
                excess = len(self._buffer) - self.max_buffer_samples
                self._buffer = self._buffer[excess:]
                # Adjust the last-VAD cursor so we don't re-run on trimmed audio
                self._last_vad_sample = max(0, self._last_vad_sample - excess)

            # Run VAD if enough new audio has arrived since the last pass
            new_samples = len(self._buffer) - self._last_vad_sample
            if new_samples >= int(self.vad_interval_sec * self.sample_rate):
                self._run_vad()

    def flush(self) -> None:
        """
        Force a VAD pass on whatever audio remains in the buffer.

        Call this when the recording session ends to catch any wave that was
        still open (has_fallen=False) at stream close time.
        """
        with self._lock:
            self._run_vad(force=True)

    def reset(self) -> None:
        """
        Clear all internal state so the tracker can start a fresh recording.
        The wave counter and saved-wave directory are NOT reset — call with
        a new ``output_dir`` if you want completely separate output.
        """
        with self._lock:
            self._buffer = np.empty(0, dtype=np.float32)
            self._last_vad_sample = 0
            self._saved_up_to_frame = 0

    def save_summary_files(
        self,
        speech_probs: Optional[List[float]] = None,
        segments: Optional[List] = None,
    ) -> None:
        """
        Save summary JSON files similar to speech_waves.py.

        In streaming mode we don't have pre-computed segments, so we pass
        an empty list. All waves will use the default seg_num=1, which
        matches the current behavior in _save_wave().
        """
        if segments is None:
            segments = []

        # Save all detected waves
        save_file(self._all_waves, self.output_dir / "speech_waves.json")

        # Save speech probabilities if provided (optional in streaming mode)
        if speech_probs is not None:
            save_file(speech_probs, self.output_dir / "speech_probs.json")

        # Build and save summary rows for display/table
        rows = build_summary_rows(self._all_waves, self.output_dir, segments)
        save_file(rows, self.output_dir / "summary.json")

        # Build and save top 5 waves report
        top5 = _top5_reports(self._all_waves, self.output_dir, segments)
        save_file(top5, self.output_dir / "top_5_waves.json")

        console.log(f"[green]✓ Summary files saved to {self.output_dir}[/green]")

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_vad(self) -> FireRedVAD:
        """Lazy-load the VAD model (expensive; only done once)."""
        if self._vad is None:
            console.log("[cyan]Loading FireRedVAD model…[/cyan]")
            self._vad = FireRedVAD(
                model_dir=SAVE_DIR,
                threshold=self.threshold,
            )
        return self._vad

    def _run_vad(self, force: bool = False) -> None:
        """
        Run FireRedVAD on the current buffer, detect waves, and save any
        newly-completed ones.

        Must be called with ``self._lock`` already held.

        Parameters
        ----------
        force : bool
            When True, process even waves that haven't fallen yet (end of
            stream).  Normally only ``has_fallen=True`` waves are saved.
        """
        audio_np = self._buffer.copy()
        if len(audio_np) < HOP_SIZE * 2:
            return  # Not enough audio for even one frame

        # ── Run VAD inference ────────────────────────────────────────────────
        vad = self._get_vad()
        try:
            frame_results, _ = vad.detect_full(audio_np)
        except Exception as exc:
            console.log(f"[red]VAD error: {exc}[/red]")
            return

        speech_probs: List[float] = [r.smoothed_prob for r in frame_results]

        # ── Detect waves ─────────────────────────────────────────────────────
        all_waves = check_speech_waves(
            speech_probs=speech_probs,
            threshold=self.threshold,
            close_threshold=self.close_threshold,
            sampling_rate=self.sample_rate,
            shape_cfg=self.shape_cfg,
            audio_np=audio_np,
            prob_weight=self.prob_weight,
            rms_weight=self.rms_weight,
        )

        # ── Emit completed waves we haven't saved yet ────────────────────────
        for wave in all_waves:
            if not wave.get("is_valid", False):
                continue

            frame_start = wave["details"]["frame_start"]
            frame_end = wave["details"]["frame_end"]

            # Skip waves we've already saved
            if frame_end <= self._saved_up_to_frame:
                continue

            # Only save waves that have fully closed (has_fallen=True),
            # unless flush() forced us to emit open waves too.
            if not wave.get("has_fallen", False) and not force:
                continue

            self._save_wave(wave, audio_np, speech_probs)
            self._saved_up_to_frame = max(self._saved_up_to_frame, frame_end)

        self._last_vad_sample = len(self._buffer)

    def _save_wave(
        self,
        wave: SpeechWave,
        audio_np: np.ndarray,
        speech_probs: List[float],
    ) -> None:
        """
        Persist audio + metadata for one valid speech wave.

        Delegates to ``save_wave_data()`` from ``speech_waves.py``.
        Each wave gets its own numbered sub-directory under ``output_dir``.
        """
        self._wave_counter += 1
        wave_idx = self._wave_counter

        console.log(
            f"[green]✓ Wave {wave_idx:03d}[/green] "
            f"[white]{wave['start_sec']:.2f}s → {wave['end_sec']:.2f}s[/white]  "
            f"dur=[yellow]{wave['details']['duration_sec']:.2f}s[/yellow]  "
            f"peak=[magenta]{wave['details']['max_prob']:.3f}[/magenta]"
        )

        save_wave_data(
            wave=wave,
            audio_np=audio_np,
            speech_probs=speech_probs,
            sampling_rate=self.sample_rate,
            output_dir=self.output_dir,
            seg_num=1,  # streaming has no pre-segmented structure
            wave_num=wave_idx,
            hop_size=HOP_SIZE,
            prob_weight=self.prob_weight,
            rms_weight=self.rms_weight,
        )
        # Accumulate wave for final summary export
        self._all_waves.append(wave)


# ── CLI ──────────────────────────────────────────────────────────────────────


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream microphone audio and save valid speech waves to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        default="generated/speech_waves_tracker",
        help="Directory where wave files are saved.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="VAD probability threshold to open a speech wave.",
    )
    parser.add_argument(
        "-c",
        "--close-threshold",
        type=float,
        default=None,
        help="Hysteresis close threshold (default: same as --threshold).",
    )
    parser.add_argument(
        "--vad-interval",
        type=float,
        default=1.0,
        help="Seconds of new audio to buffer before running VAD.",
    )
    parser.add_argument(
        "--max-buffer",
        type=float,
        default=30.0,
        help="Maximum rolling buffer length in seconds.",
    )
    parser.add_argument(
        "--prob-weight",
        type=float,
        default=0.5,
        help="Weight of VAD probability in the hybrid signal.",
    )
    parser.add_argument(
        "--rms-weight",
        type=float,
        default=0.5,
        help="Weight of RMS energy in the hybrid signal.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input device index (default: system default).",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=1600,
        help="Samples per sounddevice callback block (100 ms at 16 kHz).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording duration in seconds (0 = run until Ctrl-C).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the output directory before starting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Define OUTPUT_DIR similar to speech_waves.py for consistent output structure
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    args = get_args()

    # Use OUTPUT_DIR as default if user didn't specify a custom path
    if args.output == "generated/speech_waves_tracker":
        args.output = str(OUTPUT_DIR)

    output_dir = Path(args.output)

    shutil.rmtree(output_dir, ignore_errors=True)  # Clear before starting
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = SpeechWavesTracker(
        output_dir=output_dir,
        sample_rate=SAMPLE_RATE,
        threshold=args.threshold,
        close_threshold=args.close_threshold,
        vad_interval_sec=args.vad_interval,
        max_buffer_sec=args.max_buffer,
        prob_weight=args.prob_weight,
        rms_weight=args.rms_weight,
    )

    # sounddevice puts chunks into this queue; the main thread drains it.
    # Using a queue decouples the real-time callback from slower VAD work.
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def sd_callback(
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """Called by sounddevice in a background thread for every block."""
        if status:
            console.log(f"[yellow]Stream status: {status}[/yellow]")
        # Copy so the buffer isn't mutated after we return
        audio_queue.put(indata[:, 0].copy())

    console.print(
        "[bold cyan]🎙  Recording"
        + (f" for {args.duration}s" if args.duration else " (Ctrl-C to stop)")
        + f" → {output_dir}[/bold cyan]"
    )

    start_time = time.monotonic()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=args.blocksize,
        device=args.device,
        callback=sd_callback,
    ):
        try:
            while True:
                # Drain the queue in the main thread
                try:
                    chunk = audio_queue.get(timeout=0.05)
                    tracker.feed(chunk)
                except queue.Empty:
                    pass

                elapsed = time.monotonic() - start_time
                if args.duration > 0 and elapsed >= args.duration:
                    break

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted — flushing final buffer…[/yellow]")

    # Process any remaining audio in the queue
    while not audio_queue.empty():
        tracker.feed(audio_queue.get_nowait())

    # Finalize: flush buffer and save summary files
    tracker.flush()
    tracker.save_summary_files()

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"{tracker._wave_counter} wave(s) saved to [cyan]{output_dir}[/cyan]"
    )
