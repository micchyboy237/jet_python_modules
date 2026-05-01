from __future__ import annotations

import argparse
import queue
import shutil
import time
from pathlib import Path
from typing import Callable, List, Optional

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

    Wave deduplication
    ------------------
    Every completed wave is identified by its ``frame_start`` value — the
    frame index at which the VAD signal first crossed the open threshold.
    Because the VAD signal must dip below the close threshold between two
    utterances, no two distinct utterances can share the same frame_start.
    Once a wave is saved its frame_start is recorded in
    ``_saved_wave_starts``; subsequent VAD passes that re-detect the same
    wave (from the growing rolling buffer) are silently ignored.

    Open-wave handling
    ------------------
    A wave whose ``has_fallen=False`` is still being spoken and must NOT be
    saved until the VAD signal drops (which will happen in a later pass).
    The only exception is the final ``flush()`` call at stream end: there the
    last open wave is saved using all buffered audio as its endpoint, provided
    it passes shape and duration validation.

    Parameters
    ----------
    output_dir : Path
        Root folder where wave sub-directories are written.
    sample_rate : int
        Audio sample rate expected from the microphone (default 16 000 Hz).
    threshold : float
        VAD open threshold — hybrid signal must reach this to start a wave.
    close_threshold : float | None
        Hysteresis close threshold (defaults to ``threshold``).
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
    disable_merge : bool
        Disable merging of consecutive raw waves separated by a short gap
        (default False — up to 150 ms / 15 frames).
    on_wave : Callable[[SpeechWave, Path], None] | None
        Optional callback invoked once per saved wave.
        Receives two arguments: (wave: SpeechWave, wave_dir: Path).
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
        disable_merge: bool = False,
        on_wave: Optional[Callable[[SpeechWave, Path], None]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.close_threshold = (
            close_threshold if close_threshold is not None else threshold
        )

        if shape_cfg is not None:
            self.shape_cfg = shape_cfg
        else:
            self.shape_cfg = WaveShapeConfig(
                max_merge_gap_frames=15 if not disable_merge else 0
            )

        self.vad_interval_sec = vad_interval_sec
        self.max_buffer_samples = int(max_buffer_sec * sample_rate)
        self.prob_weight = prob_weight
        self.rms_weight = rms_weight

        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._samples: np.ndarray = np.empty(0, dtype=np.float32)
        self._wave_counter: int = 0
        # frame_start values of every wave that has already been persisted.
        # Used to skip re-detection of the same wave on subsequent VAD passes.
        self._saved_wave_starts: set[int] = set()
        self._all_waves: List[SpeechWave] = []
        self.on_wave = on_wave

        self._vad: Optional[FireRedVAD] = None
        self.total_samples: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, chunk: np.ndarray) -> None:
        """
        Ingest one audio chunk from the sounddevice callback.

        The chunk must be:
          - mono float32 (shape ``(N,)`` or ``(N, 1)``)
          - sampled at ``self.sample_rate``

        This method is intentionally lightweight — heavy work (VAD, saving)
        is triggered only when enough audio has accumulated.
        """
        chunk = np.asarray(chunk, dtype=np.float32).squeeze()
        if chunk.ndim != 1:
            raise ValueError(f"feed() expects mono audio, got shape {chunk.shape}")

        self._buffer = np.concatenate([self._buffer, chunk])
        self.total_samples = len(self._buffer)
        self._run_vad()

    def flush(self) -> None:
        """
        Force a VAD pass on whatever audio remains in the buffer.

        Call this when the recording session ends to catch any wave that was
        still open (has_fallen=False) at stream close time.  The open wave's
        ``frame_end`` will be set to the last available frame, which is the
        correct endpoint given the audio we have.
        """
        self._run_vad(force=True)

    def reset(self) -> None:
        """
        Clear all internal state so the tracker can start a fresh recording.

        The wave counter and saved-wave directory are NOT reset — call with
        a new ``output_dir`` if you want completely separate output.
        """
        self._buffer = np.empty(0, dtype=np.float32)
        self.total_samples = 0
        self._saved_wave_starts.clear()

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

        save_file(self._all_waves, self.output_dir / "speech_waves.json")

        if speech_probs is not None:
            save_file(speech_probs, self.output_dir / "speech_probs.json")

        rows = build_summary_rows(self._all_waves, self.output_dir, segments)
        save_file(rows, self.output_dir / "summary.json")

        top5 = _top5_reports(self._all_waves, self.output_dir, segments)
        save_file(top5, self.output_dir / "top_5_waves.json")

        console.log(f"[green]✓ Summary files saved to {self.output_dir}[/green]")

    def get_total_duration(self) -> float:
        """Returns current buffered duration (will always be ≤ max_buffer_sec)."""
        return self.total_samples / self.sample_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        newly-completed ones.  Waves that are still open (has_fallen=False)
        are skipped unless *force* is True (end-of-stream flush).

        Parameters
        ----------
        force : bool
            When True, also consider waves that haven't fallen yet (end of
            stream).  The open wave's frame_end will be the last available
            frame — the correct endpoint given the buffered audio.
        """
        audio_np = self._buffer.copy()
        if len(audio_np) < HOP_SIZE * 2:
            return

        vad = self._get_vad()
        try:
            frame_results, _ = vad.detect_full(audio_np)
        except Exception as exc:
            console.log(f"[red]VAD error: {exc}[/red]")
            return

        speech_probs: List[float] = [r.smoothed_prob for r in frame_results]

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

        for wave in all_waves:
            frame_start: int = wave["details"]["frame_start"]

            is_fallen = wave.get("has_fallen", False)
            is_valid = wave.get("is_valid", False)
            shape_ok = wave["details"].get("shape_passed", False)
            dur_ok = (
                wave["details"].get("duration_sec", 0.0)
                >= self.shape_cfg.min_duration_sec
            )

            # Decide whether to persist this wave:
            #
            #   Normal path  — wave is fully closed and passed all validators.
            #   Flush path   — end of stream; accept an open wave if it has a
            #                  real mountain shape and sufficient duration.
            #                  frame_end is already set to len(speech_probs),
            #                  i.e. the last available frame, by check_speech_waves.
            if is_valid and is_fallen:
                pass  # normal completed wave — save it
            elif force and shape_ok and dur_ok:
                pass  # last open wave at stream end — save it
            else:
                continue  # still growing mid-stream, or genuinely invalid

            # Deduplicate: skip any wave whose frame_start was already saved.
            # frame_start is a reliable unique key because the VAD signal must
            # dip below the close threshold between two distinct utterances,
            # so no two separate utterances can share the same opening frame.
            if frame_start in self._saved_wave_starts:
                continue

            self._saved_wave_starts.add(frame_start)
            wave_dir = self._save_wave(wave, audio_np, speech_probs)
            if self.on_wave is not None:
                self.on_wave(wave, wave_dir)

    def _save_wave(
        self,
        wave: SpeechWave,
        audio_np: np.ndarray,
        speech_probs: List[float],
    ) -> Path:
        """
        Persist audio + metadata for one valid speech wave.

        Delegates to ``save_wave_data()`` from ``speech_waves.py``.
        Each wave gets its own numbered sub-directory under ``output_dir``.
        """
        self._wave_counter += 1
        wave_idx = self._wave_counter

        wave_dir = save_wave_data(
            wave=wave,
            audio_np=audio_np,
            speech_probs=speech_probs,
            sampling_rate=self.sample_rate,
            output_dir=self.output_dir,
            seg_num=1,
            wave_num=wave_idx,
            hop_size=HOP_SIZE,
            prob_weight=self.prob_weight,
            rms_weight=self.rms_weight,
        )

        wave_json_path = wave_dir / "wave.json"
        wave_link = f"file://{wave_json_path.resolve()}"
        sound_path = wave_dir / "sound.wav"
        play_link = f"file://{sound_path.resolve()}"

        console.log(
            f"[green]✓ [link={wave_link}]Wave {wave_idx:03d}[/link][/green] "
            f"[white]{wave['start_sec']:.2f}s → {wave['end_sec']:.2f}s[/white]  "
            f"dur=[yellow]{wave['details']['duration_sec']:.2f}s[/yellow]  "
            f"peak=[magenta]{wave['details']['max_prob']:.3f}[/magenta]  "
            f"[bright_cyan][link={play_link}]▶[/link][/bright_cyan]"
        )

        self._all_waves.append(wave)
        return wave_dir


# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------


def get_args(default_output_dir: str | Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream microphone audio and save valid speech waves to disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        default=default_output_dir,
        type=Path,
        help="Directory where wave files are saved.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.1,
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
    parser.add_argument(
        "--disable-merge",
        action="store_true",
        default=False,
        help=(
            "Disable merging of consecutive raw waves separated by a short gap "
            "(default False - up to 150 ms / 15 frames)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    args = get_args(OUTPUT_DIR)

    shutil.rmtree(args.output_dir, ignore_errors=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tracker = SpeechWavesTracker(
        output_dir=args.output_dir,
        sample_rate=SAMPLE_RATE,
        threshold=args.threshold,
        close_threshold=args.close_threshold,
        vad_interval_sec=args.vad_interval,
        max_buffer_sec=args.max_buffer,
        prob_weight=args.prob_weight,
        rms_weight=args.rms_weight,
        disable_merge=args.disable_merge,
        on_wave=lambda wave, wave_dir: print(
            f"[WAVE] Detected new wave: "
            f"Start={wave['start_sec']:.2f}s, "
            f"End={wave['end_sec']:.2f}s, "
            f"Duration={wave['details']['duration_sec']:.2f}s, "
            f"PeakProb={wave['details']['max_prob']:.3f}, "
            f"Dir={wave_dir.name}"
        ),
    )

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
        audio_queue.put(indata[:, 0].copy())

    console.print(
        "[bold cyan]🎙  Recording"
        + (f" for {args.duration}s" if args.duration else " (Ctrl-C to stop)")
        + f" → {args.output_dir}[/bold cyan]"
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

    while not audio_queue.empty():
        tracker.feed(audio_queue.get_nowait())

    tracker.flush()
    tracker.save_summary_files()

    console.print(
        f"\n[bold green]Done.[/bold green] "
        f"{tracker._wave_counter} wave(s) saved to [cyan]{args.output_dir}[/cyan]"
    )
