"""
live_vad_capture.py
───────────────────
Real-time speech segmentation from a microphone using FireRedVAD
(FireRedStreamVad) with the hybrid pre-roll buffer from vad_firered_hybrid.

Architecture
────────────
  Microphone  ──►  sounddevice callback  ──►  audio queue
                                                    │
                                          VAD processing loop
                                                    │
                       ┌────────────────────────────┤
                       │                            │
                 PreRollBuffer              get_speech_prob()
                 (rolling window)           (FireRedVAD wrapper)
                       │                            │
                       └──► state machine (SILENCE / SPEECH)
                                        │
                              on_segment(audio_np, seg_num)
                                        │
                               save WAV + print info

Usage
─────
  python live_vad_capture.py                   # use default mic, save WAVs
  python live_vad_capture.py --device 1        # pick a specific input device
  python live_vad_capture.py --list-devices    # list available devices
  python live_vad_capture.py --no-save         # only print, don't save
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import torch
import torchaudio

# ── imports from the existing module ────────────────────────────────────────
from jet.audio.audio_waveform.vad.vad_firered_hybrid import (
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_PREROLL_HYBRID_THRESHOLD,
    DEFAULT_PREROLL_MAX_SEC,
    DEFAULT_PREROLL_PROB_WEIGHT,
    DEFAULT_PREROLL_RMS_WEIGHT,
    DEFAULT_SMOOTH_WINDOW_SIZE,
    DEFAULT_THRESHOLD,
    SAVE_DIR,
    FireRedVAD,
    PreRollBuffer,
)
from rich.console import Console

console = Console()

# ── constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000  # FireRedVAD hard requirement
CHUNK_SAMPLES = 160  # 10 ms per chunk  (160 / 16000 = 0.01 s)
CHUNK_SEC = CHUNK_SAMPLES / SAMPLE_RATE

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem


# ════════════════════════════════════════════════════════════════════════════
#  State machine labels
# ════════════════════════════════════════════════════════════════════════════
_SILENCE = "silence"
_SPEECH = "speech"


# ════════════════════════════════════════════════════════════════════════════
#  Default segment handler
# ════════════════════════════════════════════════════════════════════════════
def _default_on_segment(
    audio_np: np.ndarray,
    seg_num: int,
    output_dir: Path,
    save: bool,
) -> None:
    """
    Called whenever a complete speech segment is ready.

    Parameters
    ----------
    audio_np   : float32 array, shape (N,), 16 kHz mono
    seg_num    : 1-based segment counter
    output_dir : where to write WAV files
    save       : when False only print info, skip disk write
    """
    duration = len(audio_np) / SAMPLE_RATE
    console.print(
        f"[bold green]▶ Segment {seg_num:03d}[/bold green]  "
        f"dur=[bold magenta]{duration:.2f}s[/bold magenta]  "
        f"samples=[cyan]{len(audio_np)}[/cyan]"
    )

    if not save:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / f"segment_{seg_num:03d}.wav"
    try:
        torchaudio.save(
            str(wav_path),
            torch.from_numpy(audio_np).unsqueeze(0),  # (1, N)
            SAMPLE_RATE,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        console.print(f"  [dim]saved → {wav_path}[/dim]")
    except Exception as exc:
        console.print(f"  [red]WAV save failed: {exc}[/red]")


# ════════════════════════════════════════════════════════════════════════════
#  Core streaming class
# ════════════════════════════════════════════════════════════════════════════
class LiveVADCapture:
    """
    Captures microphone audio in real time and calls *on_segment* with each
    detected speech segment as a float32 numpy array at 16 kHz.

    How it works (plain English)
    ────────────────────────────
    1. sounddevice opens the mic and calls our callback every 10 ms.
    2. The callback drops the tiny audio chunk into a thread-safe queue.
    3. The processing loop reads from the queue continuously:
       a. Every chunk is fed to PreRollBuffer (rolling look-back window).
       b. Every chunk is also fed to FireRedVAD.get_speech_prob().
       c. We track a simple two-state machine:
          • SILENCE → if prob rises above threshold, grab the preroll
            (audio that was just before speech started) and switch to SPEECH.
          • SPEECH  → keep accumulating audio; if prob falls and stays below
            threshold for long enough, close the segment and call on_segment.
    """

    def __init__(
        self,
        on_segment: Optional[Callable[[np.ndarray, int], None]] = None,
        device: Optional[int | str] = None,
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_duration_sec: float = DEFAULT_MIN_SILENCE_SEC,
        min_speech_duration_sec: float = DEFAULT_MIN_SPEECH_SEC,
        max_speech_duration_sec: float = DEFAULT_MAX_SPEECH_SEC,
        smooth_window_size: int = DEFAULT_SMOOTH_WINDOW_SIZE,
        max_buffer_sec: float = DEFAULT_MAX_BUFFER_SEC,
        preroll_max_sec: float = DEFAULT_PREROLL_MAX_SEC,
        preroll_hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
        preroll_prob_weight: float = DEFAULT_PREROLL_PROB_WEIGHT,
        preroll_rms_weight: float = DEFAULT_PREROLL_RMS_WEIGHT,
        output_dir: Path = Path("live_segments"),
        save_wav: bool = True,
    ) -> None:
        self.on_segment = on_segment
        self.device = device
        self.threshold = threshold
        self.min_silence_frames = int(min_silence_duration_sec / CHUNK_SEC)
        self.min_speech_frames = int(min_speech_duration_sec / CHUNK_SEC)
        self.max_speech_frames = int(max_speech_duration_sec / CHUNK_SEC)
        self.output_dir = output_dir
        self.save_wav = save_wav

        # ── audio queue ──────────────────────────────────────────────────────
        # maxsize prevents unbounded memory growth if processing falls behind
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2000)

        # ── VAD ──────────────────────────────────────────────────────────────
        self._vad = FireRedVAD(
            model_dir=SAVE_DIR,
            threshold=threshold,
            min_silence_duration_sec=min_silence_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec,
            max_speech_duration_sec=max_speech_duration_sec,
            smooth_window_size=smooth_window_size,
            max_buffer_sec=max_buffer_sec,
        )

        # ── pre-roll buffer ───────────────────────────────────────────────────
        self._preroll = PreRollBuffer(
            max_preroll_sec=preroll_max_sec,
            hybrid_threshold=preroll_hybrid_threshold,
            prob_weight=preroll_prob_weight,
            rms_weight=preroll_rms_weight,
            sample_rate=SAMPLE_RATE,
        )

        # ── state machine ─────────────────────────────────────────────────────
        self._state: str = _SILENCE
        self._speech_buf: list[np.ndarray] = []  # chunks collected during SPEECH
        self._silence_frame_count: int = 0  # consecutive silent chunks
        self._speech_frame_count: int = 0  # frames accumulated in SPEECH
        self._seg_num: int = 0

        # ── control flags ─────────────────────────────────────────────────────
        self._running = False
        self._process_thread: Optional[threading.Thread] = None

    # ── sounddevice callback (runs in a C-level audio thread) ────────────────
    def _sd_callback(
        self,
        indata: np.ndarray,  # shape (frames, channels)
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            # e.g. input overflow — log but don't crash
            console.print(f"[yellow]sd status: {status}[/yellow]")

        # Convert to mono float32 and drop into the queue.
        # We copy because sounddevice reuses the buffer.
        mono = indata[:, 0].astype(np.float32).copy()

        try:
            self._audio_queue.put_nowait(mono)
        except queue.Full:
            # If the processing loop is too slow, drop the oldest chunk.
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(mono)
            except queue.Empty:
                pass

    # ── processing loop (runs in a dedicated thread) ─────────────────────────
    def _process_loop(self) -> None:
        """
        Reads audio chunks from the queue and runs the VAD state machine.

        State machine
        ─────────────
                ┌──────────────────────────────────────────────────────┐
                │                    SILENCE                           │
                │  • feed chunk to PreRollBuffer                       │
                │  • get speech prob from FireRedVAD                   │
                │  • if prob >= threshold → switch to SPEECH           │
                │      - grab preroll audio from PreRollBuffer         │
                │      - start accumulating speech_buf                 │
                └──────────┬───────────────────────────────────────────┘
                           │ prob >= threshold
                           ▼
                ┌──────────────────────────────────────────────────────┐
                │                    SPEECH                            │
                │  • accumulate chunk into speech_buf                  │
                │  • get speech prob                                   │
                │  • if prob <  threshold → increment silence counter  │
                │    else                 → reset   silence counter    │
                │  • if silence_frames >= min_silence_frames           │
                │      AND speech_frames >= min_speech_frames          │
                │    → EMIT segment, switch to SILENCE                 │
                │  • if speech_frames >= max_speech_frames             │
                │    → force-EMIT segment (max duration guard)         │
                └──────────────────────────────────────────────────────┘
        """
        console.print("[bold cyan]Processing loop started.[/bold cyan]")

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            prob = self._vad.get_speech_prob(chunk)

            # Always update the pre-roll buffer (we need it in SILENCE state
            # so we have context right before speech starts).
            self._preroll.push(chunk, prob)

            if self._state == _SILENCE:
                if prob >= self.threshold:
                    # ── SILENCE → SPEECH transition ──────────────────────────
                    self._state = _SPEECH
                    self._silence_frame_count = 0
                    self._speech_frame_count = 0
                    self._speech_buf.clear()

                    # Prepend the pre-roll (audio just before speech onset)
                    preroll_audio = self._preroll.get_preroll()
                    if len(preroll_audio):
                        self._speech_buf.append(preroll_audio)

                    self._speech_buf.append(chunk)
                    self._speech_frame_count += 1

            else:  # _SPEECH
                self._speech_buf.append(chunk)
                self._speech_frame_count += 1

                if prob < self.threshold:
                    self._silence_frame_count += 1
                else:
                    self._silence_frame_count = 0

                # Check: did we collect enough silence to close the segment?
                silence_long_enough = (
                    self._silence_frame_count >= self.min_silence_frames
                )
                speech_long_enough = self._speech_frame_count >= self.min_speech_frames
                speech_too_long = self._speech_frame_count >= self.max_speech_frames

                if (silence_long_enough and speech_long_enough) or speech_too_long:
                    self._emit_segment()
                    self._state = _SILENCE
                    self._preroll.reset()

        console.print("[bold cyan]Processing loop stopped.[/bold cyan]")

    def _emit_segment(self) -> None:
        """Concatenate buffered chunks and call the user's on_segment handler."""
        if not self._speech_buf:
            return

        audio_np = np.concatenate(self._speech_buf).astype(np.float32)
        self._speech_buf.clear()
        self._silence_frame_count = 0
        self._speech_frame_count = 0

        # Discard segments shorter than a reasonable minimum (e.g., mic blip)
        if len(audio_np) < int(0.1 * SAMPLE_RATE):
            return

        self._seg_num += 1

        if self.on_segment is not None:
            try:
                self.on_segment(audio_np, self._seg_num)
            except Exception as exc:
                console.print(f"[red]on_segment error: {exc}[/red]")
        else:
            _default_on_segment(
                audio_np,
                self._seg_num,
                self.output_dir,
                self.save_wav,
            )

    # ── public API ───────────────────────────────────────────────────────────
    def start(self) -> None:
        """Open the microphone stream and start processing."""
        if self._running:
            return

        self._running = True
        self._process_thread = threading.Thread(
            target=self._process_loop,
            name="vad-process",
            daemon=True,
        )
        self._process_thread.start()

        console.print(
            f"[bold green]🎙  Listening on device "
            f"[yellow]{self.device if self.device is not None else 'default'}[/yellow] "
            f"@ {SAMPLE_RATE} Hz  (Ctrl+C to stop)[/bold green]"
        )

        # Open mic stream — this blocks until stop() is called.
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            device=self.device,
            callback=self._sd_callback,
        ):
            while self._running:
                time.sleep(0.05)

    def stop(self) -> None:
        """Signal the processing loop to stop and wait for it to finish."""
        self._running = False
        if self._process_thread is not None:
            self._process_thread.join(timeout=3.0)

        # Flush any partially accumulated speech segment
        if self._state == _SPEECH and self._speech_buf:
            console.print("[yellow]Flushing partial segment on stop…[/yellow]")
            self._emit_segment()


# ════════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ════════════════════════════════════════════════════════════════════════════
def _list_devices() -> None:
    console.print(sd.query_devices())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live VAD capture using FireRedVAD + hybrid pre-roll"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="print available audio devices and exit",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input device index (default: system default)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"speech probability threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default=DEFAULT_MIN_SILENCE_SEC,
        help=f"seconds of silence to end a segment (default: {DEFAULT_MIN_SILENCE_SEC})",
    )
    parser.add_argument(
        "--min-speech",
        type=float,
        default=DEFAULT_MIN_SPEECH_SEC,
        help=f"minimum speech duration in seconds (default: {DEFAULT_MIN_SPEECH_SEC})",
    )
    parser.add_argument(
        "--max-speech",
        type=float,
        default=DEFAULT_MAX_SPEECH_SEC,
        help=f"maximum segment duration in seconds (default: {DEFAULT_MAX_SPEECH_SEC})",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW_SIZE,
        help=f"VAD smoothing window size (default: {DEFAULT_SMOOTH_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--preroll-max-sec",
        type=float,
        default=DEFAULT_PREROLL_MAX_SEC,
        help=f"max pre-roll look-back seconds (default: {DEFAULT_PREROLL_MAX_SEC})",
    )
    parser.add_argument(
        "--preroll-threshold",
        type=float,
        default=DEFAULT_PREROLL_HYBRID_THRESHOLD,
        help=f"hybrid score threshold for pre-roll (default: {DEFAULT_PREROLL_HYBRID_THRESHOLD})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory to save WAV files (default: live_segments/)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="print segment info but don't write WAV files",
    )
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        sys.exit(0)

    capturer = LiveVADCapture(
        device=args.device,
        threshold=args.threshold,
        min_silence_duration_sec=args.min_silence,
        min_speech_duration_sec=args.min_speech,
        max_speech_duration_sec=args.max_speech,
        smooth_window_size=args.smooth_window,
        preroll_max_sec=args.preroll_max_sec,
        preroll_hybrid_threshold=args.preroll_threshold,
        output_dir=args.output_dir,
        save_wav=not args.no_save,
    )

    try:
        capturer.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — stopping…[/yellow]")
    finally:
        capturer.stop()
        console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()
