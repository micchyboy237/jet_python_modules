"""
live_vad_capture.py
───────────────────
Real-time speech segmentation from a microphone using FireRedVAD with:
  • PreRollBuffer  – prepends audio that happened just *before* speech onset
  • PostRollBuffer – appends audio that happened just *after* speech drops off

Both buffers use a hybrid score:
    score = prob_weight * smoothed_prob + rms_weight * rms_norm

This mirrors the offline pre/post-roll logic in vad_firered_hybrid.py but
works incrementally frame-by-frame as audio arrives from the mic.

State machine
─────────────
  SILENCE ──(prob ≥ threshold)──► SPEECH
  SPEECH  ──(prob < threshold)──► POST-ROLL
  POST-ROLL ──(score stays low for min_silence_frames)──► emit → SILENCE
  POST-ROLL ──(score rises again)──────────────────────► back to SPEECH

Usage
─────
  python live_vad_capture.py                   # default mic, saves WAVs
  python live_vad_capture.py --device 1        # pick input device
  python live_vad_capture.py --list-devices    # list devices
  python live_vad_capture.py --no-save         # print only, no WAV
"""

from __future__ import annotations

import argparse
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import torch
import torchaudio
from jet.audio.audio_waveform.vad.vad_firered_hybrid import (
    DEFAULT_MAX_BUFFER_SEC,
    DEFAULT_MAX_SPEECH_SEC,
    DEFAULT_MIN_SILENCE_SEC,
    DEFAULT_MIN_SPEECH_SEC,
    DEFAULT_POSTROLL_HYBRID_THRESHOLD,
    DEFAULT_POSTROLL_MAX_SEC,
    DEFAULT_PREROLL_HYBRID_THRESHOLD,
    DEFAULT_PREROLL_MAX_SEC,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RMS_WEIGHT,
    DEFAULT_SMOOTH_WINDOW_SIZE,
    DEFAULT_THRESHOLD,
    SAVE_DIR,
    FireRedVAD,
)
from rich.console import Console

console = Console()

SAMPLE_RATE = 16_000  # FireRedVAD hard requirement
CHUNK_SAMPLES = 160  # 10 ms per chunk (160 / 16000 = 0.01 s)
CHUNK_SEC = CHUNK_SAMPLES / SAMPLE_RATE

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

# ── state labels ─────────────────────────────────────────────────────────────
_SILENCE = "silence"
_SPEECH = "speech"
_POSTROLL = "postroll"


# ════════════════════════════════════════════════════════════════════════════
#  PreRollBuffer  (moved here from vad_firered_hybrid.py)
# ════════════════════════════════════════════════════════════════════════════
class PreRollBuffer:
    """
    Maintains a rolling window of (audio_frame, hybrid_score) pairs.

    Purpose
    ───────
    Audio that arrives just *before* the VAD triggers often contains the
    very first consonant of a word (e.g. "H" in "Hello").  Without a
    pre-roll, those samples are lost.

    On every incoming chunk we:
      1. Compute a per-frame hybrid score = prob_weight·prob + rms_weight·rms_norm
      2. Append frames + scores to a capped ring buffer.

    At speech onset, get_preroll() scans backward and returns every
    contiguous frame whose hybrid score is above hybrid_threshold — giving us
    exactly the right amount of look-back, no more, no less.

    RMS is normalised online via a running max with mild exponential decay
    so it adapts to the room's noise floor without needing a calibration pass.
    """

    FRAME_SAMPLES = 160  # 10 ms @ 16 kHz

    def __init__(
        self,
        max_preroll_sec: float = DEFAULT_PREROLL_MAX_SEC,
        hybrid_threshold: float = DEFAULT_PREROLL_HYBRID_THRESHOLD,
        prob_weight: float = DEFAULT_PROB_WEIGHT,
        rms_weight: float = DEFAULT_RMS_WEIGHT,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.max_preroll_sec = max_preroll_sec
        self.hybrid_threshold = hybrid_threshold
        self.prob_weight = prob_weight
        self.rms_weight = rms_weight
        self.sample_rate = sample_rate
        self._max_frames = int(max_preroll_sec * 100)  # 100 frames/sec
        self._audio_frames: list[np.ndarray] = []
        self._hybrid_scores: list[float] = []
        self._rms_running_max: float = 1e-6

    def reset(self) -> None:
        self._audio_frames.clear()
        self._hybrid_scores.clear()
        self._rms_running_max = 1e-6

    def push(self, audio_chunk: np.ndarray, prob: float) -> None:
        """Ingest one mic chunk and its VAD probability."""
        if len(audio_chunk) == 0:
            return
        audio_chunk = audio_chunk.astype(np.float32)
        remainder = len(audio_chunk) % self.FRAME_SAMPLES
        if remainder:
            audio_chunk = np.pad(audio_chunk, (0, self.FRAME_SAMPLES - remainder))

        n_frames = len(audio_chunk) // self.FRAME_SAMPLES
        for i in range(n_frames):
            frame = audio_chunk[i * self.FRAME_SAMPLES : (i + 1) * self.FRAME_SAMPLES]
            rms = float(np.sqrt(np.mean(frame**2)))
            self._rms_running_max = max(self._rms_running_max * 0.9999, rms + 1e-10)
            rms_norm = min(rms / self._rms_running_max, 1.0)
            score = self.prob_weight * prob + self.rms_weight * rms_norm
            self._audio_frames.append(frame)
            self._hybrid_scores.append(score)

        if len(self._audio_frames) > self._max_frames:
            excess = len(self._audio_frames) - self._max_frames
            self._audio_frames = self._audio_frames[excess:]
            self._hybrid_scores = self._hybrid_scores[excess:]

    def get_preroll(self) -> np.ndarray:
        """
        Scan backward; return contiguous frames with score ≥ hybrid_threshold.
        Stops at the first frame below threshold (or at the buffer edge).
        """
        n = len(self._audio_frames)
        if n == 0:
            return np.array([], dtype=np.float32)
        keep = 0
        for i in range(n - 1, -1, -1):
            if self._hybrid_scores[i] >= self.hybrid_threshold:
                keep = n - i
            else:
                break
        if keep == 0:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._audio_frames[n - keep :]).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  PostRollBuffer  (new — live mirror of _compute_postroll())
# ════════════════════════════════════════════════════════════════════════════
class PostRollBuffer:
    """
    Collects audio that arrives *after* speech probability drops below
    threshold and decides — frame by frame — whether it still belongs to
    the current speech segment.

    Why we need it
    ──────────────
    When someone finishes a sentence their voice doesn't cut off instantly.
    There are trailing fricatives, plosive releases, and natural reverberation.
    Without a post-roll, those trailing sounds are either clipped (if we close
    the segment immediately) or padded with too much silence (if we wait a
    fixed duration).

    How it works
    ────────────
    Once the state machine enters POST-ROLL:
      1. Each new chunk is pushed here via push().
      2. A per-frame hybrid score is computed (same formula as PreRollBuffer).
      3. is_done() returns True when the score has stayed below
         hybrid_threshold for at least min_silence_frames consecutive frames,
         meaning the trailing energy has genuinely died out.
      4. If the score rises again (score ≥ hybrid_threshold), should_resume()
         returns True — the state machine re-enters SPEECH and the buffered
         audio is folded back into speech_buf so nothing is lost.
      5. get_postroll() returns all buffered audio to be appended to the
         segment before emitting.

    Max-duration guard
    ──────────────────
    If the speaker never pauses (or the post-roll fills up), the caller
    should fall through to the max_speech_frames check and force-emit.
    """

    FRAME_SAMPLES = 160  # 10 ms @ 16 kHz

    def __init__(
        self,
        max_postroll_sec: float = DEFAULT_POSTROLL_MAX_SEC,
        hybrid_threshold: float = DEFAULT_POSTROLL_HYBRID_THRESHOLD,
        prob_weight: float = DEFAULT_PROB_WEIGHT,
        rms_weight: float = DEFAULT_RMS_WEIGHT,
        min_silence_frames: int = 25,  # filled in by LiveVADCapture
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.max_postroll_sec = max_postroll_sec
        self.hybrid_threshold = hybrid_threshold
        self.prob_weight = prob_weight
        self.rms_weight = rms_weight
        self.min_silence_frames = min_silence_frames
        self.sample_rate = sample_rate
        self._max_frames = int(max_postroll_sec * 100)
        self._audio_frames: list[np.ndarray] = []
        self._hybrid_scores: list[float] = []
        self._rms_running_max: float = 1e-6
        # how many consecutive frames have stayed below threshold
        self._low_score_streak: int = 0

    def reset(self) -> None:
        self._audio_frames.clear()
        self._hybrid_scores.clear()
        self._rms_running_max = 1e-6
        self._low_score_streak = 0

    def push(self, audio_chunk: np.ndarray, prob: float) -> None:
        """
        Ingest one chunk that arrived after speech prob dropped.
        Updates the internal hybrid score and silence streak counters.
        """
        if len(audio_chunk) == 0:
            return
        audio_chunk = audio_chunk.astype(np.float32)
        remainder = len(audio_chunk) % self.FRAME_SAMPLES
        if remainder:
            audio_chunk = np.pad(audio_chunk, (0, self.FRAME_SAMPLES - remainder))

        n_frames = len(audio_chunk) // self.FRAME_SAMPLES
        for i in range(n_frames):
            frame = audio_chunk[i * self.FRAME_SAMPLES : (i + 1) * self.FRAME_SAMPLES]
            rms = float(np.sqrt(np.mean(frame**2)))
            self._rms_running_max = max(self._rms_running_max * 0.9999, rms + 1e-10)
            rms_norm = min(rms / self._rms_running_max, 1.0)
            score = self.prob_weight * prob + self.rms_weight * rms_norm
            self._audio_frames.append(frame)
            self._hybrid_scores.append(score)

            if score < self.hybrid_threshold:
                self._low_score_streak += 1
            else:
                self._low_score_streak = 0  # score rose → streak broken

        # Cap buffer at max_postroll_sec worth of frames
        if len(self._audio_frames) > self._max_frames:
            excess = len(self._audio_frames) - self._max_frames
            self._audio_frames = self._audio_frames[excess:]
            self._hybrid_scores = self._hybrid_scores[excess:]

    def is_done(self) -> bool:
        """
        True when the hybrid score has been below threshold for at least
        min_silence_frames consecutive frames — the trailing audio has
        genuinely died out and it's safe to close the segment.
        """
        return self._low_score_streak >= self.min_silence_frames

    def should_resume(self) -> bool:
        """
        True when the most recent frame's score climbed back above threshold,
        meaning the speaker started talking again before the segment closed.
        The state machine should re-enter SPEECH and fold this buffer back in.
        """
        if not self._hybrid_scores:
            return False
        return self._hybrid_scores[-1] >= self.hybrid_threshold

    def get_postroll(self) -> np.ndarray:
        """
        Return all buffered audio to be appended to the speech segment.
        Trims trailing frames whose score is below threshold so we don't
        pad the segment with pure silence.
        """
        n = len(self._audio_frames)
        if n == 0:
            return np.array([], dtype=np.float32)

        # Find the last frame still above threshold — only keep up to there.
        keep_until = 0
        for i in range(n):
            if self._hybrid_scores[i] >= self.hybrid_threshold:
                keep_until = i + 1  # include this frame

        if keep_until == 0:
            return np.array([], dtype=np.float32)

        return np.concatenate(self._audio_frames[:keep_until]).astype(np.float32)

    def drain(self) -> np.ndarray:
        """Return *all* buffered audio (used when force-emitting on max duration)."""
        if not self._audio_frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._audio_frames).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  Default segment handler
# ════════════════════════════════════════════════════════════════════════════
def _default_on_segment(
    audio_np: np.ndarray,
    seg_num: int,
    output_dir: Path,
    save: bool,
) -> None:
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
            torch.from_numpy(audio_np).unsqueeze(0),
            SAMPLE_RATE,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        rel_display = f"{wav_path.parent.name}/{wav_path.name}"
        console.print(
            f"  [dim]saved → [link=file://{wav_path}]{rel_display}[/link][/dim]"
        )

    except Exception as exc:
        console.print(f"  [red]WAV save failed: {exc}[/red]")


# ════════════════════════════════════════════════════════════════════════════
#  Core streaming class
# ════════════════════════════════════════════════════════════════════════════
class LiveVADCapture:
    """
    Captures microphone audio in real time and calls *on_segment* with each
    complete speech segment as a float32 numpy array at 16 kHz.

    Three-state machine
    ───────────────────
    SILENCE   — waiting; PreRollBuffer accumulates look-back audio
    SPEECH    — collecting; speech_buf grows each chunk
    POST-ROLL — trailing; PostRollBuffer decides when the segment truly ends
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
        preroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
        preroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
        postroll_max_sec: float = DEFAULT_POSTROLL_MAX_SEC,
        postroll_hybrid_threshold: float = DEFAULT_POSTROLL_HYBRID_THRESHOLD,
        postroll_prob_weight: float = DEFAULT_PROB_WEIGHT,
        postroll_rms_weight: float = DEFAULT_RMS_WEIGHT,
        output_dir: Path = Path("live_segments"),
        save_wav: bool = True,
    ) -> None:
        self.on_segment = on_segment
        self.device = device
        self.threshold = threshold
        self.output_dir = output_dir
        self.save_wav = save_wav

        self.min_silence_frames = int(min_silence_duration_sec / CHUNK_SEC)
        self.min_speech_frames = int(min_speech_duration_sec / CHUNK_SEC)
        self.max_speech_frames = int(max_speech_duration_sec / CHUNK_SEC)

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2000)

        # ── VAD model ────────────────────────────────────────────────────────
        self._vad = FireRedVAD(
            model_dir=SAVE_DIR,
            threshold=threshold,
            min_silence_duration_sec=min_silence_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec,
            max_speech_duration_sec=max_speech_duration_sec,
            smooth_window_size=smooth_window_size,
            max_buffer_sec=max_buffer_sec,
        )

        # ── pre-roll buffer ──────────────────────────────────────────────────
        self._preroll = PreRollBuffer(
            max_preroll_sec=preroll_max_sec,
            hybrid_threshold=preroll_hybrid_threshold,
            prob_weight=preroll_prob_weight,
            rms_weight=preroll_rms_weight,
        )

        # ── post-roll buffer ─────────────────────────────────────────────────
        self._postroll = PostRollBuffer(
            max_postroll_sec=postroll_max_sec,
            hybrid_threshold=postroll_hybrid_threshold,
            prob_weight=postroll_prob_weight,
            rms_weight=postroll_rms_weight,
            min_silence_frames=self.min_silence_frames,
        )

        # ── state machine ────────────────────────────────────────────────────
        self._state: str = _SILENCE
        self._speech_buf: list[np.ndarray] = []
        self._speech_frame_count: int = 0
        self._seg_num: int = 0

        self._running = False
        self._process_thread: Optional[threading.Thread] = None

    # ── sounddevice callback ─────────────────────────────────────────────────
    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            console.print(f"[yellow]sd status: {status}[/yellow]")
        mono = indata[:, 0].astype(np.float32).copy()
        try:
            self._audio_queue.put_nowait(mono)
        except queue.Full:
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(mono)
            except queue.Empty:
                pass

    # ── processing loop ──────────────────────────────────────────────────────
    def _process_loop(self) -> None:
        """
        Reads chunks from the queue and drives the three-state machine.

        SILENCE
        ───────
        • Push chunk into PreRollBuffer (rolling look-back window).
        • Get speech prob from FireRedVAD.
        • prob ≥ threshold → grab preroll, switch to SPEECH.

        SPEECH
        ──────
        • Append chunk to speech_buf.
        • prob < threshold → reset PostRollBuffer, switch to POST-ROLL.
        • speech_frames ≥ max → force-emit (max duration guard).

        POST-ROLL
        ─────────
        • Push chunk into PostRollBuffer.
        • PostRollBuffer.should_resume() → score rose, back to SPEECH;
          fold buffered postroll audio into speech_buf so nothing is lost.
        • PostRollBuffer.is_done() → score stayed low long enough; emit.
        • len(postroll_buf) ≥ max_postroll → force-emit (overflow guard).
        """
        console.print("[bold cyan]Processing loop started.[/bold cyan]")

        while self._running:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            prob = self._vad.get_speech_prob(chunk)

            # ── SILENCE ──────────────────────────────────────────────────────
            if self._state == _SILENCE:
                self._preroll.push(chunk, prob)
                if prob >= self.threshold:
                    self._state = _SPEECH
                    self._speech_buf.clear()
                    self._speech_frame_count = 0

                    preroll_audio = self._preroll.get_preroll()
                    if len(preroll_audio):
                        self._speech_buf.append(preroll_audio)

                    self._speech_buf.append(chunk)
                    self._speech_frame_count += 1
                    console.print("[dim]▸ speech onset[/dim]")

            # ── SPEECH ───────────────────────────────────────────────────────
            elif self._state == _SPEECH:
                self._speech_buf.append(chunk)
                self._speech_frame_count += 1

                if prob < self.threshold:
                    # Transition into post-roll — don't close yet.
                    self._postroll.reset()
                    self._postroll.push(chunk, prob)
                    self._state = _POSTROLL

                elif self._speech_frame_count >= self.max_speech_frames:
                    # Force-emit: speaker never paused within max duration.
                    console.print(
                        "[yellow]max speech duration reached — force emit[/yellow]"
                    )
                    self._emit_segment(force=True)

            # ── POST-ROLL ────────────────────────────────────────────────────
            elif self._state == _POSTROLL:
                self._postroll.push(chunk, prob)
                self._speech_frame_count += 1

                if self._postroll.should_resume():
                    # Score climbed back up — speaker is still talking.
                    # Fold all buffered post-roll audio back into speech_buf.
                    resumed_audio = self._postroll.drain()
                    if len(resumed_audio):
                        self._speech_buf.append(resumed_audio)
                    self._postroll.reset()
                    self._state = _SPEECH

                elif self._postroll.is_done():
                    # Score stayed low long enough — the segment is finished.
                    self._emit_segment(force=False)

                elif self._speech_frame_count >= self.max_speech_frames:
                    console.print(
                        "[yellow]max speech duration reached in post-roll — force emit[/yellow]"
                    )
                    self._emit_segment(force=True)

        console.print("[bold cyan]Processing loop stopped.[/bold cyan]")

    # ── emit ─────────────────────────────────────────────────────────────────
    def _emit_segment(self, *, force: bool = False) -> None:
        """
        Finalise and emit the current speech segment.

        If force=False (normal close): append the trimmed post-roll audio
        (frames that were still above threshold) before concatenating.
        If force=True (max-duration guard): append all post-roll audio so
        nothing is dropped mid-word.
        """
        if not self._speech_buf and not force:
            self._reset_state()
            return

        # Append post-roll tail
        tail = self._postroll.drain() if force else self._postroll.get_postroll()
        if len(tail):
            self._speech_buf.append(tail)

        audio_np = (
            np.concatenate(self._speech_buf).astype(np.float32)
            if self._speech_buf
            else np.array([], dtype=np.float32)
        )

        self._reset_state()

        # Discard very short blips (mic noise, etc.)
        if len(audio_np) < int(0.1 * SAMPLE_RATE):
            return

        self._seg_num += 1
        if self.on_segment is not None:
            try:
                self.on_segment(audio_np, self._seg_num)
            except Exception as exc:
                console.print(f"[red]on_segment error: {exc}[/red]")
        else:
            _default_on_segment(audio_np, self._seg_num, self.output_dir, self.save_wav)

    def _reset_state(self) -> None:
        """Return to SILENCE and clear all per-segment counters."""
        self._state = _SILENCE
        self._speech_buf.clear()
        self._speech_frame_count = 0
        self._postroll.reset()
        self._preroll.reset()

    # ── public API ───────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._process_thread = threading.Thread(
            target=self._process_loop, name="vad-process", daemon=True
        )
        self._process_thread.start()

        console.print(
            f"[bold green]🎙  Listening on device "
            f"[yellow]{self.device if self.device is not None else 'default'}[/yellow]"
            f" @ {SAMPLE_RATE} Hz  (Ctrl+C to stop)[/bold green]"
        )

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
        self._running = False
        if self._process_thread is not None:
            self._process_thread.join(timeout=3.0)

        # Flush whatever state we're in
        if self._state == _SPEECH and self._speech_buf:
            console.print("[yellow]Flushing SPEECH segment on stop…[/yellow]")
            self._emit_segment(force=True)
        elif self._state == _POSTROLL and self._speech_buf:
            console.print("[yellow]Flushing POST-ROLL segment on stop…[/yellow]")
            self._emit_segment(force=False)


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════
def _list_devices() -> None:
    console.print(sd.query_devices())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live VAD capture — FireRedVAD + hybrid pre-roll + post-roll"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="print available audio devices and exit",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="sounddevice input device index"
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-silence", type=float, default=DEFAULT_MIN_SILENCE_SEC)
    parser.add_argument("--min-speech", type=float, default=DEFAULT_MIN_SPEECH_SEC)
    parser.add_argument("--max-speech", type=float, default=DEFAULT_MAX_SPEECH_SEC)
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW_SIZE)
    parser.add_argument(
        "--preroll-max-sec", type=float, default=DEFAULT_PREROLL_MAX_SEC
    )
    parser.add_argument(
        "--preroll-threshold", type=float, default=DEFAULT_PREROLL_HYBRID_THRESHOLD
    )
    parser.add_argument(
        "--preroll-prob-weight", type=float, default=DEFAULT_PROB_WEIGHT
    )
    parser.add_argument("--preroll-rms-weight", type=float, default=DEFAULT_RMS_WEIGHT)
    parser.add_argument(
        "--postroll-max-sec", type=float, default=DEFAULT_POSTROLL_MAX_SEC
    )
    parser.add_argument(
        "--postroll-threshold", type=float, default=DEFAULT_POSTROLL_HYBRID_THRESHOLD
    )
    parser.add_argument(
        "--postroll-prob-weight", type=float, default=DEFAULT_PROB_WEIGHT
    )
    parser.add_argument("--postroll-rms-weight", type=float, default=DEFAULT_RMS_WEIGHT)
    parser.add_argument(
        "-o",
        "--output-dir",
        default=OUTPUT_DIR,
        type=Path,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="print info but don't write WAV files"
    )
    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
        preroll_prob_weight=args.preroll_prob_weight,
        preroll_rms_weight=args.preroll_rms_weight,
        postroll_max_sec=args.postroll_max_sec,
        postroll_hybrid_threshold=args.postroll_threshold,
        postroll_prob_weight=args.postroll_prob_weight,
        postroll_rms_weight=args.postroll_rms_weight,
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
