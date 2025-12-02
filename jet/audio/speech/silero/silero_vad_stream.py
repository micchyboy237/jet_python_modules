# silero_vad_stream.py
from __future__ import annotations
import signal
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import sounddevice as sd
import torch
import json
import logging
from rich.logging import RichHandler

# ────────────── Logging Setup (replacing global Console) ──────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("silero_vad")

model, utils = torch.hub.load(  # pyright: ignore[reportGeneralTypeIssues]
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
    verbose=False,
)

(
    get_speech_timestamps,
    save_audio,   # ← now used
    _,
    VADIterator,
    _,
) = utils


@dataclass
class SpeechSegment:
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float

    def duration(self) -> float:
        return self.duration_sec


class SileroVADStreamer:
    def __init__(
        self,
        threshold: float = 0.6,
        sample_rate: int = 16000,
        min_silence_duration_ms: int = 700,
        speech_pad_ms: int = 30,
        device: Optional[int] = None,
        block_size: int = 512,
        on_speech_start: Optional[Callable[[float], None]] = None,
        on_speech_end: Optional[Callable[[SpeechSegment], None]] = None,
        output_dir: Optional[Path | str] = None,           # ← NEW
        save_segments: bool = True,                        # ← NEW: toggle
        debug: bool = False,                               # ← NEW: enable verbose buffer logging
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.device = device
        self.on_speech_start = on_speech_start or self._default_start_handler
        self.on_speech_end = on_speech_end or self._default_end_handler
        self.debug = debug
        if debug:
            log.setLevel(logging.DEBUG)

        # Saving options
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_segments = save_segments and bool(self.output_dir)
        if self.save_segments:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._segment_counter = 0

        self.vad_iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        # Streaming state
        self._current_start: Optional[float] = None
        self._current_start_sample: Optional[int] = None
        self._total_samples: int = 0
        self._audio_buffer = deque()  # holds torch tensors (float32, mono)

        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def _default_start_handler(self, timestamp: float) -> None:
        log.info(f"[green]Speech Start[/] @ {timestamp:.3f}s")

    def _default_end_handler(self, segment: SpeechSegment) -> None:
        log.info(
            f"[bold magenta]Speech End[/] @ {segment.end_sec:.3f}s "
            f"[cyan]dur={segment.duration():.3f}s[/]"
        )

    def _audio_callback(self, indata, frames, time, status):
        if status:
            log.warning(f"Audio warning: {status}")

        # Convert and store chunk
        chunk = torch.from_numpy(indata.copy()).squeeze(1).float()  # (samples,)
        with self._lock:
            self._audio_buffer.append(chunk)
            self._total_samples += len(chunk)

            result = self.vad_iterator(chunk, return_seconds=True)

        if result is None:
            return

        if "start" in result:
            self._current_start = result["start"]
            self._current_start_sample = int(result["start"] * self.sample_rate)
            self.on_speech_start(result["start"])

        elif "end" in result and self._current_start is not None:
            end_sec = result["end"]
            end_sample_global = int(end_sec * self.sample_rate)

            start_sample = self._current_start_sample
            end_sample = end_sample_global

            duration_sec = end_sec - self._current_start

            segment = SpeechSegment(
                start_sample=start_sample,
                end_sample=end_sample,
                start_sec=self._current_start,
                end_sec=end_sec,
                duration_sec=duration_sec,
            )

            # Extract and save audio if enabled
            if self.save_segments:
                audio_tensor = self._extract_segment_audio(start_sample, end_sample)
                if audio_tensor is not None:
                    self._segment_counter += 1
                    seg_dir = self.output_dir / f"segment_{self._segment_counter:03d}"
                    seg_dir.mkdir(parents=True, exist_ok=True)

                    wav_path = seg_dir / "sound.wav"
                    json_path = seg_dir / "segment.json"

                    save_audio(str(wav_path), audio_tensor, self.sample_rate)

                    with json_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "start_sample": segment.start_sample,
                                "end_sample": segment.end_sample,
                                "start_sec": segment.start_sec,
                                "end_sec": segment.end_sec,
                                "duration_sec": segment.duration_sec,
                            },
                            f,
                            indent=2,
                        )

                    log.info(f"[bold green]Saved segment:[/] {seg_dir.name}")

            self.on_speech_end(segment)

            # Reset
            self._current_start = None
            self._current_start_sample = None

        # Periodic lazy cleanup – keeps memory bounded but never drops active speech
        if self._total_samples % (self.sample_rate * 10) == 0:  # every ~10 s
            self._trim_buffer_lazy(keep_seconds=60)

    def _extract_segment_audio(self, start_sample: int, end_sample: int) -> Optional[torch.Tensor]:
        """Extract exact speech segment from ring buffer without full concatenation."""
        if start_sample < 0 or end_sample > self._total_samples or start_sample >= end_sample:
            return None

        target_len = end_sample - start_sample
        extracted = torch.zeros(target_len, dtype=torch.float32)

        with self._lock:
            current_pos = self._total_samples - sum(len(c) for c in self._audio_buffer)
            buffer_copy = list(self._audio_buffer)  # shallow copy of references

        pos_in_buffer = 0
        out_idx = 0
        for chunk in buffer_copy:
            chunk_len = len(chunk)
            chunk_start = pos_in_buffer
            chunk_end = pos_in_buffer + chunk_len

            overlap_start = max(chunk_start, start_sample)
            overlap_end = min(chunk_end, end_sample)

            if overlap_start < overlap_end:
                rel_start = overlap_start - chunk_start
                rel_end = overlap_end - chunk_start
                seg = chunk[rel_start:rel_end]
                seg_len = len(seg)
                extracted[out_idx : out_idx + seg_len] = seg
                out_idx += seg_len

            pos_in_buffer += chunk_len
            if out_idx >= target_len:
                break

        return extracted if out_idx == target_len else None

    def _trim_buffer_lazy(self, keep_seconds: int = 60) -> None:
        """
        Lazily discard very old audio (older than `keep_seconds`).
        This guarantees that any active or recently finished speech segment
        is still fully present in the buffer when we extract it.
        """
        with self._lock:
            max_samples = int(keep_seconds * self.sample_rate)
            samples_to_discard = max(0, self._total_samples - max_samples)
            old_total = self._total_samples
            old_chunks = len(self._audio_buffer)
            discarded = 0
            while self._audio_buffer and discarded < samples_to_discard:
                chunk = self._audio_buffer[0]
                if len(chunk) + discarded <= samples_to_discard:
                    discarded += len(chunk)
                    self._audio_buffer.popleft()
                else:
                    # cut the head of the oldest chunk
                    cut = samples_to_discard - discarded
                    self._audio_buffer[0] = chunk[cut:]
                    discarded += cut
                    break
            # update total sample counter accordingly
            self._total_samples -= discarded

            # ────── Improved Buffer Logging ──────
            if discarded > 0 or self.debug:
                current_sec = self._total_samples / self.sample_rate
                chunks = len(self._audio_buffer)
                if discarded > 0:
                    log.info(
                        f"[dim]Buffer trimmed:[/] −{discarded / self.sample_rate:.2f}s "
                        f"({discarded} samples) → "
                        f"[bold]{current_sec:.2f}s[/] kept "
                        f"({chunks} chunks)"
                    )
                elif self.debug and old_total % (self.sample_rate * 30) < (len(self._audio_buffer[0]) if self._audio_buffer else 1):
                    # Every ~30s in debug mode, show status even if no trim
                    log.debug(
                        f"[dim]Buffer:[/] {current_sec:.2f}s "
                        f"({self._total_samples} samples, {chunks} chunks) – no trim needed"
                    )

    def _signal_handler(self, sig, frame):
        log.info("\nShutting down gracefully...")
        with self._lock:
            silent = torch.zeros(self.block_size)
            final = self.vad_iterator(silent, return_seconds=True)
            if final and "end" in final and self._current_start is not None:
                end_sec = final["end"]
                end_sample = int(end_sec * self.sample_rate)
                start_sample = self._current_start_sample
                duration_sec = end_sec - self._current_start
                segment = SpeechSegment(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_sec=self._current_start,
                    end_sec=end_sec,
                    duration_sec=duration_sec,
                )
                if self.save_segments:
                    audio_tensor = self._extract_segment_audio(start_sample, end_sample)
                    if audio_tensor is not None:
                        self._segment_counter += 1
                        seg_dir = self.output_dir / f"segment_{self._segment_counter:03d}"
                        seg_dir.mkdir(parents=True, exist_ok=True)

                        wav_path = seg_dir / "sound.wav"
                        json_path = seg_dir / "segment.json"

                        save_audio(str(wav_path), audio_tensor, self.sample_rate)

                        with json_path.open("w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "start_sample": segment.start_sample,
                                    "end_sample": segment.end_sample,
                                    "start_sec": segment.start_sec,
                                    "end_sec": segment.end_sec,
                                    "duration_sec": segment.duration_sec,
                                },
                                f,
                                indent=2,
                            )

                        log.info(f"[bold green]Saved final segment:[/] {seg_dir.name}")
                self.on_speech_end(segment)

    def start(self) -> None:
        """Start the microphone stream. Blocks until Ctrl+C."""
        log.info(f"Starting Silero VAD streamer • sr={self.sample_rate} • block={self.block_size}")
        log.info("Press Ctrl+C to stop.\n")
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        with self._stream:
            signal.signal(signal.SIGINT, self._signal_handler)
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                pass
        log.info("\nStream stopped.")

if __name__ == "__main__":
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    streamer = SileroVADStreamer(
        output_dir=OUTPUT_DIR,   # ← enables saving
        save_segments=True,
        debug=True,
    )
    streamer.start()