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
import logging
import json
from rich.logging import RichHandler

# ────────────── Logging Setup ──────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
log = logging.getLogger("silero_vad")

model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
    verbose=False,
)

(
    _,
    save_audio,
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
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        device: Optional[int] = None,
        block_size: int = 512,
        on_speech_start: Optional[Callable[[float], None]] = None,
        on_speech_end: Optional[Callable[[SpeechSegment], None]] = None,
        debug: bool = False,
        output_dir: Optional[Path | str] = None,
        save_segments: bool = False,
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

        # Saving setup
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_segments = save_segments and bool(self.output_dir)
        if self.save_segments:
            self.segments_dir = self.output_dir / "segments"
            self.segments_dir.mkdir(parents=True, exist_ok=True)
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
        log.info(f"[bold green]🎤 Speech START detected[/] @ {timestamp:.3f}s")

    def _default_end_handler(self, segment: SpeechSegment) -> None:
        log.info(
            f"[bold magenta]🔇 Speech END[/] @ {segment.end_sec:.3f}s "
            f"[cyan]duration: {segment.duration():.3f}s[/]"
        )

    def _extract_segment_audio(self, start_sample: int, end_sample: int) -> Optional[torch.Tensor]:
        """Extract audio for the given sample range from the circular buffer."""
        if start_sample < 0 or end_sample > self._total_samples or start_sample >= end_sample:
            return None

        target_len = end_sample - start_sample
        extracted = torch.zeros(target_len, dtype=torch.float32)

        with self._lock:
            # Calculate absolute position of the oldest sample in buffer
            buffer_start_sample = self._total_samples - sum(len(c) for c in self._audio_buffer)
            buffer_copy = list(self._audio_buffer)

        pos_in_buffer = buffer_start_sample
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

    def _save_current_segment(self, segment: SpeechSegment) -> None:
        """Save the detected segment immediately to disk (audio + metadata)."""
        if not self.save_segments:
            return

        audio_tensor = self._extract_segment_audio(segment.start_sample, segment.end_sample)
        if audio_tensor is None:
            log.warning("Failed to extract audio for segment – skipping save")
            return

        self._segment_counter += 1
        seg_dir = self.segments_dir / f"segment_{self._segment_counter:04d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Save WAV
        wav_path = seg_dir / "sound.wav"
        save_audio(str(wav_path), audio_tensor.unsqueeze(0), sampling_rate=self.sample_rate)

        # Save metadata JSON
        metadata = {
            "segment_id": self._segment_counter,
            "start_sec": round(segment.start_sec, 6),
            "end_sec": round(segment.end_sec, 6),
            "duration_sec": round(segment.duration_sec, 6),
            "start_sample": segment.start_sample,
            "end_sample": segment.end_sample,
        }
        metadata_path = seg_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        log.info(f"[bold green]💾 Saved segment[/] → {seg_dir.name} (duration {segment.duration():.3f}s)")

    def _audio_callback(self, indata, frames, time, status):
        if status:
            log.warning(f"Audio warning: {status}")

        chunk = torch.from_numpy(indata.copy()).squeeze(1).float()

        with self._lock:
            self._audio_buffer.append(chunk)
            self._total_samples += len(chunk)

        # Instantaneous probability (for debug only)
        speech_prob = self.vad_iterator.model(chunk, self.sample_rate).item()

        result = self.vad_iterator(chunk, return_seconds=True)

        if self.debug:
            triggered = getattr(self.vad_iterator, "triggered", False)
            status_text = "[bold green]SPEECH[/]" if triggered else "[dim]silence[/]"
            current_sec = self._total_samples / self.sample_rate
            log.debug(
                f"[dim]Block[/] @ {current_sec:.3f}s | prob={speech_prob:.3f} | {status_text}"
            )

        if result is None:
            return

        if "start" in result:
            self._current_start = result["start"]
            self._current_start_sample = int(result["start"] * self.sample_rate)
            self.on_speech_start(result["start"])

        elif "end" in result and self._current_start is not None:
            end_sec = result["end"]
            end_sample = int(end_sec * self.sample_rate)

            segment = SpeechSegment(
                start_sample=self._current_start_sample,
                end_sample=end_sample,
                start_sec=self._current_start,
                end_sec=end_sec,
                duration_sec=end_sec - self._current_start,
            )

            # Save immediately on end detection
            self._save_current_segment(segment)

            self.on_speech_end(segment)

            # Reset for next segment
            self._current_start = None
            self._current_start_sample = None

    def _signal_handler(self, sig, frame):
        log.info("\n[bold yellow]Shutting down gracefully...[/]")
        self.stop()

    def start(self) -> None:
        log.info(f"[bold blue]Starting Silero VAD streamer[/] • sample_rate={self.sample_rate} • block_size={self.block_size}")
        if self.save_segments:
            log.info(f"[bold blue]Segments will be saved to:[/] {self.segments_dir}")
        log.info("[bold blue]Listening for speech...[/] (Press Ctrl+C to stop)")

        def _run_audio_stream():
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.device,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            with self._stream:
                log.info("[bold green]✅ Audio stream started – microphone active[/]")
                while True:
                    sd.sleep(1000)

        threading.Thread(target=_run_audio_stream, daemon=True).start()
        signal.signal(signal.SIGINT, self._signal_handler)

        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            pass

    def stop(self) -> None:
        if self._stream:
            self._stream.close()
            log.info("[bold red]⏹ Audio stream stopped[/]")

if __name__ == "__main__":
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    streamer = SileroVADStreamer(
        output_dir=OUTPUT_DIR,
        save_segments=True,
        debug=False,  # Set True for per-block debug
    )
    streamer.start()