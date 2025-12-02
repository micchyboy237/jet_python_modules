# jet_python_modules/jet/audio/speech/silero/silero_vad_stream.py
from __future__ import annotations
import signal
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import sounddevice as sd
import torch
from rich.console import Console

console = Console()

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
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.device = device
        self.on_speech_start = on_speech_start or self._default_start_handler
        self.on_speech_end = on_speech_end or self._default_end_handler

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
        console.print(f"[green]Speech Start[/] @ {timestamp:.2f}s")

    def _default_end_handler(self, segment: SpeechSegment) -> None:
        console.print(
            f"[bold magenta]Speech End[/] @ {segment.end_sec:.2f}s "
            f"([cyan]duration: {segment.duration():.2f}s[/])"
        )

    def _audio_callback(self, indata, frames, time, status):
        if status:
            console.print(f"[yellow]Audio warning:[/] {status}", file=sys.stderr)

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
                    filename = (
                        f"speech_{self._segment_counter:03d}_"
                        f"{segment.start_sec:.2f}s_"
                        f"{segment.duration_sec:.2f}s.wav"
                    )
                    path = self.output_dir / filename
                    save_audio(str(path), audio_tensor, self.sample_rate)
                    console.print(f"[bold green]Saved:[/] {path.name}")

            self.on_speech_end(segment)

            # Reset
            self._current_start = None
            self._current_start_sample = None

            # Clean old buffer (keep only last ~10 seconds to bound memory)
            self._trim_buffer(end_sample_global)

    def _extract_segment_audio(self, start_sample: int, end_sample: int) -> Optional[torch.Tensor]:
        """Extract exact speech segment from ring buffer."""
        if start_sample < 0 or end_sample > self._total_samples:
            return None
        # Rebuild full audio up to current point
        with self._lock:
            full_audio = torch.cat(list(self._audio_buffer), dim=0)
        segment_audio = full_audio[start_sample:end_sample]
        return segment_audio

    def _trim_buffer(self, keep_up_to_sample: int) -> None:
        """Remove old chunks that are no longer needed."""
        with self._lock:
            samples_to_keep = max(0, self._total_samples - keep_up_to_sample + self.sample_rate * 10)
            while self._audio_buffer and samples_to_keep > 0:
                chunk = self._audio_buffer[0]
                if len(chunk) <= samples_to_keep:
                    samples_to_keep -= len(chunk)
                    self._audio_buffer.popleft()
                else:
                    # Partial trim (rare)
                    self._audio_buffer[0] = chunk[len(chunk) - samples_to_keep :]
                    self._total_samples = keep_up_to_sample + self.sample_rate * 10
                    break

    def _signal_handler(self, sig, frame):
        print("\nShutting down...")
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
                        filename = f"speech_final_{segment.start_sec:.2f}s_{segment.duration_sec:.2f}s.wav"
                        path = self.output_dir / filename
                        save_audio(str(path), audio_tensor, self.sample_rate)
                        console.print(f"[bold green]Saved final:[/] {path.name}")
                self.on_speech_end(segment)

    def start(self) -> None:
        """Start the microphone stream. Blocks until Ctrl+C."""
        print(f"Starting Silero VAD streamer (sr={self.sample_rate}, block={self.block_size})")
        print("Press Ctrl+C to stop.\n")
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
        print("\nStream stopped.")

if __name__ == "__main__":
    streamer = SileroVADStreamer(
        threshold=0.5,
        min_silence_duration_ms=500,
        speech_pad_ms=30,
        output_dir="vad_recordings",   # ← enables saving
        save_segments=True,
    )
    streamer.start()