# jet_python_modules/jet/audio/speech/silero/silero_vad_stream.py
from __future__ import annotations
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Optional
import sounddevice as sd
import torch
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
    verbose=False,
)
(
    get_speech_timestamps,
    _,
    _,
    VADIterator,
    _,
) = utils


@dataclass
class SpeechSegment:
    """Container for detected speech segments with complete timestamp details."""
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    duration_sec: float

    def duration(self) -> float:
        """Legacy method for duration (returns pre-computed value)."""
        return self.duration_sec


class SileroVADStreamer:
    """
    Real-time microphone VAD using Silero (streaming mode).
    Runs forever until SIGINT (Ctrl+C).
    """
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
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.device = device
        self.on_speech_start = on_speech_start or self._default_start_handler
        self.on_speech_end = on_speech_end or self._default_end_handler
        self.vad_iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self._current_start: Optional[float] = None
        self._current_start_sample: Optional[int] = None  # New: Track sample-level start
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def _default_start_handler(self, timestamp: float) -> None:
        print(f"\n[Speech Start] @ {timestamp:.2f}s")

    def _default_end_handler(self, segment: SpeechSegment) -> None:
        print(f"[Speech End] @ {segment.end_sec:.2f}s (duration: {segment.duration():.2f}s)")

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio warning: {status}", file=sys.stderr)
        audio_chunk = torch.from_numpy(indata.copy()).squeeze(1).float()
        with self._lock:
            result = self.vad_iterator(audio_chunk, return_seconds=True)
        if result is None:
            return
        if "start" in result:
            self._current_start = result["start"]
            self._current_start_sample = int(result["start"] * self.sample_rate)  # New: Compute sample
            self.on_speech_start(result["start"])
        elif "end" in result and self._current_start is not None:
            end_sec = result["end"]
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
            self._current_start = None
            self._current_start_sample = None  # Reset
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

    def _signal_handler(self, sig, frame):
        """Graceful shutdown on Ctrl+C"""
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
                self.on_speech_end(segment)


if __name__ == "__main__":
    def my_start(t: float):
        print(f"Custom: Speech began at {t:.2f}s")
    def my_end(seg: SpeechSegment):
        print(f"Custom: Speech segment → {seg.start_sec:.2f}–{seg.end_sec:.2f}s ({seg.duration():.2f}s)")
    streamer = SileroVADStreamer(
        threshold=0.6,
        min_silence_duration_ms=700,
        speech_pad_ms=40,
        on_speech_start=my_start,
        on_speech_end=my_end,
    )
    streamer.start()