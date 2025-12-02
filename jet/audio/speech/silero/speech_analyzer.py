# jet_python_modules/jet/audio/speech/silero/silero_vad_stream.py
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

# Proper rich logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("silero-vad-stream")


# Lazy load model only when needed
def _load_silero_vad() -> tuple:
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
        verbose=False,
    )
    get_speech_timestamps, save_audio, _, VADIterator, _ = utils
    return model, VADIterator, save_audio


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
        block_size: Optional[int] = None,
        on_speech_start: Optional[Callable[[float], None]] = None,
        on_speech_end: Optional[Callable[[SpeechSegment], None]] = None,
        output_dir: Optional[Path | str] = None,
        save_segments: bool = True,
        debug: bool = False,
    ):
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD only supports 8000 or 16000 Hz")
        self.sample_rate = sample_rate
        self.block_size = block_size or (512 if sample_rate == 16000 else 256)
        if self.block_size not in (256, 512):
            raise ValueError(f"block_size must be 256 (8k) or 512 (16k), got {self.block_size}")

        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.device = device
        self.on_speech_start = on_speech_start or self._default_start_handler
        self.on_speech_end = on_speech_end or self._default_end_handler
        self.debug = debug
        if debug:
            log.setLevel(logging.DEBUG)

        self.output_dir = Path(output_dir) if output_dir else None
        self.save_segments = save_segments and bool(self.output_dir)
        if self.save_segments:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._segment_counter = 0

        # Lazy load
        self.model, VADIterator, self.save_audio = _load_silero_vad()

        self.vad_iterator = VADIterator(
            model=self.model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

        # State
        self._current_start: Optional[float] = None
        self._current_start_sample: Optional[int] = None
        self._total_samples_processed: int = 0           # global time
        self._buffer_start_sample: int = 0                # start offset after trims
        self._audio_buffer: deque[torch.Tensor] = deque()
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()

    def _default_start_handler(self, timestamp: float) -> None:
        log.info(f"[green]Speech Start[/] @ {timestamp:.3f}s")

    def _default_end_handler(self, segment: SpeechSegment) -> None:
        log.info(
            f"[bold magenta]Speech End[/] @ {segment.end_sec:.3f}s "
            f"[cyan]dur={segment.duration():.3f}s[/]"
        )

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            log.warning(f"Audio callback status: {status}")
        chunk = torch.from_numpy(indata.copy()).squeeze(1).float()

        with self._lock:
            self._audio_buffer.append(chunk)
            self._total_samples_processed += len(chunk)

        result = self.vad_iterator(chunk, return_seconds=True)

        if result is None:
            return

        if "start" in result:
            start_sec = result["start"]
            self._current_start = start_sec
            self._current_start_sample = int(start_sec * self.sample_rate)
            self.on_speech_start(start_sec)

        elif "end" in result and self._current_start is not None:
            end_sec = result["end"]
            segment = SpeechSegment(
                start_sample=self._current_start_sample,
                end_sample=int(end_sec * self.sample_rate),
                start_sec=self._current_start,
                end_sec=end_sec,
                duration_sec=end_sec - self._current_start,
            )
            self._save_and_notify(segment)
            self._current_start = None
            self._current_start_sample = None

        # Periodic trim
        if self._total_samples_processed % (self.sample_rate * 10) < self.block_size:
            self._trim_buffer(keep_seconds=60)

    def _save_and_notify(self, segment: SpeechSegment) -> None:
        if self.save_segments:
            audio_tensor = self._extract_segment_audio(segment.start_sample, segment.end_sample)
            if audio_tensor is not None:
                self._segment_counter += 1
                seg_dir = self.output_dir / f"segment_{self._segment_counter:03d}"
                seg_dir.mkdir(parents=True, exist_ok=True)
                wav_path = seg_dir / "sound.wav"
                json_path = seg_dir / "segment.json"
                self.save_audio(str(wav_path), audio_tensor.unsqueeze(0), self.sample_rate)
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump({
                        "start_sample": segment.start_sample,
                        "end_sample": segment.end_sample,
                        "start_sec": round(segment.start_sec, 3),
                        "end_sec": round(segment.end_sec, 3),
                        "duration_sec": round(segment.duration_sec, 3),
                    }, f, indent=2)
                log.info(f"[bold green]Saved segment:[/] {seg_dir.name}")
        self.on_speech_end(segment)

    def _extract_segment_audio(self, start_sample: int, end_sample: int) -> Optional[torch.Tensor]:
        if start_sample >= end_sample:
            return None
        target_len = end_sample - start_sample
        extracted = torch.zeros(target_len, dtype=torch.float32)
        pos = self._buffer_start_sample
        out_idx = 0

        with self._lock:
            buffer_copy = list(self._audio_buffer)

        for chunk in buffer_copy:
            chunk_len = len(chunk)
            chunk_start = pos
            chunk_end = pos + chunk_len

            overlap_start = max(chunk_start, start_sample)
            overlap_end = min(chunk_end, end_sample)

            if overlap_start < overlap_end:
                rel_start = overlap_start - chunk_start
                rel_end = overlap_end - chunk_start
                seg = chunk[rel_start:rel_end]
                seg_len = len(seg)
                extracted[out_idx: out_idx + seg_len] = seg
                out_idx += seg_len
                if out_idx >= target_len:
                    break
            pos += chunk_len

        return extracted if out_idx == target_len else None

    def _trim_buffer(self, keep_seconds: int = 60) -> None:
        keep_samples = int(keep_seconds * self.sample_rate)
        samples_to_keep = max(0, self._total_samples_processed - keep_samples)
        
        with self._lock:
            while self._audio_buffer and self._buffer_start_sample + len(self._audio_buffer[0]) <= samples_to_keep:
                discarded_chunk = self._audio_buffer.popleft()
                self._buffer_start_sample += len(discarded_chunk)

            if self._audio_buffer and self._buffer_start_sample < samples_to_keep:
                keep_in_first = samples_to_keep - self._buffer_start_sample
                self._audio_buffer[0] = self._audio_buffer[0][keep_in_first:]
                self._buffer_start_sample += keep_in_first

            current_buffered_sec = (self._total_samples_processed - self._buffer_start_sample) / self.sample_rate
            if self.debug:
                log.debug(f"[dim]Buffer:[/] {current_buffered_sec:.2f}s kept ({len(self._audio_buffer)} chunks)")

    def _flush_final_segment(self) -> None:
        if self._current_start is not None:
            end_sec = self._total_samples_processed / self.sample_rate
            segment = SpeechSegment(
                start_sample=self._current_start_sample,
                end_sample=self._total_samples_processed,
                start_sec=self._current_start,
                end_sec=end_sec,
                duration_sec=end_sec - self._current_start,
            )
            self._save_and_notify(segment)
            self._current_start = None
            self._current_start_sample = None

    # Context manager support
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self) -> None:
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
        self._stream.start()
        self._stopped.clear()

    def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        self._flush_final_segment()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        log.info("Stream stopped gracefully.")

    def run_forever(self) -> None:
        """Blocking run (original behavior)"""
        signal.signal(signal.SIGINT, lambda s, f: self.stop())
        try:
            while not self._stopped.wait(0.1):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


if __name__ == "__main__":
    import shutil

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    with SileroVADStreamer(
        output_dir=OUTPUT_DIR,
        save_segments=True,
        debug=True,
        min_silence_duration_ms=700,
        threshold=0.6,
    ) as streamer:
        streamer.run_forever()