#!/usr/bin/env python3
"""
jp_to_en_realtime.py
Real-time Japanese → English translation with logging + tqdm progress
Saves .txt + .srt files • Fully local • Apple Silicon optimized
"""

import argparse
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Iterator, TypedDict

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from tqdm import tqdm

from jet_python_modules.jet.audio.utils import get_input_channels

# ============================= CONFIG =============================
class AudioConfig(TypedDict):
    device: str | None
    samplerate: int
    chunk_duration: float
    channels: int


class TranscriberConfig(TypedDict):
    model_size: str
    compute_type: str
    language: str
    task: str
    device: str


# ============================= LOGGER SETUP =============================
def setup_logger(log_dir: Path, quiet: bool) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("jp_to_en")
    logger.setLevel(logging.DEBUG if not quiet else logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if quiet else logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_dir / "session.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ============================= MAIN CLASS =============================
class JapaneseTranscriber:
    def __init__(
        self,
        audio_cfg: AudioConfig,
        trans_cfg: TranscriberConfig,
        output_dir: Path,
        logger: logging.Logger,
        show_progress: bool = True,
    ):
        self.audio_cfg = audio_cfg
        self.trans_cfg = trans_cfg
        self.output_dir = output_dir
        self.logger = logger
        self.show_progress = show_progress

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.txt_file = output_dir / f"translation_{timestamp}.txt"
        self.srt_file = output_dir / f"subtitles_{timestamp}.srt"

        self.txt_handle = open(self.txt_file, "w", encoding="utf-8")
        self.srt_handle = open(self.srt_file, "w", encoding="utf-8")
        self.srt_counter = 1
        self.session_start = datetime.now()

        self.logger.info("Session started")
        self.logger.info(f"Saving to: {self.output_dir.resolve()}")
        self.logger.info(f"Model: {trans_cfg['model_size']} | Chunk: {audio_cfg['chunk_duration']}s")

        # Dynamic device detection for Apple Silicon (MPS unsupported in faster-whisper)
        machine = platform.machine().lower()
        if "arm" in machine or "aarch64" in machine:
            self.trans_cfg["device"] = "cpu"  # Force CPU on Apple Silicon
            self.trans_cfg["compute_type"] = "float32"  # Stable for ARM CPU
            self.logger.info("Detected Apple Silicon: Using CPU (faster-whisper MPS unsupported)")
        else:
            self.trans_cfg["device"] = "cuda" if "cuda" in trans_cfg.get("device", "") else "cpu"
            self.logger.info(f"Using device: {self.trans_cfg['device']}")

        try:
            self.model = WhisperModel(
                trans_cfg["model_size"],
                device=self.trans_cfg["device"],
                compute_type=self.trans_cfg["compute_type"],
            )
            self.logger.info("Model loaded successfully")
        except ValueError as e:
            self.logger.error(f"Model load failed: {e}. Falling back to CPU/float32.")
            self.trans_cfg["device"] = "cpu"
            self.trans_cfg["compute_type"] = "float32"
            self.model = WhisperModel(
                trans_cfg["model_size"],
                device="cpu",
                compute_type="float32",
            )
            self.logger.info("Fallback model loaded")

        self.queue: Queue[np.ndarray] = Queue(maxsize=15)
        self.running = False
        self.processed_chunks = 0

        # tqdm progress bar
        self.pbar = tqdm(
            desc="Transcribing",
            unit="chunk",
            dynamic_ncols=True,
            colour="cyan",
            disable=not show_progress,
        )

    def _seconds_to_srt(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _save(self, text: str, start: float, end: float) -> None:
        self.txt_handle.write(text + "\n")
        self.txt_handle.flush()

        self.srt_handle.write(f"{self.srt_counter}\n")
        self.srt_handle.write(f"{self._seconds_to_srt(start)} --> {self._seconds_to_srt(end)}\n")
        self.srt_handle.write(f"{text}\n\n")
        self.srt_handle.flush()
        self.srt_counter += 1

    def _callback(self, text: str, start: float, end: float) -> None:
        print(f"→ {text}")
        self._save(text, start, end)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            self.logger.warning(f"Audio buffer overrun: {status}")
        audio = np.mean(indata, axis=1).astype(np.float32) if indata.ndim > 1 else indata.astype(np.float32)
        try:
            self.queue.put_nowait(audio)
        except:
            self.logger.debug("Queue full – dropping audio frame")

    def _worker(self) -> Iterator[tuple[str, float, float]]:
        chunk_offset = 0.0
        while self.running:
            try:
                audio = self.queue.get(timeout=1.0)
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error getting audio chunk from queue: {e}")
                continue

            try:
                self.processed_chunks += 1
                self.pbar.update(1)
                self.pbar.set_postfix(
                    {"chunks": self.processed_chunks, "elapsed": str(datetime.now() - self.session_start).split('.')[0]}
                )

                segments, _ = self.model.transcribe(
                    audio,
                    language=self.trans_cfg["language"],
                    task=self.trans_cfg["task"],
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    temperature=0.0,
                )

                for seg in segments:
                    if seg.text.strip():
                        yield seg.text.strip(), chunk_offset + seg.start, chunk_offset + seg.end

                # Always advance time — silence is part of the stream
                chunk_offset += len(audio) / self.audio_cfg["samplerate"]

            except Exception as e:
                self.logger.error(f"Transcription failed: {e}")
            finally:
                # Exactly one task_done() per successful get()
                self.queue.task_done()

    def start(self) -> None:
        self.running = True
        stream = sd.InputStream(
            samplerate=self.audio_cfg["samplerate"],
            device=self.audio_cfg["device"],
            channels=self.audio_cfg["channels"],
            blocksize=int(self.audio_cfg["samplerate"] * self.audio_cfg["chunk_duration"]),
            callback=self._audio_callback,
            dtype=self.trans_cfg["compute_type"],
        )

        with stream:
            self.logger.info("Listening... (Ctrl+C to stop)")
            # ← Fixed lambda: consume generator safely without list-comprehension
            def worker_target():
                for text, start, end in self._worker():
                    self._callback(text, start, end)

            worker = Thread(target=worker_target, daemon=True)
            worker.start()

            try:
                while self.running:
                    sd.sleep(100)
            except KeyboardInterrupt:
                self.logger.info("Shutting down...")
            finally:
                self.running = False
                self.pbar.close()
                self.txt_handle.close()
                self.srt_handle.close()
                self.logger.info(f"Session ended • Saved to {self.output_dir.resolve()}")


CHANNELS = min(2, get_input_channels())

print(f"Channels: {CHANNELS}")

# ============================= MAIN =============================
def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time Japanese → English with logging + progress")
    parser.add_argument("--device", type=int, default=None, help="Audio device index listed in sd.query_devices()")
    parser.add_argument("--model", type=str, default="turbo", choices=["tiny", "base", "small", "medium", "large-v3", "turbo"])
    parser.add_argument("--chunk", type=float, default=3.0, help="Chunk duration in seconds")
    parser.add_argument("--output-dir", type=str, default="./translations", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Minimal console output")
    parser.add_argument("--no-progress", action="store_true", help="Hide tqdm progress bar")
    args = parser.parse_args()

    output_path = Path(args.output_dir).expanduser().resolve()
    logger = setup_logger(output_path, quiet=args.quiet)

    audio_cfg: AudioConfig = {
        "device": args.device,
        "samplerate": 16000,
        "chunk_duration": args.chunk,
        "channels": CHANNELS,
    }

    # Initial compute_type (overridden in __init__ for Apple Silicon)
    compute_type = "int8" if args.model in ("tiny", "base") else "float16"

    trans_cfg: TranscriberConfig = {
        "model_size": args.model,
        "compute_type": compute_type,
        "language": "ja",
        "task": "translate",
        "device": "auto",  # Placeholder; detected in __init__
    }

    transcriber = JapaneseTranscriber(
        audio_cfg=audio_cfg,
        trans_cfg=trans_cfg,
        output_dir=output_path,
        logger=logger,
        show_progress=not args.no_progress and not args.quiet,
    )
    transcriber.start()


if __name__ == "__main__":
    main()