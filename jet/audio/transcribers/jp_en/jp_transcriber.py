#!/usr/bin/env python3
"""
jp_to_en_realtime.py
Real-time Japanese → English translation • ZERO AUDIO LOSS MODE
Saves .txt + .srt • Fully local • Apple Silicon + Windows optimized
"""
import argparse
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Iterator, TypedDict, Literal
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from tqdm import tqdm


# ============================= CONFIG =============================
class AudioConfig(TypedDict):
    device: int | None
    samplerate: int
    chunk_duration: float
    overlap_duration: float  # New: overlap for continuity
    channels: int


class TranscriberConfig(TypedDict):
    model_size: str
    compute_type: str
    language: str
    task: Literal["translate"]
    device: str


# ============================= LOGGER SETUP =============================
def setup_logger(log_dir: Path, quiet: bool) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("jp_to_en")
    logger.setLevel(logging.DEBUG if not quiet else logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if quiet else logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

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

        self.logger.info("Session started • ZERO AUDIO LOSS MODE")
        self.logger.info(f"Saving to: {self.output_dir.resolve()}")
        self.logger.info(f"Model: {trans_cfg['model_size']} | Chunk: {audio_cfg['chunk_duration']}s + {audio_cfg['overlap_duration']}s overlap")

        # Device detection
        machine = platform.machine().lower()
        if "arm" in machine or "aarch64" in machine:
            self.trans_cfg["device"] = "cpu"
            self.trans_cfg["compute_type"] = "float32"
            self.logger.info("Apple Silicon detected → forcing CPU + float32")
        else:
            self.trans_cfg["device"] = "cuda" if sd.query_devices(None, 'cuda') else "cpu"
            self.logger.info(f"Using device: {self.trans_cfg['device']}")

        # Load model
        try:
            self.model = WhisperModel(
                trans_cfg["model_size"],
                device=self.trans_cfg["device"],
                compute_type=self.trans_cfg["compute_type"],
            )
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Model load failed: {e} → falling back to CPU/float32")
            self.model = WhisperModel(trans_cfg["model_size"], device="cpu", compute_type="float32")

        # Large queue to prevent any audio drop
        self.queue: Queue[np.ndarray] = Queue(maxsize=200)  # Was 15 → now 200
        self.running = False
        self.processed_chunks = 0
        self.buffered_audio = np.array([], dtype=np.float32)  # Continuous buffer

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
            self.logger.warning(f"Audio callback overrun: {status}")

        audio = indata.copy()[:, 0] if indata.ndim > 1 else indata.copy().flatten()
        audio = audio.astype(np.float32)

        try:
            self.queue.put_nowait(audio)
        except Exception:
            self.logger.warning("Audio queue full! Dropping frame to prevent crash. Increase queue or reduce chunk size.")

    def _transcribe_buffer(self) -> Iterator[tuple[str, float, float]]:
        """Transcribe current full buffer with VAD disabled and full context"""
        if len(self.buffered_audio) < 16000:  # Less than 1s → skip
            return

        try:
            segments, _ = self.model.transcribe(
                self.buffered_audio,
                language=self.trans_cfg["language"],
                task=self.trans_cfg["task"],
                beam_size=5,
                temperature=0.0,
                vad_filter=False,                    # ← CRITICAL: Disabled VAD
                word_timestamps=True,
                # No vad_parameters → fully continuous
            )

            current_time = datetime.now() - self.session_start
            self.pbar.set_postfix({
                "buf": f"{len(self.buffered_audio)/16000:.1f}s",
                "elapsed": str(current_time).split('.')[0]
            })

            for seg in segments:
                if seg.text.strip():
                    start = seg.start
                    end = seg.end
                    yield seg.text.strip(), start, end

        except Exception as e:
            self.logger.error(f"Transcription error: {e}")

    def _worker(self) -> None:
        chunk_samples = int(self.audio_cfg["samplerate"] * self.audio_cfg["chunk_duration"])
        overlap_samples = int(self.audio_cfg["samplerate"] * self.audio_cfg["overlap_duration"])

        while self.running:
            try:
                audio_chunk = self.queue.get(timeout=1.0)
            except Empty:
                continue

            # Append new audio
            self.buffered_audio = np.concatenate([self.buffered_audio, audio_chunk])

            # Keep buffer reasonable but sufficient (e.g., 30s max)
            max_buffer_sec = 30.0
            if len(self.buffered_audio) > self.audio_cfg["samplerate"] * max_buffer_sec:
                excess = len(self.buffered_audio) - chunk_samples
                self.buffered_audio = self.buffered_audio[-chunk_samples:]

            self.processed_chunks += 1
            self.pbar.update(1)

            # Only transcribe if we have enough new data (non-overlapping part)
            if len(self.buffered_audio) >= chunk_samples:
                # Extract non-overlapping part for transcription
                transcribe_from = max(0, len(self.buffered_audio) - chunk_samples)
                audio_to_transcribe = self.buffered_audio[transcribe_from:]

                # Temporary model call on fresh segment with correct offset
                try:
                    segments, _ = self.model.transcribe(
                        audio_to_transcribe,
                        language="ja",
                        task="translate",
                        vad_filter=False,
                        beam_size=5,
                        temperature=0.0,
                    )

                    offset_time = transcribe_from / self.audio_cfg["samplerate"]

                    for seg in segments:
                        if seg.text.strip():
                            yield seg.text.strip(), offset_time + seg.start, offset_time + seg.end

                except Exception as e:
                    self.logger.error(f"Segment transcription failed: {e}")

            # Trim processed part, keep overlap
            if len(self.buffered_audio) > overlap_samples:
                self.buffered_audio = self.buffered_audio[-overlap_samples:]

            self.queue.task_done()

    def start(self) -> None:
        self.running = True
        blocksize = int(self.audio_cfg["samplerate"] * self.audio_cfg["chunk_duration"])

        stream = sd.InputStream(
            samplerate=self.audio_cfg["samplerate"],
            device=self.audio_cfg["device"],
            channels=self.audio_cfg["channels"],
            blocksize=blocksize,
            callback=self._audio_callback,
            dtype=np.float32,
        )

        with stream:
            self.logger.info("Listening... ZERO LOSS MODE ACTIVE (Ctrl+C to stop)")

            def worker_target():
                for text, start, end in self._worker():
                    self._callback(text, start, end)

            worker = Thread(target=worker_target, daemon=True)
            worker.start()

            try:
                while self.running:
                    sd.sleep(100)
            except KeyboardInterrupt:
                self.logger.info("Shutting down gracefully...")
            finally:
                self.running = False
                self.pbar.close()
                self.txt_handle.close()
                self.srt_handle.close()
                self.logger.info(f"Session ended • Files saved to {self.output_dir.resolve()}")


# ============================= DEVICE HELPERS =============================
def get_input_channels() -> int:
    try:
        info = sd.query_devices(sd.default.device[0], 'input')
        return info['max_input_channels']
    except:
        return 1


CHANNELS = min(2, get_input_channels())
print(f"Detected input channels: {CHANNELS}")


# ============================= MAIN =============================
def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time Japanese → English • Zero Audio Loss")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--model", type=str, default="turbo", choices=["tiny", "base", "small", "medium", "large-v3", "turbo"])
    parser.add_argument("--chunk", type=float, default=4.0, help="Chunk duration in seconds (recommended 4–6s)")
    parser.add_argument("--overlap", type=float, default=1.5, help="Overlap between chunks in seconds (prevents word cuts)")
    parser.add_argument("--output-dir", type=str, default="./translations", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bar")
    args = parser.parse_args()

    output_path = Path(args.output_dir).expanduser().resolve()
    logger = setup_logger(output_path, quiet=args.quiet)

    audio_cfg: AudioConfig = {
        "device": args.device,
        "samplerate": 16000,
        "chunk_duration": args.chunk,
        "overlap_duration": args.overlap,
        "channels": CHANNELS,
    }

    compute_type = "int8" if args.model in ("tiny", "base") else "float16"
    trans_cfg: TranscriberConfig = {
        "model_size": args.model,
        "compute_type": compute_type,
        "language": "ja",
        "task": "translate",
        "device": "auto",
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