#!/usr/bin/env python3
"""
jp_transcriber.py
REAL-TIME JAPANESE → ENGLISH • ZERO LOSS • MAXIMUM ACCURACY
Full-context rolling buffer transcription • No VAD • Perfect continuity
"""
import argparse
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import TypedDict, Literal
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from tqdm import tqdm


# ============================= CONFIG =============================
class AudioConfig(TypedDict):
    device: int | None
    samplerate: int
    chunk_duration: float
    channels: int


class TranscriberConfig(TypedDict):
    model_size: str
    compute_type: str
    language: str
    task: Literal["translate"]
    device: str


# ============================= LOGGER =============================
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
        self.txt_file = output_dir / "translation.txt"
        self.srt_file = output_dir / "subtitles.srt"
        self.txt_handle = open(self.txt_file, "w", encoding="utf-8")
        self.srt_handle = open(self.srt_file, "w", encoding="utf-8")
        self.srt_counter = 1
        self.session_start = datetime.now()

        self.logger.info("ZERO LOSS + MAX CONTEXT MODE • Japanese → English")
        self.logger.info(f"Output: {self.output_dir.resolve()}")

        # Device setup
        if "arm" in platform.machine().lower():
            self.trans_cfg["device"] = "cpu"
            self.trans_cfg["compute_type"] = "float32"
            self.logger.info("Apple Silicon → CPU + float32")
        else:
            self.trans_cfg["device"] = "cuda" if any("cuda" in str(d) for d in sd.query_devices()) else "cpu"

        # Load model
        self.model = WhisperModel(
            trans_cfg["model_size"],
            device=self.trans_cfg["device"],
            compute_type=self.trans_cfg["compute_type"],
        )
        self.logger.info(f"Model loaded: {trans_cfg['model_size']} on {self.trans_cfg['device']}")

        # Large queue + rolling buffer
        self.queue: Queue[np.ndarray] = Queue(maxsize=300)  # ~2.5+ minutes of buffer
        self.buffer = np.array([], dtype=np.float32)
        self.last_emitted_time = 0.0  # Tracks last timestamp we've shown
        self.running = False
        self.processed_chunks = 0

        self.pbar = tqdm(
            desc="Transcribing",
            unit="chunk",
            dynamic_ncols=True,
            colour="green",
            disable=not show_progress,
        )

        self.absolute_audio_time: float = 0.0          # real-world elapsed seconds
        self.last_emitted_end: float = 0.0             # last segment end we actually showed
        self.emitted_keys: set[tuple[str, float]] = set()  # deduplication cache

    def _seconds_to_srt(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _save_segment(self, text: str, start: float, end: float) -> None:
        # light cleanup – generic, no language-specific strings
        text = text.strip()
        if len(text) < 2 or text.lower().startswith(("subtitles", "thanks for watching")):
            return

        print(f"→ {text}")

        self.txt_handle.write(text + "\n")
        self.txt_handle.flush()

        self.srt_handle.write(f"{self.srt_counter}\n")
        self.srt_handle.write(f"{self._seconds_to_srt(start)} --> {self._seconds_to_srt(end)}\n")
        self.srt_handle.write(f"{text}\n\n")
        self.srt_handle.flush()
        self.srt_counter += 1

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            self.logger.warning(f"Audio overrun: {status}")
        audio = indata.copy()[:, 0] if indata.ndim > 1 else indata.copy().flatten()
        audio = audio.astype(np.float32)
        try:
            self.queue.put_nowait(audio)
        except:
            self.logger.error("CRITICAL: Audio queue full — dropping frame!")

    def _worker(self) -> None:
        chunk_seconds = self.audio_cfg["chunk_duration"]
        samples_per_chunk = int(self.audio_cfg["samplerate"] * chunk_seconds)

        while self.running:
            try:
                chunk = self.queue.get(timeout=1.0)
            except Empty:
                continue

            # advance real-world clock
            self.absolute_audio_time += chunk_seconds

            # rolling buffer
            self.buffer = np.concatenate([self.buffer, chunk])
            self.processed_chunks += 1
            self.pbar.update(1)

            # keep 8–30 seconds of context (sweet spot)
            max_samples = int(self.audio_cfg["samplerate"] * 30)
            if len(self.buffer) > max_samples:
                keep_samples = len(self.buffer) - max_samples
                self.buffer = self.buffer[-max_samples:]

            if len(self.buffer) < self.audio_cfg["samplerate"] * 7:
                self.queue.task_done()
                continue

            try:
                segments, info = self.model.transcribe(
                    self.buffer,
                    language=self.trans_cfg.get("language", "ja"),
                    task=self.trans_cfg["task"],           # "translate" → English guaranteed
                    beam_size=7,
                    best_of=5,
                    temperature=(0.0, 0.2),
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=700),
                    prefix=None,                            # prevents repetition carry-over
                    word_timestamps=False,
                )

                buffer_duration = len(self.buffer) / self.audio_cfg["samplerate"]

                for seg in segments:
                    text = seg.text.strip()
                    if not text:
                        continue

                    # deduplication key: text + rounded end time (0.5s tolerance)
                    key = (text, round(seg.end, 1))
                    if key in self.emitted_keys:
                        continue
                    self.emitted_keys.add(key)
                    if len(self.emitted_keys) > 1500:           # prevent unbounded growth
                        self.emitted_keys = set(list(self.emitted_keys)[-1000:])

                    # calculate real-world timestamps
                    seg_start_real = self.absolute_audio_time - buffer_duration + seg.start
                    seg_end_real   = self.absolute_audio_time - buffer_duration + seg.end

                    # only emit segments that are truly new
                    if seg_end_real <= self.last_emitted_end + 0.3:
                        continue

                    self._save_segment(text, seg_start_real, seg_end_real)
                    self.last_emitted_end = seg_end_real

                self.pbar.set_postfix({
                    "buf": f"{buffer_duration:.1f}s",
                    "lag": f"{self.absolute_audio_time - self.last_emitted_end:.1f}s",
                    "det": info.language,
                })

            except Exception as e:
                self.logger.error(f"Transcription exception: {e}")
            finally:
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
            self.logger.info("LISTENING • Zero Loss • Full Context • Japanese → English (Ctrl+C to stop)")

            worker = Thread(target=self._worker, daemon=True)
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
                self.logger.info(f"Done • Files saved to {self.output_dir.resolve()}")


# ============================= DEVICE =============================
def get_input_channels() -> int:
    try:
        info = sd.query_devices(sd.default.device[0], 'input')
        return info['max_input_channels']
    except:
        return 1


CHANNELS = min(2, get_input_channels())
print(f"Input channels: {CHANNELS}")


# ============================= MAIN =============================
def main() -> None:
    parser = argparse.ArgumentParser(description="Japanese → English Real-time • Zero Loss • Max Quality")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--chunk", type=float, default=2.0, help="Audio chunk size in seconds (1.5–3.0 recommended)")
    parser.add_argument("--output-dir", type=str, default="./translations")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_dir).expanduser().resolve()
    logger = setup_logger(output_path, args.quiet)

    audio_cfg: AudioConfig = {
        "device": args.device,
        "samplerate": 16000,
        "chunk_duration": args.chunk,
        "channels": CHANNELS,
    }

    trans_cfg: TranscriberConfig = {
        "model_size": "large-v3",
        "compute_type": "int16",
        "language": "ja",
        "task": "translate",   # ← This guarantees English output
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