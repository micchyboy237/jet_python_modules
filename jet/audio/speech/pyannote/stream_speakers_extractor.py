#!/usr/bin/env python3
"""
stream_speakers_extractor.py
Live microphone → real-time speaker diarization with pyannote/speaker-diarization-3.1
→ saves clean speaker turns (wav + json)
"""

from __future__ import annotations

import os
import shutil
import json
import wave
import time
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# ====================== CONFIG ======================
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path(__file__).parent / "generated" / SCRIPT_NAME
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
SPEAKERS_DIR = OUTPUT_DIR / "speakers"
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32

MODEL_NAME = "pyannote/speaker-diarization-3.1"
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN environment variable")

WINDOW_DURATION = 10.0
STEP_DURATION = 5.0          # 50 % overlap
SPEAKER_CHANGE_THRESHOLD = 0.60
MIN_TURN_DURATION = 1.0
# ===================================================

console = Console()
logging = console.log


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str
    confidence: float = 1.0
    embedding: Optional[np.ndarray] = None


class LiveSpeakerExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging(f"Loading {MODEL_NAME} on {self.device}...")

        self.pipeline: Pipeline = Pipeline.from_pretrained(
            MODEL_NAME,
        )
        self.pipeline.to(self.device)

        self.buffer: List[np.ndarray] = []
        self.buffer_start_time: float = 0.0
        self.current_time: float = 0.0

        # persistent speaker identity tracking
        self.known_speakers: Dict[str, np.ndarray] = {}   # id → normalized avg embedding
        self.next_speaker_num: int = 0

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            console.print(f"[yellow]Audio status: {status}[/yellow]")
        chunk = indata.copy().flatten().astype(DTYPE)
        self.buffer.append(chunk)
        self.current_time += len(chunk) / SAMPLE_RATE

    def save_turn(self, turn: SpeakerTurn, audio_data: np.ndarray) -> Path:
        turn_dir = SPEAKERS_DIR / f"{self.next_speaker_num:03d}_{turn.start:.2f}_{turn.end:.2f}_{turn.speaker_id}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        wav_path = turn_dir / "sound.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        summary = {
            "turn_id": self.next_speaker_num,
            "speaker_id": turn.speaker_id,
            "start_sec": round(turn.start, 3),
            "end_sec": round(turn.end, 3),
            "duration_sec": round(turn.end - turn.start, 3),
            "global_start": turn.start,
            "utc_iso": datetime.utcnow().isoformat() + "Z",
            "confidence": round(turn.confidence, 4),
            "model": MODEL_NAME,
        }
        (turn_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        logging(f"[green]Saved[/green] [bold]{turn.speaker_id}[/bold] ({summary['duration_sec']}s)")
        self.next_speaker_num += 1
        return turn_dir

    def process_window(self, window_audio: np.ndarray, window_start: float) -> List[SpeakerTurn]:
        # write window to temporary wav (pipeline expects a file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f.name, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((window_audio * 32767).astype(np.int16).tobytes())
            temp_path = f.name

        try:
            # Full diarization – returns DiarizeOutput (pyannote 3.1)
            output = self.pipeline(temp_path)                     # type: ignore

            diarization: Annotation = output.speaker_diarization
            embeddings: np.ndarray = output.speaker_embeddings        # (num_speakers, dim)

            # map label → embedding (labels are SPEAKER_00, SPEAKER_01, …)
            label_to_emb = {
                label: emb / np.linalg.norm(emb)
                for label, emb in zip(diarization.labels(), embeddings)
            }

        finally:
            os.unlink(temp_path)

        turns: List[SpeakerTurn] = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            duration = turn.end - turn.start
            if duration < MIN_TURN_DURATION:
                continue

            # extract audio segment
            s_start = int(turn.start * SAMPLE_RATE)
            s_end   = int(turn.end   * SAMPLE_RATE)
            segment_audio = window_audio[s_start:s_end]

            # current speaker embedding (already normalized)
            emb = label_to_emb.get(speaker, np.zeros(512, dtype=np.float32))

            # match against known speakers
            best_sim = 0.0
            best_match: Optional[str] = None
            for known_id, known_emb in self.known_speakers.items():
                sim = float(np.dot(emb, known_emb))
                if best_sim < sim:
                    best_sim = sim
                    best_match = known_id

            if best_match and best_sim > SPEAKER_CHANGE_THRESHOLD:
                final_id = best_match
            else:
                final_id = f"SPEAKER_{len(self.known_speakers):02d}"
                self.known_speakers[final_id] = emb.copy()

            # rolling average update
            avg = self.known_speakers[final_id]
            self.known_speakers[final_id] = 0.7 * avg + 0.3 * emb

            turn_obj = SpeakerTurn(
                start=window_start + turn.start,
                end=window_start + turn.end,
                speaker_id=final_id,
                confidence=best_sim if best_match else 1.0,
                embedding=emb.copy(),
            )
            turns.append(turn_obj)
            self.save_turn(turn_obj, segment_audio)

        return turns

    def run(self) -> None:
        console.rule("[bold magenta]Live Speaker Diarization – pyannote/speaker-diarization-3.1[/bold magenta]")
        logging(f"Output → {SPEAKERS_DIR.resolve()}")

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=int(SAMPLE_RATE * 0.1),
                callback=self.audio_callback,
            ):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Listening…[/bold blue]"),
                    console=console,
                ) as progress:
                    progress.add_task("live", total=None)

                    while True:
                        time.sleep(0.1)

                        buffered_sec = sum(len(c) for c in self.buffer) / SAMPLE_RATE
                        if buffered_sec < WINDOW_DURATION:
                            continue

                        # take the last WINDOW_DURATION seconds
                        samples = int(WINDOW_DURATION * SAMPLE_RATE)
                        window_audio = np.concatenate(self.buffer)[-samples:]

                        self.process_window(window_audio, self.current_time - WINDOW_DURATION)

                        # slide window – keep overlap
                        keep_sec = WINDOW_DURATION - STEP_DURATION
                        keep_samples = int(keep_sec * SAMPLE_RATE)
                        self.buffer = [np.concatenate(self.buffer)[-keep_samples:]]
                        self.buffer_start_time = self.current_time - keep_sec

        except KeyboardInterrupt:
            console.print("\n[bold red]Stopped[/bold red]")
        finally:
            console.rule("[bold green]Session finished[/bold green]")
            logging(f"Total turns saved: {self.next_speaker_num}")
            logging(f"Data → [cyan]{SPEAKERS_DIR.resolve()}[/cyan]")


if __name__ == "__main__":
    extractor = LiveSpeakerExtractor()
    extractor.run()