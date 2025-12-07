#!/usr/bin/env python3
"""
stream_speakers_extractor.py

Live microphone → real-time speaker diarization → save each speaker turn
with audio clip and metadata.

Features:
- Streaming audio capture via sounddevice
- Rolling buffer + periodic diarization with context preservation
- Speaker change detection with embedding clustering
- Saves each turn: OUTPUT_DIR/speakers/<num>_<start>_<end>/
    ├── sound.wav
    └── summary.json
- Fully typed, modular, testable, DRY
"""

from __future__ import annotations

import os
import shutil
import json
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# ====================== CONFIG ======================
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR = Path(OUTPUT_DIR)
SPEAKERS_DIR = OUTPUT_DIR / "speakers"
SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.float32

# Diarization settings
MODEL_NAME = "pyannote/speaker-diarization-community-1"
HF_TOKEN = os.getenv("HF_TOKEN")  # Set in env: export HF_TOKEN=hf_...

# Processing window (seconds)
WINDOW_DURATION = 10.0
STEP_DURATION = 5.0  # Overlap for smooth speaker tracking

# Speaker change detection threshold (cosine similarity)
SPEAKER_CHANGE_THRESHOLD = 0.55

# ===================================================

console = Console()
logging = console.log
logging_handler = RichHandler(rich_tracebacks=True)
logging_handler.setLevel("INFO")

@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str
    confidence: float = 1.0
    embedding: Optional[np.ndarray] = None

class LiveSpeakerExtractor:
    def __init__(self, hf_token: str = HF_TOKEN):
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is required")

        self.pipeline: Pipeline = Pipeline.from_pretrained(MODEL_NAME)
        self.pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.buffer: List[np.ndarray] = []
        self.buffer_start_time: float = 0.0
        self.current_time: float = 0.0
        self.known_speakers: Dict[str, np.ndarray] = {}  # speaker_id → avg embedding
        self.next_speaker_num: int = 0
        self.turns: List[SpeakerTurn] = []

    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback — append incoming audio"""
        if status:
            console.print(f"[yellow]Audio callback status: {status}[/yellow]")
        audio_chunk = indata.copy().flatten().astype(DTYPE)
        self.buffer.append(audio_chunk)
        self.current_time += len(audio_chunk) / SAMPLE_RATE

    def save_turn(self, turn: SpeakerTurn, audio_data: np.ndarray) -> Path:
        """Save a single speaker turn to disk"""
        start_str = datetime.fromtimestamp(self.buffer_start_time + turn.start).strftime("%Y%m%d_%H%M%S")
        turn_dir = SPEAKERS_DIR / f"{self.next_speaker_num:03d}_{turn.start:.2f}_{turn.end:.2f}_{turn.speaker_id}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        # Save WAV
        wav_path = turn_dir / "sound.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

        # Save summary.json
        summary = {
            "turn_id": self.next_speaker_num,
            "speaker_id": turn.speaker_id,
            "start_sec": round(turn.start, 3),
            "end_sec": round(turn.end, 3),
            "duration_sec": round(turn.end - turn.start, 3),
            "global_start_timestamp": self.buffer_start_time + turn.start,
            "utc_timestamp": datetime.utcnow().isoformat(),
            "confidence": round(turn.confidence, 4),
            "source": "live_stream",
        }
        (turn_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        logging(f"[green]Saved[/green] {turn.speaker_id} → [bold]{turn_dir.name}[/bold]")
        self.next_speaker_num += 1
        return turn_dir

    def process_window(self, window_audio: np.ndarray, window_start: float) -> List[SpeakerTurn]:
        """Run diarization on a window and return detected turns with speaker matching"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            with wave.open(f.name, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes((window_audio * 32767).astype(np.int16).tobytes())
            temp_path = f.name

        try:
            # Correct: pipeline returns DiarizeOutput, not Annotation
            output: DiarizeOutput = self.pipeline(temp_path)

            # Use the correct diarization (with overlaps)
            diarization = output.speaker_diarization

        finally:
            os.unlink(temp_path)

        turns: List[SpeakerTurn] = []

        # CORRECT itertracks usage: turn, track, speaker
        for turn, track, speaker in diarization.itertracks(yield_label=True):
            start_sec = turn.start
            end_sec = turn.end
            duration = end_sec - start_sec

            # Extract segment audio
            start_sample = int(start_sec * SAMPLE_RATE)
            end_sample = int(end_sec * SAMPLE_RATE)
            segment_audio = window_audio[start_sample:end_sample]

            # Extract embedding using the embedding model (more reliable)
            try:
                emb = self.pipeline._embedding(
                    {"waveform": torch.from_numpy(window_audio[None, None, :]),
                     "file": {"uri": "temp"}},
                    segment=turn
                )
                emb = emb / np.linalg.norm(emb)
            except:
                # Fallback: use centroid if available
                label_idx = diarization.labels().index(speaker)
                if output.speaker_embeddings is not None and label_idx < len(output.speaker_embeddings):
                    emb = output.speaker_embeddings[label_idx]
                    emb = emb / np.linalg.norm(emb)
                else:
                    emb = np.zeros(512)  # fallback

            # Match to known speakers
            best_match = None
            best_sim = 0.0
            for known_id, known_emb in self.known_speakers.items():
                sim = float(np.dot(emb, known_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_match = known_id

            if best_match and best_sim > SPEAKER_CHANGE_THRESHOLD:
                final_speaker_id = best_match
            else:
                final_speaker_id = f"SPEAKER_{len(self.known_speakers):02d}"
                self.known_speakers[final_speaker_id] = emb

            # Update rolling average embedding
            if final_speaker_id in self.known_speakers:
                old = self.known_speakers[final_speaker_id]
                self.known_speakers[final_speaker_id] = 0.7 * old + 0.3 * emb

            turn_obj = SpeakerTurn(
                start=start_sec + window_start,
                end=end_sec + window_start,
                speaker_id=final_speaker_id,
                confidence=best_sim if best_match else 1.0,
                embedding=emb
            )
            turns.append(turn_obj)

            # Save only meaningful turns
            if duration > 1.0 and len(segment_audio) > SAMPLE_RATE * 0.5:
                self.save_turn(turn_obj, segment_audio)

        return turns

    def run(self) -> None:
        """Main streaming loop"""
        console.rule("[bold blue]Live Speaker Extractor Started[/bold blue]")
        logging(f"Output directory: {SPEAKERS_DIR.resolve()}")
        logging(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
                callback=self.audio_callback,
            ):
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Streaming & diarizing...", total=None)

                    while True:
                        time.sleep(0.1)

                        # Wait until we have enough audio
                        current_buffer_duration = sum(len(chunk) for chunk in self.buffer) / SAMPLE_RATE
                        if current_buffer_duration < WINDOW_DURATION:
                            continue

                        # Extract window
                        window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
                        window_audio = np.concatenate(self.buffer)[-window_samples:]
                        self.process_window(window_audio, self.current_time - WINDOW_DURATION)

                        # Slide window: keep overlap
                        keep_duration = WINDOW_DURATION - STEP_DURATION
                        keep_samples = int(keep_duration * SAMPLE_RATE)
                        self.buffer = [np.concatenate(self.buffer)[-keep_samples:]]
                        self.buffer_start_time = self.current_time - keep_duration

        except KeyboardInterrupt:
            console.print("\n[bold red]Stopped by user[/bold red]")
        finally:
            console.rule("[bold green]Session Summary[/bold green]")
            logging(f"Total speaker turns saved: {self.next_speaker_num}")
            logging(f"All files in: {SPEAKERS_DIR.resolve()}")


if __name__ == "__main__":
    extractor = LiveSpeakerExtractor()
    extractor.run()