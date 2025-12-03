# client/file_transcriber.py
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import AsyncGenerator

import httpx
import numpy as np
import soundfile as sf
import torch
from rich.logging import RichHandler
from rich.progress import Progress, ProgressColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from collections import deque
from typing import Deque
import typing


# ─────────────────────────────────────────────────────────────────────────────
# Silero VAD setup
# ─────────────────────────────────────────────────────────────────────────────
# Set hub cache dir to project-local
torch.hub.set_dir(str(Path(__file__).parent / ".cache" / "torch_hub"))

# Load Silero VAD model and utilities (once)
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True,
    verbose=False,
)
get_speech_timestamps, _, _, _, _ = utils

OUTPUT_DIR = (Path(__file__).parent / "generated" / Path(__file__).stem)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class MofNCompleteColumn(ProgressColumn):
    """Renders completed count/total of tasks."""
    def render(self, task) -> Text:
        completed = int(task.completed) if task.completed is not None else 0
        total = int(task.total) if task.total is not None else 0
        return Text(
            f"{completed}/{total}",
            style="progress.download"
        )

class SpeedColumn(ProgressColumn):
    """Render speed safely, fallback to '?.??' if not available yet"""
    def render(self, task) -> Text:
        if task.speed is None:
            return Text("?.?? seg/s", style="progress.data.speed")
        return Text(f"{task.speed:.2f} seg/s", style="progress.data.speed")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
API_URL = "http://shawn-pc.local:8001/transcribe_chunk?task=translate"
TARGET_SR = 16_000

# Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%X",
    handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_path=False)],
)
log = logging.getLogger("file-client")

async def transcribe_chunk(audio_chunk: np.ndarray) -> dict:
    """Send raw float32 mono 16kHz chunk to server"""
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            API_URL,
            content=audio_chunk.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
        return response.json()

def load_and_resample(path: Path | str) -> np.ndarray:
    """Load any audio file → float32, 16kHz, mono"""
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # stereo → mono
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)

class ChunkResult(typing.NamedTuple):
    start_sec: float
    end_sec: float
    text: str
    language: str
    language_prob: float
    segments: list[dict]
    words: list[dict]

async def transcribe_file(
    file_path: Path | str,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 700,
    speech_pad_ms: int = 30,
    threshold: float = 0.6,
    output_dir: Path | None = None,
) -> AsyncGenerator[ChunkResult, None]:
    """
    Use Silero VAD to extract speech segments,
    transcribe each with the /transcribe_chunk endpoint,
    and yield as ChunkResult (with rich progress, logging, and save per chunk).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[bold blue]Loading audio:[/] {file_path.name}")
    audio = load_and_resample(file_path)
    audio_tensor = torch.from_numpy(audio)

    log.info("[bold blue]Running Silero VAD...[/]")
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        sampling_rate=TARGET_SR,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )

    total_chunks = len(speech_timestamps)
    duration_total = len(audio) / TARGET_SR
    if total_chunks == 0:
        log.warning("[yellow]No speech detected in file[/]")
        return

    log.info(f"[bold]Found[/] {total_chunks} speech segment(s) in {duration_total:.2f}s audio")

    results_buffer: Deque[ChunkResult] = deque()

    with Progress(
        TextColumn("[bold blue]Transcribing speech segments:[/] {task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "<",
        SpeedColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task("Transcribing...", total=total_chunks)

        for idx, seg in enumerate(speech_timestamps):
            start_sec = seg["start"]
            end_sec = seg["end"]
            start_sample = int(round(start_sec * TARGET_SR))
            end_sample = int(round(end_sec * TARGET_SR))

            chunk = audio[start_sample:end_sample]
            # Minimal padding for model: pad to next multiple of 512
            if len(chunk) % 512 != 0:
                chunk = np.pad(chunk, (0, 512 - (len(chunk) % 512)))

            result = await transcribe_chunk(chunk)

            text = result.get("text", "").strip()
            lang = result.get("language", "unknown")
            prob = result.get("language_probability", 0.0)
            segments = result.get("segments", [])
            words = result.get("words", [])

            chunk_result = ChunkResult(
                start_sec=start_sec,
                end_sec=end_sec,
                text=text,
                language=lang,
                language_prob=prob,
                segments=segments,
                words=words,
            )
            results_buffer.append(chunk_result)

            if output_dir:
                # Create a dedicated folder for this speech chunk
                chunk_dir = output_dir / f"speech_{idx:04d}_{start_sec:.2f}s-{end_sec:.2f}s"
                chunk_dir.mkdir(parents=True, exist_ok=True)

                # Save the actual speech audio
                wav_path = chunk_dir / "chunk.wav"
                sf.write(wav_path, chunk, samplerate=TARGET_SR, subtype="PCM_16")

                # Save transcription result as JSON in the same folder
                import json
                json_path = chunk_dir / "result.json"
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(chunk_result._asdict(), f, ensure_ascii=False, indent=2)

                log.debug(f"Saved chunk → {chunk_dir.relative_to(output_dir.parent)}")

            time_range = f"{start_sec:6.2f}─{end_sec:6.2f}s"
            duration = end_sec - start_sec
            if text:
                log.info(
                    f"[cyan]{time_range}[/] "
                    f"[bold magenta]{lang}[/] ({prob:.3f}) "
                    f"[dim]({duration:.2f}s)[/] → [green]\"{text}\"[/]"
                )
            else:
                log.debug(f"[dim]{time_range} → (no transcription)[/]")

            progress.update(task, advance=1)
            yield chunk_result

    spoken_chunks = [r for r in results_buffer if r.text]
    log.info(
        f"[bold green]Transcription complete![/] "
        f"{len(spoken_chunks)} speech segments transcribed "
        f"(from {total_chunks} VAD-detected regions)"
    )
    transcribe_file.results_buffer = results_buffer  # type: ignore

# --------------------------------------------------------------------------- #
# Simple CLI + real-world example -- with default file provided
# --------------------------------------------------------------------------- #
DEFAULT_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"

# ─────────────────────────────────────────────────────────────────────────────
# Updated main() – now uses the saved results + better final output, VAD args
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test /transcribe_chunk with VAD-detected speech in any audio file")
    parser.add_argument("file", nargs="?", type=Path, default=DEFAULT_FILE, help="Path to audio file")
    parser.add_argument("--min-speech-ms", type=int, default=500, help="Minimum speech duration (ms)")
    parser.add_argument("--min-silence-ms", type=int, default=300, help="Minimum silence duration (ms)")
    parser.add_argument("--speech-pad-ms", type=int, default=30, help="Pad speech segments (ms)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Silero VAD probability threshold")
    args = parser.parse_args()

    full_text_parts: list[str] = []
    async for chunk_result in transcribe_file(
        args.file,
        min_speech_duration_ms=args.min_speech_ms,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
        threshold=args.threshold,
        output_dir=OUTPUT_DIR
    ):
        if chunk_result.text:
            full_text_parts.append(chunk_result.text)

    print("\n" + "═" * 70)
    print("[bold green]FINAL TRANSCRIPTION[/]")
    print("═" * 70)
    print(" ".join(full_text_parts))
    print("═" * 70)

    # Bonus: show saved results are available
    if hasattr(transcribe_file, "results_buffer"):
        log.info(f"[bold]All {len(transcribe_file.results_buffer)} speech segment results saved in `transcribe_file.results_buffer`[/]")

if __name__ == "__main__":
    asyncio.run(main())