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
from rich.logging import RichHandler
from rich.progress import Progress, ProgressColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

# ─────────────────────────────────────────────────────────────────────────────
# Updated imports (add these two)
# ─────────────────────────────────────────────────────────────────────────────
from collections import deque
from typing import Deque
import typing

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
            return Text("?.?? chunks/s", style="progress.data.speed")
        return Text(f"{task.speed:.2f} chunks/s", style="progress.data.speed")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
API_URL = "http://shawn-pc.local:8001/transcribe_chunk"
TARGET_SR = 16_000
CHUNK_DURATION_SEC = 2.0           # 1–4s works best with VAD filtering on server
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_DURATION_SEC)

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
    async with httpx.AsyncClient(timeout=30.0) as client:
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


# ─────────────────────────────────────────────────────────────────────────────
# New: Structured result storage
# ─────────────────────────────────────────────────────────────────────────────
class ChunkResult(typing.NamedTuple):
    start_sec: float
    end_sec: float
    text: str
    language: str
    language_prob: float
    segments: list[dict]


# ─────────────────────────────────────────────────────────────────────────────
# Replace entire `transcribe_file` function with this improved version
# ─────────────────────────────────────────────────────────────────────────────
async def transcribe_file(
    file_path: Path | str,
    chunk_duration_sec: float = CHUNK_DURATION_SEC,
    output_dir: Path | None = None,   # ← NEW
) -> AsyncGenerator[ChunkResult, None]:
    """
    Enhanced version with:
    - Richer progress bar (adds percentage + rate)
    - Per-chunk detailed logging with timestamps
    - Saves every non-empty result in a deque (accessible later)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)  # ← NEW

    log.info(f"[bold blue]Loading audio:[/] {file_path.name}")
    audio = load_and_resample(file_path)
    duration_total = len(audio) / TARGET_SR

    chunk_samples = int(TARGET_SR * chunk_duration_sec)
    total_chunks = int(np.ceil(len(audio) / chunk_samples))

    log.info(f"[bold]Duration:[/] {duration_total:.2f}s → {total_chunks} chunks of {chunk_duration_sec}s")

    # Store results for later reuse (e.g. export to JSON/SRT)
    results_buffer: Deque[ChunkResult] = deque()

    with Progress(
        TextColumn("[bold blue]Transcribing:[/] {task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        "•",
        TimeElapsedColumn(),
        "<",
        SpeedColumn(),  # ← Replaced the unsafe string with safe column
        transient=False,
    ) as progress:
        task = progress.add_task("Processing chunks...", total=total_chunks)

        for idx in range(total_chunks):
            start_sample = idx * chunk_samples
            end_sample = min(start_sample + chunk_samples, len(audio))
            chunk = audio[start_sample:end_sample]

            # Zero-pad if needed (keeps model happy)
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

            result = await transcribe_chunk(chunk)

            start_sec = start_sample / TARGET_SR
            end_sec = end_sample / TARGET_SR
            text = result["text"].strip()
            lang = result["language"]
            prob = result["language_probability"]

            chunk_result = ChunkResult(
                start_sec=start_sec,
                end_sec=end_sec,
                text=text,
                language=lang,
                language_prob=prob,
                segments=result.get("segments", []),
            )
            results_buffer.append(chunk_result)

            # Save chunk output to disk (if enabled)  ← NEW
            if output_dir:
                import json
                out_path = output_dir / f"chunk_{idx:04d}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(chunk_result._asdict(), f, ensure_ascii=False, indent=2)

            # Rich per-chunk logging
            time_range = f"{start_sec:6.2f}─{end_sec:6.2f}s"
            if text:
                log.info(
                    f"[cyan]{time_range}[/] "
                    f"[bold magenta]{lang}[/] ({prob:.3f}) → [green]\"{text}\"[/]"
                )
            else:
                log.debug(f"[dim]{time_range} → (silence/no speech)[/]")

            progress.update(task, advance=1)

            yield chunk_result

    # Final summary
    spoken_chunks = [r for r in results_buffer if r.text]
    log.info(
        f"[bold green]Done![/] {len(spoken_chunks)} speech chunks out of {total_chunks} "
        f"({len(spoken_chunks)/total_chunks:.1%})"
    )

    # Make buffer accessible outside the function if needed
    transcribe_file.results_buffer = results_buffer  # type: ignore


# --------------------------------------------------------------------------- #
# Simple CLI + real-world example -- with default file provided
# --------------------------------------------------------------------------- #
DEFAULT_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"

# ─────────────────────────────────────────────────────────────────────────────
# Updated main() – now uses the saved results + better final output
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test /transcribe_chunk with any audio file")
    parser.add_argument("file", nargs="?", type=Path, default=DEFAULT_FILE,
                        help="Path to audio file")
    parser.add_argument("--chunk-sec", type=float, default=2.0,
                        help="Chunk size in seconds (default: 2.0)")
    args = parser.parse_args()

    full_text_parts: list[str] = []
    async for chunk_result in transcribe_file(args.file, chunk_duration_sec=args.chunk_sec, output_dir=OUTPUT_DIR):
        if chunk_result.text:
            full_text_parts.append(chunk_result.text)

    print("\n" + "═" * 70)
    print("[bold green]FINAL TRANSCRIPTION[/]")
    print("═" * 70)
    print(" ".join(full_text_parts))
    print("═" * 70)

    # Bonus: show saved results are available
    if hasattr(transcribe_file, "results_buffer"):
        log.info(f"[bold]All {len(transcribe_file.results_buffer)} chunk results saved in `transcribe_file.results_buffer`[/]")

if __name__ == "__main__":
    asyncio.run(main())