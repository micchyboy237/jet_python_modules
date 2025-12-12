#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diarize a single recording using pyannote-audio 3.1+
Exports:
    <OUTPUT_DIR>/
    ├── summary.json
    └── segments/
        └── segment_0000/
            ├── segment.wav
            └── segment.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import torch
import torchaudio
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils import get_devices
from pyannote.core import Annotation

console = Console()


def diarize_file(
    audio_path: Path | str,
    output_dir: Path | str,
    *,
    pipeline_id: str = "pyannote/speaker-diarization-3.1",
    device: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Dict[str, Any]:
    """
    Run speaker diarization and save one wav + json per speaker turn.
    """
    audio_path = Path(audio_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"Loading pipeline [bold cyan]{pipeline_id}[/]")
    pipeline: Pipeline = Pipeline.from_pretrained(pipeline_id)

    if device is None:
        device = get_devices(needs=1)[0]
    console.log(f"Using device: [bold green]{device}[/]")
    pipeline.to(torch.device(device))

    console.log(f"Processing: [bold]{audio_path.name}[/] ({audio_path.stat().st_size / 1e6:.1f} MB)")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running diarization...", total=None)
        result = pipeline(
            str(audio_path),
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        progress.update(task, completed=True)

    # Handle both legacy (Annotation) and new (DiarizeOutput) return types
    if hasattr(result, "speaker_diarization"):
        diarization: Annotation = result.speaker_diarization  # type: ignore
    else:
        diarization = result  # direct Annotation in legacy mode

    console.log(f"Detected {len(diarization.labels())} speaker(s): {sorted(diarization.labels())}")

    # Load full audio once
    waveform, sample_rate = torchaudio.load(audio_path)
    total_seconds = waveform.shape[1] / sample_rate  # <-- fixed: proper duration calculation

    turns: list[Dict[str, Any]] = []

    for idx, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start_sec = round(segment.start, 3)
        end_sec = round(segment.end, 3)
        duration_sec = round(end_sec - start_sec, 3)

        seg_dir = segments_dir / f"segment_{idx:04d}"
        seg_dir.mkdir(exist_ok=True)

        # Crop waveform
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        wav_path = seg_dir / "segment.wav"
        torchaudio.save(str(wav_path), segment_waveform, sample_rate)

        meta = {
            "segment_index": idx,
            "speaker": speaker,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "wav_path": str(wav_path.relative_to(output_dir)),
        }

        json_path = seg_dir / "segment.json"
        json_path.write_text(json.dumps(meta, indent=2))

        turns.append(meta)

        console.log(
            f"[green]Exported[/] segment_{idx:04d} | {speaker.ljust(12)} | "
            f"{start_sec:>7.3f}s -> {end_sec:>7.3f}s ({duration_sec:>5.3f}s)"
        )

    # Summary
    summary = {
        "audio_file": str(audio_path),
        "total_duration_sec": round(total_seconds, 3),
        "sample_rate": sample_rate,
        "num_speakers": len(diarization.labels()),
        "speakers": sorted(diarization.labels()),
        "num_segments": len(turns),
        "segments": turns,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    console.log(f"[bold green]All done![/] Summary saved to [bold]{summary_path}[/]")

    return summary


if __name__ == "__main__":
    AUDIO_FILE = Path(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_20251212_041845.wav"
    )
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    diarize_file(
        audio_path=AUDIO_FILE,
        output_dir=OUTPUT_DIR,
        # use_auth_token="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXX",   # uncomment if you hit rate-limits
        # num_speakers=2,          # force exact number when known
    )