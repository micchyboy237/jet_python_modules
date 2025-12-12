#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diarize + export everything:
  • speaker_probabilities.png (full file)
  • frame_level_probabilities.json
  • per-segment plots + confidence stats
Works perfectly with pyannote/speaker-diarization-3.1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils import get_devices
from pyannote.core import Annotation, SlidingWindowFeature
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

console = Console()


def save_speaker_probability_plot(
    scores: SlidingWindowFeature,
    output_dir: Path,
    audio_duration: float,
    title: str = "Speaker Probabilities (Frame-Level)",
) -> Path:
    data = scores.data
    times = np.linspace(0, audio_duration, len(data), endpoint=False)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    cmap = plt.colormaps["tab10"]

    for spk_idx in range(data.shape[1]):
        ax.plot(times, data[:, spk_idx], label=f"Speaker {spk_idx}", color=cmap(spk_idx), linewidth=1.8)

    ax.set_xlim(0, audio_duration)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()

    plot_path = output_dir / "speaker_probabilities.png"
    fig.savefig(plot_path, bbox_inches="tight", facecolor="white")
    fig.savefig(plot_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    console.log("[bold magenta]Full plot saved[/] → speaker_probabilities.png")
    return plot_path


def save_segment_plot(
    scores: SlidingWindowFeature,
    seg_dir: Path,
    start_sec: float,
    end_sec: float,
    assigned_speaker: str,
) -> None:
    """Save per-segment probability plot – correctly handles 3D scores."""
    data_3d = np.asarray(scores.data)
    step = scores.sliding_window.step
    total_frames = data_3d.shape[0] * data_3d.shape[1]
    times = np.linspace(0, total_frames * step, total_frames, endpoint=False)

    mask = (times >= start_sec) & (times < end_sec)
    if not mask.any():
        return

    flat_probs = data_3d.reshape(-1, data_3d.shape[2])
    segment_probs = flat_probs[mask]
    segment_times = times[mask]

    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=150)
    cmap = plt.colormaps["tab10"]

    for spk_idx in range(segment_probs.shape[1]):
        label = f"Speaker {spk_idx}" + (" (assigned)" if str(spk_idx) == assigned_speaker else "")
        lw = 2.8 if str(spk_idx) == assigned_speaker else 1.2
        ax.plot(segment_times, segment_probs[:, spk_idx], label=label,
                color=cmap(spk_idx), linewidth=lw)

    ax.set_xlim(start_sec, end_sec)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Segment – {assigned_speaker}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    (seg_dir / "speaker_probabilities.png").write_bytes(fig.to_png())
    plt.close(fig)


def save_frame_level_probabilities_json(
    scores: SlidingWindowFeature,
    audio_duration: float,
    output_dir: Path,
) -> Path:
    """
    Save frame-level speaker probabilities to JSON.
    Handles the real pyannote 3.1 output format where scores.data is 3D:
        (num_chunks, frames_per_chunk, num_speakers)
    """
    # scores.data shape → (num_chunks, chunk_frames, num_speakers)
    data_3d = np.asarray(scores.data)                     # e.g. (N, 512, K)
    num_chunks, frames_per_chunk, num_speakers = data_3d.shape

    # Reconstruct correct time axis
    total_frames = num_chunks * frames_per_chunk
    times = np.linspace(0, audio_duration, total_frames, endpoint=False)

    # Flatten to 2D: (total_frames, num_speakers)
    flat_probs = data_3d.reshape(-1, num_speakers)

    frames = [
        {
            "time_sec": round(float(t), 4),
            "probabilities": {
                f"speaker_{i}": round(float(p), 6)
                for i, p in enumerate(frame_probs)
            }
        }
        for t, frame_probs in zip(times, flat_probs)
    ]

    json_path = output_dir / "frame_level_probabilities.json"
    json_path.write_text(json.dumps(frames, indent=2))
    console.log(
        f"[bold cyan]Frame-level JSON saved[/] → {json_path.name} "
        f"({len(frames)} frames, {num_speakers} speakers)"
    )
    return json_path


def get_segment_speaker_stats(
    scores: SlidingWindowFeature,
    start_sec: float,
    end_sec: float,
) -> Dict[str, float]:
    """Return max/avg probability per speaker inside a segment – handles 3D data."""
    data_3d = np.asarray(scores.data)                       # (chunks, frames_per_chunk, speakers)
    step = scores.sliding_window.step
    total_frames = data_3d.shape[0] * data_3d.shape[1]
    times = np.linspace(0, total_frames * step, total_frames, endpoint=False)

    # Find frame indices belonging to the segment
    mask = (times >= start_sec) & (times < end_sec)
    if not mask.any():
        # Segment too short or out of bounds → return zeros
        n = data_3d.shape[2]
        return {f"speaker_{i}_max_prob_max": 0.0 for i in range(n)} | \
               {f"speaker_{i}_prob_avg": 0.0 for i in range(n)}

    flat_probs = data_3d.reshape(-1, data_3d.shape[2])      # (total_frames, speakers)
    segment_probs = flat_probs[mask]

    max_probs = segment_probs.max(axis=0)
    avg_probs = segment_probs.mean(axis=0)

    return (
        {f"speaker_{i}_prob_max": round(float(v), 4) for i, v in enumerate(max_probs)}
        | {f"speaker_{i}_prob_avg": round(float(v), 4) for i, v in enumerate(avg_probs)}
    )


def diarize_file(
    audio_path: Path | str,
    output_dir: Path | str,
    *,
    pipeline_id: str = "pyannote/speaker-diarization-3.1",
    device: str | None = None,
    num_speakers: int | None = None,
    return_scores: bool = False,
) -> Tuple[Dict[str, Any], SlidingWindowFeature | None]:
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

    console.log(f"Running diarization on [bold]{audio_path.name}[/]")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Diarizing...", total=None)
        result = pipeline(str(audio_path), num_speakers=num_speakers)
        progress.update(task, completed=True)

    diarization: Annotation = (
        result.speaker_diarization if hasattr(result, "speaker_diarization") else result
    )
    console.log(f"Detected speakers: {sorted(diarization.labels())}")

    waveform, sample_rate = torchaudio.load(audio_path)
    total_seconds = waveform.shape[1] / sample_rate

    scores: SlidingWindowFeature | None = None
    if return_scores:
        console.log("Extracting raw frame-level speaker probabilities...")
        seg_inference = pipeline._segmentation
        scores = seg_inference(str(audio_path))

        np.save(output_dir / "segmentation_scores.npy", scores.data)

        timing = {
            "start": float(scores.sliding_window.start),
            "duration": float(scores.sliding_window.duration),
            "step": float(scores.sliding_window.step),
            "num_frames": int(scores.data.shape[0]),
            "num_speakers": int(scores.data.shape[1]),
            "frame_rate_hz": round(1 / scores.sliding_window.step, 2),
        }
        (output_dir / "segmentation_scores_timing.json").write_text(json.dumps(timing, indent=2))

        save_frame_level_probabilities_json(scores, total_seconds, output_dir)
        save_speaker_probability_plot(
            scores=scores,
            output_dir=output_dir,
            audio_duration=total_seconds,
            title=f"Speaker Diarization – {audio_path.name}",
        )

    # Export segments + per-segment plots & stats
    turns = []
    for idx, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start_sec = round(segment.start, 3)
        end_sec = round(segment.end, 3)
        duration_sec = round(end_sec - start_sec, 3)

        seg_dir = segments_dir / f"segment_{idx:04d}"
        seg_dir.mkdir(exist_ok=True)

        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        seg_wav = waveform[:, start_sample:end_sample]
        wav_path = seg_dir / "segment.wav"
        torchaudio.save(str(wav_path), seg_wav, sample_rate)

        seg_stats = (
            get_segment_speaker_stats(scores, start_sec, end_sec)
            if return_scores and scores is not None
            else {}
        )

        meta = {
            "segment_index": idx,
            "speaker": str(speaker),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "wav_path": str(wav_path.relative_to(output_dir)),
            **seg_stats,
        }
        (seg_dir / "segment.json").write_text(json.dumps(meta, indent=2))

        # Save per-segment plot (only when scores are available)
        if return_scores and scores is not None:
            save_segment_plot(
                scores=scores,
                seg_dir=seg_dir,
                start_sec=start_sec,
                end_sec=end_sec,
                assigned_speaker=speaker,  # e.g. "SPEAKER_00"
            )

        turns.append(meta)

        console.log(
            f"[green]Saved[/] segment_{idx:04d} | {str(speaker):<12} | "
            f"{start_sec:>7.3f}s → {end_sec:>7.3f}s"
        )

    summary = {
        "audio_file": str(audio_path),
        "total_duration_sec": round(total_seconds, 3),
        "sample_rate": sample_rate,
        "num_speakers": len(diarization.labels()),
        "speakers": sorted(str(s) for s in diarization.labels()),
        "num_segments": len(turns),
        "segments": turns,
        "segmentation_scores_npy": "segmentation_scores.npy" if return_scores else None,
        "frame_level_probabilities_json": "frame_level_probabilities.json" if return_scores else None,
        "speaker_probability_plot": "speaker_probabilities.png" if return_scores else None,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    console.log(f"[bold green]Finished![/] → {summary_path}")

    return summary, scores


if __name__ == "__main__":
    import shutil

    AUDIO_FILE = Path(
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_20251212_041845.wav"
    )
    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    diarize_file(
        audio_path=AUDIO_FILE,
        output_dir=OUTPUT_DIR,
        return_scores=True,
    )