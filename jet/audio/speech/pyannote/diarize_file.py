"""
Diarize a recording + optionally export raw segmentation scores (logits/probabilities)
Works with pyannote-audio 3.1+ (including current 4.0.3 version)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torchaudio
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines.utils import get_devices
from pyannote.core import Annotation, SlidingWindowFeature

console = Console()

def compute_scores_insights(scores: SlidingWindowFeature) -> Dict[str, Any]:
    """
    Analyze the raw segmentation scores and return useful statistics.
    Works with pyannote 3.1 where scores.data is 3D: (chunks, frames_per_chunk, speakers)
    """
    data = np.asarray(scores.data)  # shape: (chunks, frames_per_chunk, speakers)

    # Handle both 2D and 3D cases safely
    if data.ndim == 3:
        num_chunks, frames_per_chunk, num_speakers = data.shape
        total_frames = num_chunks * frames_per_chunk
        flat_data = data.reshape(total_frames, num_speakers)
    elif data.ndim == 2:
        total_frames, num_speakers = data.shape
        flat_data = data
    else:
        raise ValueError(f"Unexpected scores.data shape: {data.shape}")

    step = scores.sliding_window.step
    duration = total_frames * step

    # Per-speaker stats
    max_per_speaker = flat_data.max(axis=0)
    avg_per_speaker = flat_data.mean(axis=0)
    active_ratio = (flat_data > 0.5).mean(axis=0)  # how often model thinks speaker is active

    # Global stats
    max_prob_per_frame = flat_data.max(axis=1)
    confidence_mean = max_prob_per_frame.mean()
    confidence_std = max_prob_per_frame.std()
    overlap_frames = (flat_data.sum(axis=1) > 1.1).sum()  # rough overlap detection

    return {
        "total_duration_sec": round(float(duration), 3),
        "total_frames": int(total_frames),
        "frame_step_sec": float(step),
        "num_speakers": int(num_speakers),
        "confidence_mean": round(float(confidence_mean), 4),
        "confidence_std": round(float(confidence_std), 4),
        "overlap_frame_count": int(overlap_frames),
        "overlap_percentage": round(100 * overlap_frames / total_frames, 2),
        "per_speaker": {
            f"speaker_{i}": {
                "max_probability": round(float(m), 4),
                "avg_probability": round(float(a), 4),
                "active_ratio": round(float(r), 4),
            }
            for i, (m, a, r) in enumerate(zip(max_per_speaker, avg_per_speaker, active_ratio))
        },
    }

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
    pipeline: Pipeline = Pipeline.from_pretrained(pipeline_id, revision="main")
    if device is None:
        device = get_devices(needs=1)[0]
    console.log(f"Using device: [bold green]{device}[/]")
    pipeline.to(torch.device(device))
    console.log(f"Running full diarization on [bold]{audio_path.name}[/]")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Diarizing...", total=None)
        diarization_result = pipeline(
            str(audio_path),
            num_speakers=num_speakers,
        )
        progress.update(task, completed=True)
    diarization: Annotation = (
        diarization_result.speaker_diarization
        if hasattr(diarization_result, "speaker_diarization")
        else diarization_result
    )
    console.log(f"Detected {len(diarization.labels())} speaker(s): {sorted(str(s) for s in diarization.labels())}")
    waveform, sample_rate = torchaudio.load(audio_path)
    total_seconds = waveform.shape[1] / sample_rate
    turns: list[Dict[str, Any]] = []
    for idx, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start_sec = round(segment.start, 3)
        end_sec = round(segment.end, 3)
        duration_sec = round(end_sec - start_sec, 3)
        seg_dir = segments_dir / f"segment_{idx:04d}"
        seg_dir.mkdir(exist_ok=True)
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]
        wav_path = seg_dir / "segment.wav"
        torchaudio.save(str(wav_path), segment_waveform, sample_rate)
        meta = {
            "segment_index": idx,
            "speaker": str(speaker),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "wav_path": str(wav_path.relative_to(output_dir)),
        }
        (seg_dir / "segment.json").write_text(json.dumps(meta, indent=2))
        turns.append(meta)
        console.log(
            f"[green]Saved[/] segment_{idx:04d} | {str(speaker):<12} | "
            f"{start_sec:>7.3f}s -> {end_sec:>7.3f}s ({duration_sec:>5.3f}s)"
        )
    scores: SlidingWindowFeature | None = None
    if return_scores:
        console.log("Extracting raw frame-level segmentation scores...")
        # In pyannote-audio 4.0+, instantiate pipeline and access internal model
        pipeline.instantiate({})
        try:
            # Access the bundled segmentation model via private attribute
            seg_model: Model = pipeline._segmentation.model
            console.log("[dim]Using bundled segmentation model[/]")
        except AttributeError:
            console.log("[red]Warning: Could not access internal model. Falling back to direct load.[/]")
            # Fallback for custom pipelines: load segmentation directly
            seg_model = Model.from_pretrained("pyannote/segmentation-3.0", revision="main")
            seg_model.to(torch.device(device))
        from pyannote.audio import Inference
        duration = seg_model.specifications.duration
        inference = Inference(
            seg_model,
            duration=duration,
            step=0.1 * duration,  # 90% overlap (default for smooth sliding)
        )
        scores = inference(str(audio_path))
        scores_path = output_dir / "segmentation_scores.npy"
        np.save(scores_path, scores.data)
        console.log(f"[bold blue]Raw scores saved[/] → {scores_path}")
        # Corrected timing block
        if scores.data.ndim == 3:
            num_chunks = scores.data.shape[0]
            frames_per_chunk = scores.data.shape[1]
            num_speakers = scores.data.shape[2]
            total_frames = num_chunks * frames_per_chunk
        else:
            total_frames, num_speakers = scores.data.shape
            frames_per_chunk = None
            num_chunks = None
        timing = {
            "start": float(scores.sliding_window.start),
            "duration": float(scores.sliding_window.duration),
            "step": float(scores.sliding_window.step),
            "total_frames": int(total_frames),
            "frames_per_chunk": frames_per_chunk,
            "num_chunks": int(num_chunks) if scores.data.ndim == 3 else None,
            "num_speakers": int(num_speakers),
            "frame_rate_hz": round(1 / scores.sliding_window.step, 3),
        }
        (output_dir / "segmentation_scores_timing.json").write_text(json.dumps(timing, indent=2))
        # New: Compute and save insights
        console.log("Analyzing raw speaker probabilities...")
        insights = compute_scores_insights(scores)
        insights_path = output_dir / "segmentation_scores_insights.json"
        (insights_path).write_text(json.dumps(insights, indent=2))
        console.log("[bold yellow]Scores insights saved[/] → scores_insights.json")
    else:
        scores_path = None
    summary = {
        "audio_file": str(audio_path),
        "total_duration_sec": round(total_seconds, 3),
        "sample_rate": sample_rate,
        "num_speakers": len(diarization.labels()),
        "speakers": sorted(str(s) for s in diarization.labels()),
        "num_segments": len(turns),
        "segments": turns,
        "segmentation_scores_npy": str(scores_path) if scores_path else None,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    console.log(f"[bold green]Done![/] Summary → {summary_path}")

    # === NEW: Export RTTM (standard format) ===
    rttm_path = output_dir / "diarization.rttm"
    with rttm_path.open("w") as f:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            line = (
                f"SPEAKER {audio_path.stem} 1 "
                f"{turn.start:.3f} {turn.duration:.3f} "
                f"<NA> <NA> {speaker} <NA> <NA>\n"
            )
            f.write(line)
    console.log(f"[bold magenta]RTTM exported[/] → {rttm_path.name}")

    # === NEW: Per-segment confidence from raw scores ===
    if scores is not None:
        # Build a mapping from time → max speaker probability
        frame_times = np.arange(
            scores.sliding_window.start,
            scores.sliding_window.start + scores.data.shape[0] * scores.sliding_window.step,
            scores.sliding_window.step,
        )
        # Flatten 3D → 2D if needed
        if scores.data.ndim == 3:
            flat_scores = scores.data.reshape(-1, scores.data.shape[-1])
        else:
            flat_scores = scores.data
        confidences = flat_scores.max(axis=1)  # highest speaker prob per frame

        # Attach confidence to each segment
        for seg in turns:
            mask = (frame_times >= seg["start_sec"]) & (frame_times < seg["end_sec"])
            if mask.any():
                seg_conf = float(confidences[mask].mean())
            else:
                seg_conf = 0.0
            seg["confidence"] = round(seg_conf, 4)

        # Update summary with confidences
        summary["segments"] = turns

    # === NEW: Beautiful HTML timeline visualization ===
    html_path = output_dir / "timeline.html"

    # Modern, accessible, high-contrast colors
    colors = ["#8b5cf6", "#ec4899", "#10b981", "#f59e0b", "#3b82f6", "#ef4444"]

    rows = ""
    for i, seg in enumerate(turns):
        speaker_color = colors[ord(seg["speaker"][-1]) % len(colors)]
        conf = seg.get("confidence", 0.0)
        opacity = max(0.4, conf)
        rows += f'''
        <div class="segment"
             style="left: {seg["start_sec"]/total_seconds*100:.3f}%;
                    width: {seg["duration_sec"]/total_seconds*100:.3f}%;
                    background: {speaker_color}{int(opacity*255):02x};"
             title="Speaker: {seg['speaker']} | {seg['start_sec']:.3f}–{seg['end_sec']:.3f}s | Duration: {seg['duration_sec']:.3f}s | Conf: {conf:.4f}">
          <span class="speaker">{seg['speaker']}</span>
          <span class="time">{seg['start_sec']:.2f}s</span>
          <span class="conf">{conf:.3f}</span>
        </div>'''

    # Time ruler every 2 seconds
    ticks = []
    for t in np.arange(0, total_seconds + 0.001, 2):
        percent = t / total_seconds * 100 if total_seconds > 0 else 0
        ticks.append(f'<div class="tick" style="left:{percent:.3f}%"></div>')
        ticks.append(f'<div class="tick-label" style="left:{percent:.3f}%">{t:.1f}s</div>')
    ticks_html = "\n        ".join(ticks)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Diarization • {audio_path.name}</title>
  <style>
    :root {{ --bg: #0d1117; --fg: #c9d1d9; --border: #30363d; }}
    body {{ 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg); color: var(--fg); margin: 0; padding: 2rem;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ margin: 0 0 1.5rem; font-weight: 600; }}
    .info {{ opacity: 0.8; margin-bottom: 1rem; font-size: 0.95rem; }}
    .timeline {{ 
      position: relative; height: 80px; background: #161b22; border-radius: 12px; 
      overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .ruler {{ 
      position: absolute; top: 0; left: 0; right: 0; height: 20px;
      border-bottom: 1px solid var(--border);
    }}
    .tick {{ position: absolute; top: 20px; width: 1px; height: 8px; background: var(--border); }}
    .tick-label {{ position: absolute; top: 24px; font-size: 0.75rem; transform: translateX(-50%); }}
    .track {{ position: relative; height: 60px; margin-top: 20px; }}
    .segment {{
      position: absolute; height: 44px; top: 8px; border-radius: 8px;
      padding: 0 10px; display: flex; align-items: center; gap: 8px;
      color: white; font-weight: 600; font-size: 0.9rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
      transition: all 0.2s; white-space: nowrap; overflow: hidden;
    }}
    .segment:hover {{ transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.5); z-index: 10; }}
    .speaker {{ font-weight: 700; }}
    .time {{ font-size: 0.8rem; opacity: 0.9; }}
    .conf {{ margin-left: auto; font-size: 0.8rem; padding: 2px 6px; background: rgba(0,0,0,0.3); border-radius: 4px; }}
    footer {{ margin-top: 2rem; text-align: center; opacity: 0.6; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Diarization Timeline</h1>
    <div class="info">
      {audio_path.name} • {total_seconds:.2f}s • {len(turns)} segments • {len(diarization.labels())} speakers
    </div>

    <div class="timeline">
      <div class="ruler">
        {ticks_html}
      </div>
      <div class="track">
        {rows}
      </div>
    </div>

    <footer>
      Generated with pyannote-audio • Confidence = mean max-speaker probability from raw segmentation model
    </footer>
  </div>
</body>
</html>"""

    html_path.write_text(html, encoding="utf-8")
    console.log(f"[bold cyan]Interactive timeline[/] → {html_path.name}")

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