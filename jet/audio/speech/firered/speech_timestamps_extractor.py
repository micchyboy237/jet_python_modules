import json
import shutil
from pathlib import Path
from typing import List, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from jet.audio.helpers.energy_base import compute_rms_per_frame
from jet.audio.speech.firered.config import SAVE_DIR
from jet.audio.speech.firered.speech_types import SpeechSegment
from jet.audio.speech.firered.vad import FireRedVAD
from jet.audio.utils import load_audio
from rich.console import Console

console = Console()


def extract_speech_timestamps(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    threshold: float = 0.5,
    min_silence_duration_sec: float = 0.250,
    min_speech_duration_sec: float = 0.250,
    max_speech_duration_sec: float | None = None,
    return_seconds: bool = False,
    with_scores: bool = False,
    include_non_speech: bool = False,
    **kwargs,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech timestamps using FireRedVAD.
    When include_non_speech=True, returns both speech and non-speech (silence) segments.
    """
    if max_speech_duration_sec is None:
        max_speech_duration_sec = 15.0
    # Convert input audio to numpy array
    audio_np, sr = load_audio(
        audio,
        sr=16000,  # FireRedVAD expects 16000 Hz
        mono=True,
    )
    if sr != 16000:
        raise ValueError(f"FireRedVAD requires 16000 Hz, got {sr}")

    # Initialize FireRedVAD
    vad = FireRedVAD(
        model_dir=SAVE_DIR,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
    )

    # Run VAD inference
    with console.status("[bold blue]Running FireRedVAD inference...[/bold blue]"):
        frame_results, result = vad.detect_full(audio_np)

    # Extract timestamps
    timestamps = result["timestamps"]
    probs = [r.smoothed_prob for r in frame_results]
    hop_sec = 0.010  # FireRedVAD frame shift (10ms)

    def make_segment(
        num: int,
        start_sec: float,
        end_sec: float,
        seg_type: Literal["speech", "non-speech"],
    ) -> SpeechSegment:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        frame_start = int(start_sec / hop_sec)
        frame_end = int(end_sec / hop_sec)
        segment_probs_slice = probs[frame_start : frame_end + 1]
        avg_prob = np.mean(segment_probs_slice) if segment_probs_slice else 0.0
        duration_sec = end_sec - start_sec
        start_val = start_sec if return_seconds else start_sample
        end_val = end_sec if return_seconds else end_sample
        return SpeechSegment(
            num=num,
            start=start_val,
            end=end_val,
            prob=avg_prob,
            duration=duration_sec,
            frames_length=len(segment_probs_slice),
            frame_start=frame_start,
            frame_end=frame_end,
            type=seg_type,
            segment_probs=segment_probs_slice if with_scores else [],
        )

    enhanced: List[SpeechSegment] = []
    current_time = 0.0
    seg_num = 1

    # Handle initial non-speech segment
    if include_non_speech and timestamps and timestamps[0][0] > 0.001:
        enhanced.append(make_segment(seg_num, 0.0, timestamps[0][0], "non-speech"))
        seg_num += 1
        current_time = timestamps[0][0]

    # Process speech segments
    for start_sec, end_sec in timestamps:
        if include_non_speech and start_sec > current_time + 0.01:
            enhanced.append(
                make_segment(seg_num, current_time, start_sec, "non-speech")
            )
            seg_num += 1
        enhanced.append(make_segment(seg_num, start_sec, end_sec, "speech"))
        seg_num += 1
        current_time = end_sec

    # Handle final non-speech segment
    total_duration = result["dur"]
    if include_non_speech and current_time < total_duration - 0.01:
        enhanced.append(
            make_segment(seg_num, current_time, total_duration, "non-speech")
        )

    if with_scores:
        return enhanced, probs
    return enhanced


def extract_speech_audio(
    audio: Union[str, Path, np.ndarray, torch.Tensor, list[np.ndarray]],
    sampling_rate: int = 16000,
    threshold: float = 0.5,
    min_silence_duration_sec: float = 0.250,
    min_speech_duration_sec: float = 0.250,
    max_speech_duration_sec: float | None = None,
) -> List[np.ndarray]:
    """
    Extract contiguous speech segments from the input audio using FireRedVAD.
    Returns a flat list of numpy arrays where each array represents one complete
    speech segment in float32 format, normalized to [-1.0, 1.0].
    """
    if sampling_rate != 16000:
        raise ValueError(f"FireRedVAD requires 16000 Hz, got {sampling_rate}")

    speech_segments = extract_speech_timestamps(
        audio=audio,
        threshold=threshold,
        min_silence_duration_sec=min_silence_duration_sec,
        min_speech_duration_sec=min_speech_duration_sec,
        max_speech_duration_sec=max_speech_duration_sec,
        return_seconds=True,
        include_non_speech=False,
    )

    audio_np, sr = load_audio(
        audio=audio,
        sr=sampling_rate,
        mono=True,
    )
    if sr != sampling_rate:
        raise ValueError(
            f"Loaded sample rate {sr} does not match requested {sampling_rate}"
        )

    speech_audio_chunks: List[np.ndarray] = []
    for segment in speech_segments:
        start_sec: float = segment["start"]
        end_sec: float = segment["end"]
        start_sample = int(round(start_sec * sr))
        end_sample = int(round(end_sec * sr))
        segment_audio = audio_np[start_sample:end_sample]
        if len(segment_audio) == 0:
            continue
        segment_audio = segment_audio.astype(np.float32, copy=False)
        speech_audio_chunks.append(segment_audio)

    return speech_audio_chunks


def save_plot(
    probs: List[float],
    rms_values: List[float],
    output_path: Path,
    title: str = "Speech Probabilities and RMS Energy",
) -> None:
    """Create a dual subplot figure and save as PNG."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    frames = np.arange(len(probs))
    ax1.plot(frames, probs, color="blue", linewidth=1)
    ax1.set_ylabel("VAD Probability")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)
    ax2.plot(frames, rms_values, color="green", linewidth=1)
    ax2.set_xlabel("Frame Index")
    ax2.set_ylabel("RMS Energy")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_json(data, path: Path) -> None:
    """Save data as JSON with indentation."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    import argparse

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    DEFAULT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"
    parser = argparse.ArgumentParser(
        description="Extract speech timestamps from audio using TEN VAD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_AUDIO,
        help=f"Input audio file path (default: {DEFAULT_AUDIO})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(OUTPUT_DIR),
        help=f"Output results dir (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5, help="VAD probability threshold"
    )
    parser.add_argument(
        "-s", "--hop-size", type=int, default=160, help="Frame hop size in samples"
    )
    parser.add_argument(
        "--min-speech-duration",
        "-d",
        type=int,
        default=250,
        help="Minimum speech segment duration in ms",
    )
    parser.add_argument(
        "--min-silence-duration",
        "-g",
        type=int,
        default=100,
        help="Minimum silence duration in ms",
    )
    parser.add_argument(
        "--include-non-speech",
        "-n",
        action="store_true",
        help="Include non-speech segments",
    )
    args = parser.parse_args()

    segments, scores = extract_speech_timestamps(
        audio=args.input,
        include_non_speech=args.include_non_speech,
        hop_size=args.hop_size,
        threshold=args.threshold,
        min_speech_duration_ms=args.min_speech_duration,
        min_silence_duration_ms=args.min_silence_duration,
        with_scores=True,
    )

    audio_np, sr = load_audio(args.input, sr=16000, mono=True)

    output_dir = Path(args.output)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    speech_segments_json = output_dir / "speech_segments.json"
    speech_probs_json = output_dir / "speech_probs.json"

    with open(speech_segments_json, "w") as f:
        json.dump(
            {
                "count": len(segments),
                "segments": segments,
            },
            f,
            indent=2,
        )
    segments_url = f"file://{speech_segments_json.resolve()}"
    console.print(
        f"[green]{len(segments)} Segments saved to [link={segments_url}]{speech_segments_json}[/link][/green]"
    )

    with open(speech_probs_json, "w") as f:
        json.dump(scores, f, indent=2)
    probs_url = f"file://{speech_probs_json.resolve()}"
    console.print(
        f"[green]{len(scores)} probabilities saved to [link={probs_url}]{speech_probs_json}[/link][/green]"
    )

    console.print(
        f"\n[bold]Generating {len(segments)} per‑segment files for {len(segments)} speech segments...[/bold]"
    )
    for seg in segments:
        num = seg["num"]
        seg_subdir = segments_dir / f"segment_{num:03d}"
        seg_subdir.mkdir(exist_ok=True)

        start_frame = seg["frame_start"]
        end_frame = seg["frame_end"]
        hop_size = args.hop_size
        start_sample = start_frame * hop_size
        end_sample = (end_frame + 1) * hop_size
        segment_audio_np = audio_np[start_sample:end_sample]

        wav_path = seg_subdir / "sound.wav"
        wavfile.write(wav_path, sr, segment_audio_np)

        segment_probs = seg["segment_probs"]
        probs_path = seg_subdir / "speech_probs.json"
        save_json(segment_probs, probs_path)

        rms_values = compute_rms_per_frame(audio_np, hop_size, start_frame, end_frame)
        energies_path = seg_subdir / "energies.json"
        save_json(rms_values, energies_path)

        segment_json_path = seg_subdir / "segment.json"
        save_json(seg, segment_json_path)

        plot_path = seg_subdir / "speech_and_rms.png"
        save_plot(segment_probs, rms_values, plot_path, title=f"Segment {num:03d}")

        color = "green" if seg["type"] == "speech" else "red"
        type_label = seg["type"].replace("-", "_")
        console.print(
            f"  [cyan]✓[/cyan] segment_{num:03d}: [bold cyan]{len(segment_probs)}[/bold cyan] frames,  [bold cyan]{seg['duration']:.2f}[/bold cyan]s, avg_prob={seg['prob']:.3f}, avg_rms={np.mean(rms_values):.3f}, [{color}]{type_label}[/{color}]"
        )

    console.print(
        f"\n[bold green]All {len(segments)} per‑segment files saved under: {segments_dir}[/bold green]"
    )
