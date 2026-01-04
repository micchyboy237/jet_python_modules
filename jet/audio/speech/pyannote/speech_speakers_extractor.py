# speech_speakers_extractor.py
import io
from typing import List, Sequence, TypedDict, Union
from pathlib import Path
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

# Silero VAD
from silero_vad.utils_vad import get_speech_timestamps
import torchaudio
from tqdm import tqdm

from jet.audio.utils import resolve_audio_paths

console = Console()


AudioInput = Union[str, Path, np.ndarray, torch.Tensor, bytes]

class SpeechSpeakerSegment(TypedDict):
    idx: int
    start: float
    end: float
    speaker: str
    duration: float
    prob: float


def _post_process_diarization(
    segments: List[SpeechSpeakerSegment],
    min_duration: float = 0.45,
    max_gap: float = 0.35,
    *,
    overlap_threshold_ratio: float = 0.30,
    momentum_threshold_ratio: float = 0.60,
    intruder_max_ratio: float = 0.85,
) -> List[SpeechSpeakerSegment]:
    """
    Fully adaptive post-processing with dynamic thresholds based on actual turn statistics.
    No more brittle magic numbers like 0.2s, 1.0s, 1.5s.
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x["start"])

    # Compute statistics from real turns (after basic min_duration filter)
    valid_durations = [s["duration"] for s in segments if s["duration"] >= min_duration]
    if not valid_durations:
        median_dur = 2.0  # safe fallback
    else:
        median_dur = float(np.median(valid_durations))

    # Dynamic thresholds
    momentum_threshold = max(0.7, median_dur * momentum_threshold_ratio)
    intruder_threshold = median_dur * intruder_max_ratio

    merged: List[SpeechSpeakerSegment] = []
    current = segments[0].copy()

    for next_seg in segments[1:]:
        gap = next_seg["start"] - current["end"]
        overlap_dur = max(0.0, current["end"] - next_seg["start"])
        shorter_dur = min(current["duration"], next_seg["duration"])
        overlap_ratio = overlap_dur / shorter_dur if shorter_dur > 0 else 0.0

        # 1. Same speaker + close in time → merge
        if next_seg["speaker"] == current["speaker"] and (gap <= max_gap or overlap_dur > 0):
            current["end"] = max(current["end"], next_seg["end"])
            current["duration"] = round(current["end"] - current["start"], 3)
            continue

        # 2. Significant overlap → adaptive conflict resolution
        if overlap_ratio >= overlap_threshold_ratio:
            current_has_momentum = current["duration"] >= momentum_threshold
            next_is_intruder = next_seg["duration"] < intruder_threshold

            if current_has_momentum and next_is_intruder:
                # Absorb short wrong-speaker glitch
                current["end"] = max(current["end"], next_seg["end"])
                current["duration"] = round(current["end"] - current["start"], 3)
                continue
            elif next_seg["duration"] > current["duration"] * 1.8 and not current_has_momentum:
                # Current was likely noise → replace with stronger turn
                current = next_seg.copy()
                continue
            else:
                # Real speaker overlap → keep both
                merged.append(current)
                current = next_seg.copy()
        else:
            # No significant overlap
            if current["duration"] >= min_duration:
                merged.append(current)
            current = next_seg.copy()

    # Don't forget the last segment
    if current["duration"] >= min_duration:
        merged.append(current)

    # Re-index
    for i, seg in enumerate(merged):
        seg["idx"] = i

    return merged


@torch.no_grad()
def extract_speech_speakers(
    audio: AudioInput,
    sampling_rate: int = 16000,
    time_resolution: int = 3,
    min_duration: float = 0.45,
    max_gap: float = 0.35,
    use_silero_vad: bool = True,
    vad_threshold: float = 0.5,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 700,
    speech_pad_ms: int = 30,
) -> List[SpeechSpeakerSegment]:
    # --- Load Silero VAD ---
    vad_model = None
    if use_silero_vad:
        with console.status("[bold green]Loading Silero VAD...[/bold green]"):
            vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                verbose=False,
            )
        console.print("Silero VAD ready")

    # --- Load pyannote pipeline ---
    with console.status("[bold green]Loading pyannote diarization model...[/bold green]"):
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
        console.print("Pyannote model loaded")

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        console.print("Using GPU")

    # --- Audio input handling ---
    if isinstance(audio, (str, Path)):
        waveform, sr = torchaudio.load(str(audio))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            waveform = resampler(waveform)
        audio_tensor = waveform.squeeze(0)

    elif isinstance(audio, bytes):
        # Handle raw bytes (assumed to be a valid audio file in memory)
        buffer = io.BytesIO(audio)
        waveform, sr = torchaudio.load(buffer)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            waveform = resampler(waveform)
        audio_tensor = waveform.squeeze(0)

    elif isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio.astype(np.float32))
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=0)

    elif isinstance(audio, torch.Tensor):
        audio_tensor = audio.float()
        if audio_tensor.ndim > 1 and audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0)

    else:
        raise TypeError("audio must be file path, bytes, np.ndarray, or torch.Tensor")

    audio_tensor = audio_tensor.unsqueeze(0)  # (1, T)

    # --- Silero VAD masking ---
    if use_silero_vad and vad_model is not None:
        speech_ts = get_speech_timestamps(
            audio_tensor.squeeze(0),
            vad_model,
            threshold=vad_threshold,
            sampling_rate=sampling_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        if not speech_ts:
            console.print("[bold red]No speech detected by Silero VAD[/bold red]")
            return []

        mask = torch.zeros_like(audio_tensor.squeeze(0))
        for seg in speech_ts:
            mask[seg["start"]:seg["end"]] = 1.0
        fade = int(0.02 * sampling_rate)
        if fade > 0:
            mask[:fade] *= torch.linspace(0, 1, fade)
            mask[-fade:] *= torch.linspace(1, 0, fade)
        audio_tensor = audio_tensor * mask.unsqueeze(0)
        console.print(f"Silero kept {len(speech_ts)} speech regions")

    # --- Run pyannote diarization ---
    with Progress(
        SpinnerColumn(),
        "[bold blue]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running speaker diarization...", total=1)
        with ProgressHook() as hook:
            diarization = pipeline({"waveform": audio_tensor, "sample_rate": sampling_rate}, hook=hook)
        progress.update(task, completed=1)

    # --- Build raw segments (supports both legacy and full pipeline output) ---
    if hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    else:
        annotation = diarization  # community model returns Annotation directly

    raw_segments: List[SpeechSpeakerSegment] = [
        SpeechSpeakerSegment(
            idx=i,
            start=round(turn.start, time_resolution),
            end=round(turn.end, time_resolution),
            speaker=speaker,
            duration=round(turn.end - turn.start, 3),
            prob=1.0,
        )
        for i, (turn, _, speaker) in enumerate(annotation.itertracks(yield_label=True))
    ]

    # --- Adaptive post-processing ---
    processed_segments = _post_process_diarization(
        raw_segments,
        min_duration=min_duration,
        max_gap=max_gap,
    )

    # --- Final clean segments ---
    final_segments = [
        SpeechSpeakerSegment(
            idx=i,
            start=round(s["start"], time_resolution),
            end=round(s["end"], time_resolution),
            speaker=s["speaker"],
            duration=round(s["end"] - s["start"], 3),
            prob=1.0,
        )
        for i, s in enumerate(processed_segments)
        if s["duration"] >= min_duration
    ]

    console.print(
        f"[bold green]Diarization complete:[/bold green] "
        f"{len(raw_segments)} → {len(processed_segments)} → {len(final_segments)} segments"
        + (" [cyan](VAD pre-filtered)[/cyan]" if use_silero_vad else "")
    )

    return final_segments


def batch_extract_speech_speakers(  # Renamed from extract_speech_speakers_from_files
    audio_inputs: Sequence[AudioInput],  # Now accepts List[AudioInput] or any sequence
    sampling_rate: int = 16000,
    time_resolution: int = 3,
    min_duration: float = 0.45,
    max_gap: float = 0.35,
    use_silero_vad: bool = True,
    vad_threshold: float = 0.5,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 700,
    speech_pad_ms: int = 30,
) -> List[SpeechSpeakerSegment]:
    """
    Process multiple audio inputs sequentially, merging segments with adjusted timestamps.
    Supports file paths, bytes, numpy arrays, and torch tensors.
    """
    all_segments: List[SpeechSpeakerSegment] = []
    cumulative_duration = 0.0

    with tqdm(total=len(audio_inputs), desc="Processing audio inputs", colour="cyan") as pbar:
        for idx, audio_input in enumerate(audio_inputs):
            # Get display name for progress
            input_name = (
                Path(audio_input).name if isinstance(audio_input, (str, Path))
                else f"input_{idx}"
                if isinstance(audio_input, (bytes, np.ndarray, torch.Tensor))
                else "unknown"
            )
            console.print(f"[bold cyan]Processing {idx+1}/{len(audio_inputs)}: {input_name}[/bold cyan]")

            # Get duration BEFORE processing (for file paths/bytes) or estimate for tensors/arrays
            if isinstance(audio_input, (str, Path)):
                waveform, sr = torchaudio.load(str(audio_input))
                input_duration = waveform.shape[1] / sr
            elif isinstance(audio_input, bytes):
                buffer = io.BytesIO(audio_input)
                waveform, sr = torchaudio.load(buffer)
                input_duration = waveform.shape[1] / sr
            else:  # np.ndarray or torch.Tensor
                # Assume already at target sampling_rate or estimate
                if isinstance(audio_input, torch.Tensor):
                    input_duration = audio_input.shape[-1] / sampling_rate
                else:  # np.ndarray
                    input_duration = len(audio_input) / sampling_rate

            # Run diarization on this input
            input_segments = extract_speech_speakers(
                audio=audio_input,
                sampling_rate=sampling_rate,
                time_resolution=time_resolution,
                min_duration=min_duration,
                max_gap=max_gap,
                use_silero_vad=use_silero_vad,
                vad_threshold=vad_threshold,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
            )

            # Adjust timestamps for sequential merging
            for seg in input_segments:
                seg["start"] += cumulative_duration
                seg["end"] += cumulative_duration
                seg["idx"] = len(all_segments)  # Re-index globally
                all_segments.append(seg)

            # Update cumulative duration
            cumulative_duration += input_duration
            pbar.update(1)
            pbar.set_postfix({"segments": len(input_segments)})

    console.print(f"[bold green]Processed {len(audio_inputs)} inputs with {len(all_segments)} total segments.[/bold green]")
    return all_segments


# if __name__ == "__main__":
#     audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"
#     console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")

#     segments = extract_speech_speakers(
#         audio_file,
#         time_resolution=2,
#         use_silero_vad=True,
#     )

#     console.print(f"\n[bold green]{len(segments)} final speaker segments:[/bold green]\n")
#     for s in segments:
#         console.print(
#             f"[yellow][[/yellow] {s['start']:6.2f} - {s['end']:6.2f} [yellow]][/yellow] "
#             f"{s['speaker']:>10} | {s['duration']:5.2f}s"
#         )

if __name__ == "__main__":
    audio_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Windows_Workspace/servers/live_subtitles/generated/live_subtitles_client_with_overlay/segments"
    audio_files = resolve_audio_paths(audio_dir, recursive=True)
    audio_files = audio_files[:5]

    console.print("[bold magenta]Processing multiple audio files...[/bold magenta]")

    segments = batch_extract_speech_speakers(
        audio_inputs=audio_files,
        time_resolution=2,
        use_silero_vad=True,
    )

    console.print(f"\n[bold green]{len(segments)} final speaker segments across all files:[/bold green]\n")
    for s in segments:
        console.print(
            f"[yellow][[/yellow] {s['start']:6.2f} - {s['end']:6.2f} [yellow]][/yellow] "
            f"{s['speaker']:>10} | {s['duration']:5.2f}s"
        )