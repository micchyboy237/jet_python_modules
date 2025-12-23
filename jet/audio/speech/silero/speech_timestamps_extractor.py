from typing import List, Tuple
from pathlib import Path
import numpy as np
import torch
from silero_vad.utils_vad import get_speech_timestamps, read_audio

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

from jet.audio.speech.silero.speech_types import SpeechSegment
from jet.audio.speech.silero.speech_utils import get_speech_waves

console = Console()


def _load_model() -> torch.nn.Module:
    """Lazily load the latest Silero VAD model from torch hub."""
    with console.status("[bold green]Downloading latest Silero VAD model...[/bold green]"):
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
    console.print("✅ Silero VAD model ready")
    return model


def _normalize_input(audio: str | Path | np.ndarray | torch.Tensor, sampling_rate: int) -> torch.Tensor:
    """Normalize various input types to a mono float32 torch.Tensor at the target sampling rate."""
    if isinstance(audio, (str, Path)):
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio = read_audio(str(audio_path), sampling_rate=sampling_rate)

    if isinstance(audio, np.ndarray):
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio)

    if not isinstance(audio, torch.Tensor):
        raise TypeError("audio must be torch.Tensor, np.ndarray, str or Path")

    if audio.ndim != 1:
        raise ValueError("Audio must be mono (1D tensor)")

    if sampling_rate not in (8000, 16000):
        raise ValueError("Silero VAD supports only 8000 or 16000 Hz")

    return audio.float()


def _run_vad_inference(model: torch.nn.Module, audio: torch.Tensor, sampling_rate: int) -> List[float]:
    """Run VAD inference chunk-by-chunk with a clean rich progress bar and return speech probabilities."""
    window_size_samples = 512 if sampling_rate == 16000 else 256
    model.reset_states()

    total_samples = len(audio)
    step = window_size_samples
    steps = (total_samples + step - 1) // step

    speech_probs: List[float] = []

    with Progress(
        SpinnerColumn(),
        "[bold blue]{task.description}",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running VAD inference...", total=steps)

        for i in range(0, total_samples, step):
            chunk = audio[i: i + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))

            prob = model(chunk.unsqueeze(0), sampling_rate).item()
            speech_probs.append(prob)
            progress.update(task, advance=1)

    return speech_probs


@torch.no_grad()
def extract_speech_timestamps(
    audio: str | Path | np.ndarray | torch.Tensor,
    model=None,
    threshold: float = 0.3,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    time_resolution: int = 1,
    with_scores: bool = False,
    refine_waves: bool = False,
    wave_threshold: float = 0.7,
) -> List[SpeechSegment] | Tuple[List[SpeechSegment], List[float]]:
    """
    Extract speech timestamps using Silero VAD.
    
    If refine_waves=True, the coarse VAD segments are further split into complete
    speech waves (rise → high → fall) using a higher threshold.
    """
    if model is None:
        model = _load_model()

    audio_tensor = _normalize_input(audio, sampling_rate)

    speech_probs = _run_vad_inference(model, audio_tensor, sampling_rate)

    window_size_samples = 512 if sampling_rate == 16000 else 256

    # Coarse segments from original Silero VAD
    coarse_segments = get_speech_timestamps(
        audio=audio_tensor,
        model=model,
        threshold=threshold,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )

    enhanced: List[SpeechSegment] = []

    segments_to_process = coarse_segments

    if refine_waves:
        # Collect all refined waves and treat each as a final segment
        all_waves: List[Tuple[float, float]] = []
        for coarse_seg in coarse_segments:
            # Temporary SpeechSegment just to reuse get_speech_waves logic
            temp_seg = SpeechSegment(
                num=0,
                start=coarse_seg["start"],
                end=coarse_seg["end"],
                prob=0.0,
                duration=0.0,
                frames_length=0,
                frame_start=0,
                frame_end=0,
                segment_probs=[],
            )
            waves = get_speech_waves(
                temp_seg,
                speech_probs,
                threshold=wave_threshold,
                sampling_rate=sampling_rate,
            )
            all_waves.extend(waves)

        # Convert waves to sample-based segments for consistent processing
        segments_to_process = []
        for wave_start_sec, wave_end_sec in all_waves:
            segments_to_process.append({
                "start": int(round(wave_start_sec * sampling_rate)),
                "end": int(round(wave_end_sec * sampling_rate)),
            })

    # Process final segments (either coarse or refined waves)
    for idx, seg in enumerate(segments_to_process):
        start_sample = seg["start"]
        end_sample = seg["end"]

        start_idx = max(0, start_sample // window_size_samples)
        end_idx = min(len(speech_probs), (end_sample + window_size_samples - 1) // window_size_samples)

        frames_length = end_idx - start_idx
        frame_start = start_idx
        frame_end = end_idx - 1

        segment_prob_slice = speech_probs[start_idx:end_idx]
        avg_prob = (
            sum(segment_prob_slice) / len(segment_prob_slice)
            if segment_prob_slice else 0.0
        )
        duration_sec = (end_sample - start_sample) / sampling_rate

        enhanced.append(
            SpeechSegment(
                num=idx + 1,
                start=round(start_sample / sampling_rate, time_resolution) if return_seconds else start_sample,
                end=round(end_sample / sampling_rate, time_resolution) if return_seconds else end_sample,
                prob=round(avg_prob, 4),
                duration=round(duration_sec, 3),
                frames_length=frames_length,
                frame_start=frame_start,
                frame_end=frame_end,
                segment_probs=segment_prob_slice if with_scores else [],
            )
        )

    if with_scores:
        return enhanced, speech_probs
    return enhanced


if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"

    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")

    segments = extract_speech_timestamps(
        audio_file,
        threshold=0.3,
        sampling_rate=16000,
        return_seconds=True,
        time_resolution=2,
    )

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white] - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"duration=[bold magenta]{seg['duration']}s[/bold magenta] "
            f"prob=[bold cyan]{seg['prob']:.3f}[/bold cyan]"
        )