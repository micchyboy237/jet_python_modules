from pathlib import Path

import numpy as np
import torch
from jet.audio.norm.norm_speech_loudness import (
    normalize_speech_loudness,  # ← added import
)
from jet.audio.speech.silero.speech_types import SpeechSegment
from rich.console import Console
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn
from silero_vad.utils_vad import get_speech_timestamps, read_audio

console = Console()


def _load_model() -> torch.nn.Module:
    """Lazily load the latest Silero VAD model from torch hub."""
    with console.status(
        "[bold green]Downloading latest Silero VAD model...[/bold green]"
    ):
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
    console.print("✅ Silero VAD model ready")
    return model


def _normalize_input(
    audio: str | Path | np.ndarray | torch.Tensor, sampling_rate: int
) -> torch.Tensor:
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


def _preprocess_for_vad(
    audio: np.ndarray | torch.Tensor,
    sampling_rate: int,
    target_lufs: float = -14.0,
    peak_target: float = 0.98,
) -> torch.Tensor:
    """
    Preprocess audio specifically for better VAD performance:
    - Ensure numpy array
    - Downmix to mono if multi-channel
    - Apply speech-weighted loudness normalization
    - Return float32 torch.Tensor suitable for inference
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    if not isinstance(audio, np.ndarray):
        raise TypeError("Expected np.ndarray or torch.Tensor")

    # Downmix to mono
    if audio.ndim > 1:
        if audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            # Simple average downmix (can be improved to energy-weighted later)
            audio = np.mean(audio.astype(np.float64), axis=1).astype(np.float32)

    if audio.ndim != 1:
        raise ValueError(f"Audio must be 1D after downmix, got {audio.shape}")

    # Loudness normalization (speech-prob weighted)
    try:
        audio = normalize_speech_loudness(
            audio=audio,
            sample_rate=sampling_rate,
            target_lufs=target_lufs,
            peak_target=peak_target,
        )
    except Exception as e:
        console.print(
            f"[yellow]Pre-VAD loudness normalization failed: {e} → falling back to original[/yellow]"
        )
        # Keep original (already float32 from earlier steps)

    # Final safety: clip to reasonable range
    max_abs = np.max(np.abs(audio))
    if max_abs > 1.0001 and max_abs > 0:
        audio = audio / max_abs

    return torch.from_numpy(audio).float()


def _run_vad_inference(
    model: torch.nn.Module, audio: torch.Tensor, sampling_rate: int
) -> list[float]:
    """Run VAD inference chunk-by-chunk with a clean rich progress bar and return speech probabilities."""
    window_size_samples = 512 if sampling_rate == 16000 else 256
    model.reset_states()

    total_samples = len(audio)
    step = window_size_samples
    steps = (total_samples + step - 1) // step

    speech_probs: list[float] = []

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
            chunk = audio[i : i + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, window_size_samples - len(chunk))
                )

            prob = model(chunk.unsqueeze(0), sampling_rate).item()
            speech_probs.append(prob)
            progress.update(task, advance=1)

    return speech_probs


@torch.no_grad()
def extract_speech_timestamps(
    audio: str | Path | np.ndarray | torch.Tensor,
    model=None,
    threshold: float = 0.3,
    neg_threshold: float | None = None,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 0,
    return_seconds: bool = False,
    time_resolution: int = 1,
    with_scores: bool = False,
) -> list[SpeechSegment] | tuple[list[SpeechSegment], list[float]]:
    """
    Extract speech timestamps using Silero VAD with enhanced preprocessing.
    """
    if model is None:
        model = _load_model()

    # 1. Basic type / file / dtype normalization
    raw_tensor = _normalize_input(audio, sampling_rate)

    # 2. VAD-specific preprocessing: mono downmix + loudness normalization
    processed_tensor = _preprocess_for_vad(
        raw_tensor,
        sampling_rate=sampling_rate,
        target_lufs=-14.0,
        peak_target=0.98,
    )

    # 3. Run inference on the preprocessed audio
    speech_probs = _run_vad_inference(model, processed_tensor, sampling_rate)

    window_size_samples = 512 if sampling_rate == 16000 else 256

    # Get base timestamps from Silero
    segments = get_speech_timestamps(
        audio=processed_tensor,  # use the normalized version here too
        model=model,
        threshold=threshold,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        neg_threshold=neg_threshold,
        return_seconds=False,
    )

    # Enhance segments with per-segment average probability
    enhanced: list[SpeechSegment] = []
    for idx, seg in enumerate(segments):
        start_sample = seg["start"]
        end_sample = seg["end"]

        start_idx = max(0, start_sample // window_size_samples)
        end_idx = min(
            len(speech_probs),
            (end_sample + window_size_samples - 1) // window_size_samples,
        )

        frames_length = end_idx - start_idx
        frame_start = start_idx
        frame_end = end_idx - 1

        if end_idx > start_idx:
            avg_prob = sum(speech_probs[start_idx:end_idx]) / (end_idx - start_idx)
        else:
            avg_prob = 0.0

        duration_sec = (end_sample - start_sample) / sampling_rate

        enhanced.append(
            SpeechSegment(
                num=idx + 1,
                start=round(start_sample / sampling_rate, time_resolution)
                if return_seconds
                else start_sample,
                end=round(end_sample / sampling_rate, time_resolution)
                if return_seconds
                else end_sample,
                prob=round(avg_prob, 4),
                duration=round(duration_sec, 3),
                frames_length=frames_length,
                frame_start=frame_start,
                frame_end=frame_end,
                segment_probs=speech_probs[start_idx:end_idx] if with_scores else [],
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
