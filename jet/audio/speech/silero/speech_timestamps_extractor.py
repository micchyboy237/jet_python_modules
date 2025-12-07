from typing import List, TypedDict
from pathlib import Path
import numpy as np
import torch
from silero_vad.utils_vad import get_speech_timestamps, read_audio

# ── Rich imports for pretty output ──────────────────────────────
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

console = Console()


class SpeechSegment(TypedDict):
    idx: int
    start: float | int
    end: float | int
    prob: float
    duration: float


@torch.no_grad()
def extract_speech_timestamps(
    audio: str | Path | np.ndarray | torch.Tensor,
    model=None,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 500,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 700,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    time_resolution: int = 2,
) -> List[SpeechSegment]:
    # ── Lazy-load model ─────────────────────────────────────────
    if model is None:
        with console.status("[bold green]Downloading latest Silero VAD model...[/bold green]"):
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
        console.print("✅ Silero VAD model ready")

    # ── Input normalization ─────────────────────────────────────
    if isinstance(audio, (str, Path)):
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio = read_audio(audio_path, sampling_rate=sampling_rate)

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
    audio = audio.float()

    if sampling_rate not in (8000, 16000):
        raise ValueError("Silero VAD supports only 8000 or 16000 Hz")

    # ── VAD inference with clean progress bar ───────────────────
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
            chunk = audio[i : i + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))

            prob = model(chunk.unsqueeze(0), sampling_rate).item()
            speech_probs.append(prob)
            progress.update(task, advance=1)

    # ── Get raw segments from Silero utils ───────────────────────
    segments = get_speech_timestamps(
        audio=audio,
        model=model,
        threshold=threshold,
        sampling_rate=sampling_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )

    # ── Enhance with average probability & nice formatting ───────
    enhanced: List[SpeechSegment] = []
    for idx, seg in enumerate(segments):
        start_sample = seg["start"]
        end_sample = seg["end"]

        start_idx = max(0, start_sample // window_size_samples)
        end_idx = min(len(speech_probs), (end_sample + window_size_samples - 1) // window_size_samples)

        avg_prob = sum(speech_probs[start_idx:end_idx]) / (end_idx - start_idx) if end_idx > start_idx else 0.0
        duration_sec = (end_sample - start_sample) / sampling_rate

        enhanced.append(
            SpeechSegment(
                idx=idx,
                start=round(start_sample / sampling_rate, time_resolution) if return_seconds else start_sample,
                end=round(end_sample / sampling_rate, time_resolution) if return_seconds else end_sample,
                prob=round(avg_prob, 4),
                duration=round(duration_sec, 3),
            )
        )

    return enhanced


if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"

    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")

    segments = extract_speech_timestamps(
        audio_file,
        threshold=0.5,
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