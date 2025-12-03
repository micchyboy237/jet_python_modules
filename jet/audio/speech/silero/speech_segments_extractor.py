from typing import List, TypedDict, Callable
from pathlib import Path
import numpy as np
import torch
from silero_vad.utils_vad import get_speech_timestamps, read_audio


class SpeechSegment(TypedDict):
    idx: int
    start: int | float
    end: int | float
    prob: float
    duration: float


@torch.no_grad()
def extract_speech_timestamps(
    audio: str | Path | np.ndarray | torch.Tensor,
    model=None,  # ← now optional
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float('inf'),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    time_resolution: int = 1,
    progress_tracking_callback: Callable[[float], None] | None = None,
) -> List[SpeechSegment]:
    """
    Accepts audio as file path, np.ndarray or torch.Tensor.
    If model is None → automatically downloads the latest Silero VAD model.
    """
    # ── Lazy-load model if not provided ─────────────────────────────
    if model is None:
        from rich.console import Console
        console = Console()
        console.print("[bold green]Downloading latest Silero VAD model...[/bold green]", end="")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
        console.print(" [bold green]done[/bold green]")

    # ── Input normalization (unchanged) ─────────────────────────────
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
    if audio.dtype != torch.float32:
        audio = audio.float()

    # ── Rest of the function (100% unchanged) ───────────────────────
    if sampling_rate not in (8000, 16000):
        raise ValueError("Silero VAD supports only 8000 or 16000 Hz")

    window_size_samples = 512 if sampling_rate == 16000 else 256
    model.reset_states()

    speech_probs: List[float] = []
    total_samples = len(audio)
    step = window_size_samples

    for i, start in enumerate(range(0, total_samples, step)):
        end = min(start + window_size_samples, total_samples)
        chunk = audio[start:end]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))

        prob = model(chunk.unsqueeze(0), sampling_rate).item()
        speech_probs.append(prob)

        if progress_tracking_callback:
            progress = min((i + 1) * step, total_samples) / total_samples * 100
            progress_tracking_callback(progress)

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

    enhanced: List[SpeechSegment] = []
    for idx, seg in enumerate(segments):
        start_sample = seg["start"]
        end_sample = seg["end"]

        start_idx = max(0, start_sample // window_size_samples)
        end_idx = min(len(speech_probs), (end_sample + window_size_samples - 1) // window_size_samples)

        window_probs = speech_probs[start_idx:end_idx]
        avg_prob = sum(window_probs) / len(window_probs) if window_probs else 0.0
        duration_sec = (end_sample - start_sample) / sampling_rate

        enhanced.append(SpeechSegment(
            idx=idx,
            start=round(start_sample / sampling_rate, time_resolution) if return_seconds else start_sample,
            end=round(end_sample / sampling_rate, time_resolution) if return_seconds else end_sample,
            prob=round(avg_prob, 4),
            duration=round(duration_sec, 3),
        ))

    return enhanced

if __name__ == "__main__":
    from silero_vad.utils_vad import read_audio
    import torch

    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic_stream/recording_20251126_212124.wav"
    segments = extract_speech_timestamps(
        audio_file,
        threshold=0.5,
        sampling_rate=16000,
        return_seconds=True,
        progress_tracking_callback=print  # optional
    )

    print(f"Segments: {len(segments)}")
    for seg in segments[:5]:
        print(f"[{seg['start']:.2f} - {seg['end']:.2f}] duration={seg['duration']}s prob={seg['prob']:.3f}")
