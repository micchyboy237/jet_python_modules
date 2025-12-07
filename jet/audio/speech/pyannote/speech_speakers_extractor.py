# speech_speakers_extractor.py
from typing import List, TypedDict, Optional
from pathlib import Path
import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

# --- Silero VAD (optional, strong accuracy boost) ---
from silero_vad.utils_vad import get_speech_timestamps

console = Console()

class SpeechSpeakerSegment(TypedDict):
    idx: int
    start: float
    end: float
    speaker: str
    duration: float
    prob: float  # Pyannote does not provide per-segment probability

def _merge_speaker_turns(
    raw_segments: List[SpeechSpeakerSegment],
    min_duration: float = 0.4,
    max_gap: float = 0.35,
) -> List[SpeechSpeakerSegment]:
    """
    Merge overlapping/contiguous turns and remove silence fragments.
    """
    if not raw_segments:
        return []

    # Sort by start time
    raw_segments = sorted(raw_segments, key=lambda x: x["start"])
    merged: List[SpeechSpeakerSegment] = []
    current = raw_segments[0].copy()

    for next_seg in raw_segments[1:]:
        same_speaker = next_seg["speaker"] == current["speaker"]
        gap = next_seg["start"] - current["end"]

        # Merge if same speaker and overlapping or very close
        if same_speaker and gap <= max_gap:
            current["end"] = max(current["end"], next_seg["end"])
            current["duration"] = round(current["end"] - current["start"], 3)
        else:
            if current["duration"] >= min_duration:
                merged.append(current)
            current = next_seg.copy()

    if current["duration"] >= min_duration:
        merged.append(current)

    # Re-index
    for i, seg in enumerate(merged):
        seg["idx"] = i
    return merged

@torch.no_grad()
def extract_speech_speakers(
    audio: str | Path | np.ndarray | torch.Tensor,
    threshold: float = 0.5,  # Only used if use_silero_vad=True
    sampling_rate: int = 16000,
    time_resolution: int = 3,   # Higher precision
    min_duration: float = 0.45, # min segment length after merge (sec)
    max_gap: float = 0.35,      # max gap for merge (sec)
    use_silero_vad: bool = True,
    vad_threshold: float = 0.5,
    min_speech_duration_ms: int = 500,
    min_silence_duration_ms: int = 700,
    speech_pad_ms: int = 30,
) -> List[SpeechSpeakerSegment]:
    """
    Extract **clean** speaker diarization segments using pyannote-audio.

    Optionally uses Silero VAD (recommended) for dramatically better speech detection.
    Added post-processing to merge cut/too-short micro-segments.
    """
    # Optionally: Load Silero VAD if requested
    vad_model = None
    if use_silero_vad:
        with console.status("[bold green]Loading Silero VAD model...[/bold green]"):
            vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
        console.print("✅ Silero VAD ready")

    # Load pyannote model with status
    with console.status("[bold green]Loading pyannote speaker diarization model...[/bold green]"):
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
            )
            console.print("✅ Pyannote model loaded")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model: {e}. Ensure model conditions accepted at https://hf.co/pyannote/speaker-diarization-community-1"
            )
    # Auto GPU if available
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        console.print("🚀 Using GPU for inference")

    # Audio input handling and forced mono preparation
    if isinstance(audio, (str, Path)):
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            console.print(f"[yellow]⚠️  Audio has {waveform.shape[0]} channels → downmixing to mono[/yellow]")
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sampling_rate:
            console.print(f"[dim]Resampling from {sr} Hz → {sampling_rate} Hz[/dim]")
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            waveform = resampler(waveform)
        audio = waveform.squeeze(0).numpy()
    elif isinstance(audio, np.ndarray):
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            if audio.shape[0] > 1:
                console.print(f"[yellow]⚠️  NumPy input has {audio.shape[0]} channels → downmixing to mono[/yellow]")
                audio = audio.mean(axis=0)
            else:
                audio = audio.squeeze()
    elif isinstance(audio, torch.Tensor):
        if audio.dtype == torch.int16:
            audio = audio.float() / 32768.0
        elif audio.dtype != torch.float32:
            audio = audio.float()
        if audio.ndim > 1:
            if audio.shape[0] > 1:
                console.print(f"[yellow]⚠️  Torch input has {audio.shape[0]} channels → downmixing to mono[/yellow]")
                audio = audio.mean(dim=0)
            else:
                audio = audio.squeeze(dim=0)
        audio = audio.numpy()
    else:
        raise TypeError("audio must be torch.Tensor, np.ndarray, str or Path")

    if audio.ndim != 1:
        raise ValueError("Audio must be mono (1D array)")

    # ALWAYS: Convert to torch.tensor for consistency downstream
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(audio)
    if audio.dtype != torch.float32:
        audio = audio.float()
    # Always (1, N) shape for pyannote
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    console.print("✅ Audio prepared as torch tensor (mono, 16kHz)")

    # --- OPTIONAL PRE-FILTERING BY SILERO VAD ---
    speech_mask: Optional[torch.Tensor] = None
    if use_silero_vad:
        vad_waveform = audio.clone().squeeze(0)  # Silero expects 1D float32
        if vad_waveform.dtype != torch.float32:
            vad_waveform = vad_waveform.float()

        speech_ts = get_speech_timestamps(
            vad_waveform,
            vad_model,
            threshold=vad_threshold,
            sampling_rate=sampling_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=False,
        )

        if not speech_ts:
            console.print("[bold red]Silero VAD found no speech – returning empty result[/bold red]")
            return []

        # Build mask
        mask = torch.zeros_like(vad_waveform)
        for seg in speech_ts:
            start = seg["start"]
            end = seg["end"]
            mask[start:end] = 1.0
        # Soft fade at edges
        fade_samples = int(0.02 * sampling_rate)
        if fade_samples > 0 and fade_samples < mask.numel():
            mask[:fade_samples] *= torch.linspace(0, 1, fade_samples)
            mask[-fade_samples:] *= torch.linspace(1, 0, fade_samples)
        # Apply mask: silence regions → -80 dB (almost zero but not exactly zero)
        audio = audio * mask.unsqueeze(0)
        speech_mask = mask
        console.print(f"✅ Silero VAD applied → kept {len(speech_ts)} speech regions")

    pyannote_input = {"waveform": audio, "sample_rate": sampling_rate}

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
            output = pipeline(pyannote_input, hook=hook)
        progress.update(task, completed=1)

    # Build raw segments
    raw: List[SpeechSpeakerSegment] = []
    for idx, (turn, _, speaker) in enumerate(
        output.speaker_diarization.itertracks(yield_label=True)
    ):
        raw.append(
            SpeechSpeakerSegment(
                idx=idx,
                start=round(turn.start, time_resolution),
                end=round(turn.end, time_resolution),
                speaker=speaker,
                duration=round(turn.end - turn.start, 3),
                prob=1.0,
            )
        )

    # Merge & clean segments
    cleaned_segments = _merge_speaker_turns(
        raw_segments=raw,
        min_duration=min_duration,
        max_gap=max_gap,
    )

    console.print(
        f"[bold green]Speaker diarization done → {len(raw)} raw → {len(cleaned_segments)} cleaned segments[/bold green]"
        + (" ([cyan]Silero VAD pre-filtered[/cyan])" if use_silero_vad else "")
    )
    return cleaned_segments

if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"
    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")
    segments = extract_speech_speakers(
        audio_file,
        threshold=0.5,
        time_resolution=2,
        use_silero_vad=True,
    )
    console.print(f"\n[bold green]Speaker segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        console.print(
            f"[yellow][[/yellow] [bold white]{seg['start']:.2f}[/bold white] - [bold white]{seg['end']:.2f}[/bold white] [yellow]][/yellow] "
            f"speaker=[bold magenta]{seg['speaker']}[/bold magenta] "
            f"duration=[bold cyan]{seg['duration']}s[/bold cyan] "
            f"prob=[bold green]{seg['prob']:.3f}[/bold green]"
        )