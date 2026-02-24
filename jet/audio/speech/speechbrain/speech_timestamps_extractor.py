import os
import tempfile
from pathlib import Path
from typing import List, Literal, Optional, TypedDict, Union

import numpy as np
import torch
import torchaudio
from jet.audio.norm.norm_speech_loudness import normalize_speech_loudness
from rich.console import Console
from speechbrain.inference.VAD import VAD

console = Console()


class SpeechSegment(TypedDict):
    num: int
    start: Union[float, int]
    end: Union[float, int]
    prob: float
    duration: float
    frames_length: int
    frame_start: int
    frame_end: int
    type: Literal["speech", "non-speech"]
    # Only present when with_scores=True
    segment_probs: List[float]
    # speech_waves: List[SpeechWave]  # commented / optional


def _load_speechbrain_vad() -> VAD:
    """Lazily load the SpeechBrain CRDNN VAD model."""
    with console.status("[bold green]Loading SpeechBrain VAD model...[/bold green]"):
        vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad-crdnn-libriparty",
            # run_opts={"device": "cuda"}  # uncomment for GPU (GTX 1660)
        )
    console.print("✅ SpeechBrain VAD model ready")
    return vad


def _prepare_mono_16khz_path(
    audio: Union[str, Path, np.ndarray, torch.Tensor],
    sampling_rate: Optional[int] = None,
    target_lufs: float = -14.0,
    peak_target: float = 0.98,
    normalize_loudness: bool = False,
) -> tuple[str, int]:
    """
    Load/convert audio → mono, 16 kHz, apply loudness norm if requested,
    save to temporary WAV file and return path + sample_rate (always 16000).
    Caller must delete the temp file after use.
    """
    # 1. Load to waveform tensor
    if isinstance(audio, (str, Path)):
        waveform, sr = torchaudio.load(str(Path(audio)))
    elif isinstance(audio, np.ndarray):
        waveform = torch.from_numpy(audio).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        sr = sampling_rate
        if sr is None:
            raise ValueError("sampling_rate required when audio is numpy array")
    elif isinstance(audio, torch.Tensor):
        waveform = audio.float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        sr = sampling_rate
        if sr is None:
            raise ValueError("sampling_rate required when audio is tensor")
    else:
        raise TypeError("audio must be path (str/Path), np.ndarray, or torch.Tensor")

    # 2. Force mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 3. Resample to 16 kHz (model requirement)
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # 4. Apply loudness normalization (speech-weighted)
    audio_np = waveform.squeeze(0).cpu().numpy()
    if normalize_loudness:
        try:
            audio_np = normalize_speech_loudness(
                audio=audio_np,
                sample_rate=target_sr,
                target_lufs=target_lufs,
                peak_target=peak_target,
            )
        except Exception as e:
            console.print(
                f"[yellow]Loudness normalization failed: {e} — using original[/yellow]"
            )

    # 5. Safety clip + back to tensor
    waveform = torch.from_numpy(audio_np).unsqueeze(0).clamp(-1.0, 1.0)

    # 6. Save to temporary file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, waveform, target_sr)

    return tmp.name, target_sr


@torch.no_grad()
def extract_speech_timestamps(
    audio: Union[str, Path, np.ndarray, torch.Tensor],
    vad: Optional[VAD] = None,
    threshold: float = 0.5,  # activation_th
    neg_threshold: float = 0.25,  # deactivation_th
    sampling_rate: Optional[int] = None,  # only needed for array/tensor input
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 0,
    return_seconds: bool = False,
    time_resolution: int = 2,
    with_scores: bool = False,
    normalize_loudness: bool = False,
    include_non_speech: bool = False,
    large_chunk_size: int = 30,
    small_chunk_size: int = 10,
    double_check: bool = False,
) -> Union[List[SpeechSegment], tuple[List[SpeechSegment], List[float]]]:
    """
    Extract speech timestamps using SpeechBrain VAD (vad-crdnn-libriparty).
    When include_non_speech=True, returns both speech and non-speech (silence) segments.
    """
    if vad is None:
        vad = _load_speechbrain_vad()

    # Prepare temporary mono 16 kHz file
    temp_path, sr = _prepare_mono_16khz_path(
        audio,
        sampling_rate=sampling_rate,
        normalize_loudness=normalize_loudness,
    )

    try:
        with console.status(
            "[bold blue]Running SpeechBrain VAD inference...[/bold blue]"
        ):
            # Get speech boundaries in seconds
            boundaries_sec = vad.get_speech_segments(
                temp_path,
                large_chunk_size=large_chunk_size,
                small_chunk_size=small_chunk_size,
                activation_th=threshold,
                deactivation_th=neg_threshold,
                double_check=double_check,
            )

        # Convert flat tensor [N,2] → list of (start_sec, end_sec) pairs
        boundaries_sec = boundaries_sec.view(-1).tolist()
        speech_pairs = list(zip(boundaries_sec[::2], boundaries_sec[1::2]))

        # Get frame-level speech probabilities (still needed for avg prob & with_scores)
        prob_tensor = vad.get_speech_prob_file(
            temp_path,
            large_chunk_size=large_chunk_size,
            small_chunk_size=small_chunk_size,
        )
        probs = prob_tensor.squeeze().cpu().tolist()  # list of float

        # Approximate frame hop (SpeechBrain CRDNN typically ~10 ms)
        hop_samples = 160
        hop_sec = hop_samples / sr

        def make_segment(
            num: int,
            start_sec: float,
            end_sec: float,
            seg_type: Literal["speech", "non-speech"],
        ) -> SpeechSegment:
            start_sample = int(round(start_sec * sr))
            end_sample = int(round(end_sec * sr))

            frame_start = int(start_sec / hop_sec)
            frame_end = int(end_sec / hop_sec)
            segment_probs_slice = probs[frame_start : frame_end + 1]

            avg_prob = np.mean(segment_probs_slice) if segment_probs_slice else 0.0

            duration_sec = end_sec - start_sec

            start_val = (
                round(start_sec, time_resolution) if return_seconds else start_sample
            )
            end_val = round(end_sec, time_resolution) if return_seconds else end_sample

            return SpeechSegment(
                num=num,
                start=start_val,
                end=end_val,
                prob=round(avg_prob, 4),
                duration=round(duration_sec, 3),
                frames_length=len(segment_probs_slice),
                frame_start=frame_start,
                frame_end=frame_end,
                type=seg_type,
                segment_probs=segment_probs_slice if with_scores else [],
            )

        enhanced: List[SpeechSegment] = []
        current_time = 0.0
        seg_num = 1

        if include_non_speech and speech_pairs:
            first_start = speech_pairs[0][0]
            if first_start > 0.001:
                enhanced.append(make_segment(seg_num, 0.0, first_start, "non-speech"))
                seg_num += 1
            current_time = first_start

        for start_sec, end_sec in speech_pairs:
            if include_non_speech and (start_sec > current_time + 0.01):
                enhanced.append(
                    make_segment(seg_num, current_time, start_sec, "non-speech")
                )
                seg_num += 1

            enhanced.append(make_segment(seg_num, start_sec, end_sec, "speech"))
            seg_num += 1
            current_time = end_sec

        if include_non_speech:
            total_duration = len(probs) * hop_sec
            if current_time < total_duration - 0.01:
                enhanced.append(
                    make_segment(seg_num, current_time, total_duration, "non-speech")
                )
                seg_num += 1

        if with_scores:
            return enhanced, probs
        return enhanced

    finally:
        os.remove(temp_path)


if __name__ == "__main__":
    audio_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_3_speakers.wav"

    console.print(f"[bold cyan]Processing:[/bold cyan] {Path(audio_file).name}")

    segments = extract_speech_timestamps(
        audio_file,
        threshold=0.3,  # SpeechBrain default activation_th
        neg_threshold=0.1,
        return_seconds=True,
        time_resolution=2,
        normalize_loudness=False,
    )

    console.print(f"\n[bold green]Segments found:[/bold green] {len(segments)}\n")
    for seg in segments:
        console.print(
            f"[yellow][[/yellow] [bold white]{seg.start:.2f}[/bold white] - [bold white]{seg.end:.2f}[/bold white] [yellow]][/yellow] "
            f"duration=[bold magenta]{seg.duration}s[/bold magenta] "
            f"prob=[bold cyan]{seg.prob:.3f}[/bold cyan]"
        )
