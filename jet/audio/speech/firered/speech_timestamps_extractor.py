from pathlib import Path
from typing import List, Literal, Union

import numpy as np
import torch
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
