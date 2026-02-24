from typing import Any, Dict, List

import numpy as np
from jet.audio.speech.speechbrain.speech_timestamps_extractor import (
    extract_speech_timestamps,
)
from rich.console import Console

# A "SpeechSegment" is a dictionary with float time keys and optionally a 'type' string.
SpeechSegment = Dict[str, Any]

console = Console()

CHUNK_DURATION_SEC = 30.0

def extract_audio_segment(
    audio_buffer: bytearray, start_sec: float, end_sec: float, sample_rate: int = 16000
) -> bytearray:
    """Extracts the raw int16 PCM byte slice for the time range [start_sec, end_sec].

    Defensive clamps + debug output (temporary because tests are failing).
    """
    start_sample = int(round(start_sec * sample_rate))
    end_sample = int(round(end_sec * sample_rate))
    start_byte = max(0, start_sample * 2)
    end_byte = min(len(audio_buffer), end_sample * 2)

    console.print(
        f"[dim]DEBUG extract_audio_segment: {start_sec:.3f}-{end_sec:.3f}s "
        f"→ bytes[{start_byte}:{end_byte}] (buf_len={len(audio_buffer)})[/dim]"
    )
    return audio_buffer[start_byte:end_byte]


def combine_segments(
    audio_buffer: bytearray, segments: List[SpeechSegment]
) -> bytearray:
    """Concatenates audio bytes from multiple segments into a single bytearray."""
    combined = bytearray()
    for seg in segments:
        seg_bytes = extract_audio_segment(audio_buffer, seg["start"], seg["end"])
        combined.extend(seg_bytes)
    return combined


def extract_buffered_segments(
    _audio_buffer: bytearray, is_partial: bool = False
) -> tuple[bytearray, bytearray | None]:
    """Renamed from extract_and_display_buffered_segments.

    Removes display_segments / logging logic.
    Returns (joined_audio_bytes_from_all_but_last_non_speech, trailing_non_speech_bytes_or_None).
    """
    console.print(
        f"[cyan]DEBUG extract_buffered_segments called - buf_len={len(_audio_buffer)}[/cyan]"
    )

    if len(_audio_buffer) == 0:
        return bytearray(), None

    _audio_np = np.frombuffer(_audio_buffer, dtype=np.int16).copy()

    buffer_segments: List[SpeechSegment] = extract_speech_timestamps(
        _audio_np,
        max_speech_duration_sec=CHUNK_DURATION_SEC,
        return_seconds=True,
        time_resolution=3,
        with_scores=False,
        normalize_loudness=False,
        include_non_speech=True,
        double_check=True,
    )

    console.print(f"[cyan]DEBUG: VAD returned {len(buffer_segments)} segments[/cyan]")
    for i, seg in enumerate(buffer_segments):
        typ = seg.get("type") if isinstance(seg, dict) else getattr(seg, "type", None)
        console.print(
            f"[dim]  seg {i}: type={typ!r}  [{seg.get('start', '?'):.3f}-{seg.get('end', '?'):.3f}]s[/dim]"
        )

    if not buffer_segments:
        return bytearray(), None

    last_segment = buffer_segments[-1]
    last_type = (
        last_segment.get("type")
        if isinstance(last_segment, dict)
        else getattr(last_segment, "type", None)
    )
    if last_type == "non-speech":
        segments_to_combine = buffer_segments[:-1]
        trailing_buffer = extract_audio_segment(
            _audio_buffer, last_segment["start"], last_segment["end"]
        )
        console.print(
            f"[yellow]DEBUG: last=non-speech → trailing len={len(trailing_buffer)}[/yellow]"
        )
    else:
        segments_to_combine = buffer_segments
        trailing_buffer = None
        console.print("[dim]DEBUG: ends with speech → no trailing buffer[/dim]")

    combined_buffer = combine_segments(_audio_buffer, segments_to_combine)
    console.print(
        f"[green]DEBUG: FINAL combined={len(combined_buffer)} bytes, trailing={len(trailing_buffer) if trailing_buffer else None}[/green]"
    )

    return combined_buffer, trailing_buffer
