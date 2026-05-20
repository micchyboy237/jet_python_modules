from typing import Literal

import numpy as np
import torch
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.helpers.config import SAMPLE_RATE
from jet.audio.helpers.silence import (
    SAMPLE_RATE,
)
from rich.table import Table


def display_segments(
    speech_segs: list[SpeechSegment],
    done: bool = False,
    include_speech_type: bool = False,
    time_format: Literal["seconds", "ms", "samples"] = "seconds",
):
    """Display detected speech segments in a clean Rich table.

    Args:
        speech_segs: List of speech segments.
        done: Whether all segments are finalized (highlights current/last).
        include_speech_type: Whether to include a column for segment type.
        time_format: Unit of start/end values in the segments.
                     "seconds"  — already in seconds (float)
                     "ms"       — milliseconds; divided by 1000 for display
                     "samples"  — raw sample indices; divided by SAMPLE_RATE
    """
    if not speech_segs:
        return

    segs_to_display = speech_segs[:-1] if done else speech_segs

    if not segs_to_display:
        return

    def to_seconds(value: float | int) -> float:
        if time_format == "ms":
            return value / 1000.0
        if time_format == "samples":
            return value / SAMPLE_RATE
        return float(value)

    END_REASON_COLORS = {
        "silence": "yellow",
        "valley": "blue",
        "hard_limit": "red",
    }

    total_seconds = to_seconds(max(seg["end"] for seg in segs_to_display))
    table = Table(title=f"Speech segments (total ~{total_seconds:.1f}s recorded)")

    table.add_column("#", style="cyan", justify="right")
    table.add_column("Start (s)", justify="right")
    table.add_column("End (s)", justify="right")
    table.add_column("Dur (s)", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("LastNS", justify="right")
    if include_speech_type:
        table.add_column("Spch", justify="center")
    table.add_column("EndRsn", justify="center")
    table.add_column("Ongng", justify="center")
    table.add_column("Status", style="green")

    for seg in segs_to_display:
        start_s = to_seconds(seg["start"])
        end_s = to_seconds(seg["end"])
        duration_s = end_s - start_s

        prob = seg.get("prob", seg.get("score", "-"))
        try:
            prob_val = f"{prob:.2f}"
        except Exception:
            prob_val = str(prob)

        last_ns = seg.get("last_non_speech_sec")
        last_ns_val = f"{last_ns:.2f}" if last_ns is not None else "-"

        speech_check = "✅" if seg.get("type") == "speech" else "❌"

        is_ongoing = seg.get("is_ongoing", False)
        ongoing_icon = "✅" if (isinstance(is_ongoing, bool) and is_ongoing) else "❌"

        end_reason = seg.get("end_reason")
        if end_reason is None:
            pretty_end_reason = "-"
        else:
            color = END_REASON_COLORS.get(end_reason, "magenta")
            pretty_end_reason = f"[{color}]{end_reason}[/]"

        row = [
            str(seg["num"]),
            f"{start_s:.2f}",
            f"{end_s:.2f}",
            f"{duration_s:.2f}",
            prob_val,
            last_ns_val,
        ]
        if include_speech_type:
            row.append(speech_check)
        row.append(pretty_end_reason)
        row.append(ongoing_icon)

        is_last = seg is segs_to_display[-1]
        row.append("active" if not done and is_last else "")
        table.add_row(*row)

    from rich import print as rprint

    rprint("\n", table, "\n")


def convert_audio_to_tensor(audio_data: np.ndarray | list[np.ndarray]) -> torch.Tensor:
    """
    Convert numpy audio array or list of chunks to torch tensor suitable for Silero VAD.
    - Ensures mono
    - Converts to float32 in range [-1.0, 1.0]
    - Requires 16kHz input!
    """
    # Accept either a single np.ndarray or a list of chunks
    if isinstance(audio_data, list):
        audio = np.concatenate(audio_data, axis=0)
    else:
        audio = np.asarray(audio_data)

    # Normalize integer PCM to float32 in [-1, 1]
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    # If already float, ensure [-1, 1]
    elif np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
    else:
        raise ValueError("Unsupported audio dtype")

    tensor = torch.from_numpy(audio)

    # Convert to mono if multi-channel (average channels)
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=1)

    # Sanity checks
    assert tensor.abs().max() <= 1.0 + 1e-5, "Audio not normalized!"
    assert SAMPLE_RATE == 16000, "Wrong sample rate for Silero VAD: must be 16000 Hz"

    return tensor  # shape: (N_samples,), float32, [-1, 1], 16kHz
