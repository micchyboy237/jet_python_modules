import numpy as np
import torch
from jet.audio.helpers.silence import (
    SAMPLE_RATE,
)
from rich.table import Table


def display_segments(speech_ts):
    """Display detected speech segments in a clean Rich table with correct time in seconds."""
    if not speech_ts:
        return

    # Total recorded time approximated by the end of the last speech segment (in samples)
    total_samples = max(seg["end"] for seg in speech_ts)
    recorded_seconds = total_samples

    table = Table(title=f"Speech segments (total ~{recorded_seconds:.1f}s recorded)")

    table.add_column("Segment", style="cyan", justify="right")
    table.add_column("Start (s)", justify="right")
    table.add_column("End (s)", justify="right")
    table.add_column("Duration (s)", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Status", style="green")

    for i, seg in enumerate(speech_ts, 1):
        start_sec = seg["start"]
        end_sec = seg["end"]
        duration_sec = seg["end"] - seg["start"]

        table.add_row(
            str(i),
            f"{start_sec:.2f}",
            f"{end_sec:.2f}",
            f"{duration_sec:.2f}",
            f"{seg.get('prob', seg.get('score', '-')):.2f}",
            "active" if i == len(speech_ts) else "",
        )

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
