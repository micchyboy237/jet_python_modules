# audio_duration.py

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from jet.audio.audio_types import AudioInput

try:
    import torch
except ImportError:
    torch = None


def get_audio_duration(
    audio: AudioInput,
    sample_rate: Optional[int] = None,
) -> float:
    """
    Return duration of audio input in seconds.

    Supports:
        - file path (str | PathLike)
        - raw bytes
        - numpy array (requires sample_rate)
        - torch tensor (requires sample_rate)

    Args:
        audio: Audio input
        sample_rate: Required if audio is array/tensor

    Returns:
        Duration in seconds (float)

    Raises:
        FileNotFoundError
        ValueError
        TypeError
    """

    # ─────────────────────────────
    # Case 1: File path
    # ─────────────────────────────
    if isinstance(audio, (str, os.PathLike)):
        path = Path(audio)

        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            return float(librosa.get_duration(path=str(path)))
        except Exception as e:
            raise ValueError(f"Failed to read audio file {path}: {e}") from e

    # ─────────────────────────────
    # Case 2: Raw bytes
    # ─────────────────────────────
    if isinstance(audio, bytes):
        try:
            buffer = io.BytesIO(audio)
            y, sr = librosa.load(buffer, sr=None, mono=False)
            return float(librosa.get_duration(y=y, sr=sr))
        except Exception as e:
            raise ValueError(f"Failed to decode audio bytes: {e}") from e

    # ─────────────────────────────
    # Case 3: numpy array
    # ─────────────────────────────
    if isinstance(audio, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate is required for numpy array input")
        return float(librosa.get_duration(y=audio, sr=sample_rate))

    # ─────────────────────────────
    # Case 4: torch tensor
    # ─────────────────────────────
    if torch is not None and isinstance(audio, torch.Tensor):
        if sample_rate is None:
            raise ValueError("sample_rate is required for torch.Tensor input")

        y = audio.detach().cpu().numpy()
        return float(librosa.get_duration(y=y, sr=sample_rate))

    raise TypeError(f"Unsupported audio input type: {type(audio)}")


def seconds_to_hms(total_seconds: float) -> str:
    """
    Convert seconds → human-readable format.
    """
    if total_seconds <= 0:
        return "0:00"

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"

    return f"{minutes}:{seconds:02d}"


def get_audio_duration_friendly(
    audio: AudioInput,
    sample_rate: Optional[int] = None,
) -> str:
    """
    Return formatted duration string.
    """
    seconds = get_audio_duration(audio, sample_rate=sample_rate)
    return seconds_to_hms(seconds)
