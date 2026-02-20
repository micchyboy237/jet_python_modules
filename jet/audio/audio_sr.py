from __future__ import annotations

import os

import librosa
import numpy as np
import torch
from jet.audio.audio_types import AudioInput


def get_audio_sr(audio: AudioInput) -> int:
    """
    Return the sampling rate of the given audio input without loading the full signal.

    Supports:
    - file path (str or PathLike)
    - raw bytes (e.g. from file.read())
    - already loaded numpy array
    - torch Tensor (1D or 2D)

    Raises
    ------
    ValueError
        If the input type is not supported or sample rate cannot be determined
    RuntimeError
        If librosa cannot read the file/bytes
    """
    if isinstance(audio, (str, os.PathLike)):
        try:
            return librosa.get_samplerate(audio)
        except Exception as e:
            raise RuntimeError(f"Failed to read sample rate from file {audio!r}") from e

    elif isinstance(audio, bytes):
        # Write to temporary file-like object or use soundfile directly
        from io import BytesIO

        import soundfile as sf

        try:
            with BytesIO(audio) as buf:
                info = sf.info(buf)
            return info.samplerate
        except Exception as e:
            raise ValueError("Cannot determine sample rate from bytes") from e

    elif isinstance(audio, np.ndarray):
        # We cannot know the true sample rate from array alone
        raise ValueError(
            "Cannot determine sample rate from numpy array alone. "
            "Pass original path/bytes or use a wrapper that keeps sr metadata."
        )

    elif isinstance(audio, torch.Tensor):
        if audio.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D torch Tensor, got shape {audio.shape}")

        # We also cannot know sample rate from tensor alone
        raise ValueError(
            "Cannot determine sample rate from torch.Tensor alone. "
            "Pass original path/bytes or keep sample rate in metadata."
        )

    else:
        raise TypeError(
            f"Unsupported audio input type: {type(audio).__name__}. "
            f"Expected one of: str, PathLike, bytes, np.ndarray, torch.Tensor"
        )
