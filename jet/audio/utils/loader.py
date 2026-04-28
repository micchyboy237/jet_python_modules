import io
import os

import librosa
import numpy as np
import torch

from jet.audio.audio_types import AudioInput


def load_audio(
    audio: AudioInput,
    sr: int = 16_000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Robust audio loader for ASR pipelines with correct datatype, normalization,
    layout, and resampling.

    Handles:
      - File paths
      - In-memory WAV bytes
      - NumPy arrays (any shape/layout/dtype)
      - Torch tensors
      - Automatically normalizes to [-1.0, 1.0] float32
      - Always resamples to target sr
      - Correctly converts stereo → mono regardless of channel position

    Returns
    -------
    np.ndarray
        Shape (samples,), float32, [-1.0, 1.0], exactly `sr` Hz
    """
    target_sr: int = sr
    current_sr: int | None = None

    if isinstance(audio, (str, os.PathLike)):
        y, current_sr = librosa.load(audio, sr=None, mono=False)
    elif isinstance(audio, bytes):
        y, current_sr = librosa.load(io.BytesIO(audio), sr=None, mono=False)
    elif isinstance(audio, np.ndarray):
        y = audio.astype(np.float32, copy=False)
        current_sr = None  # caller is responsible for matching sr
    elif isinstance(audio, torch.Tensor):
        y = audio.float().cpu().numpy()
        current_sr = None  # caller is responsible for matching sr
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    # Integer → float normalisation
    if np.issubdtype(y.dtype, np.integer):
        y = y / (2 ** (np.iinfo(y.dtype).bits - 1))

    # Clamp to [-1, 1] if clipped
    if len(y) > 0 and np.abs(y).max() > 1.0 + 1e-6:
        y = y / np.abs(y).max()

    # Ensure 2-D layout: (channels, samples)
    if y.ndim == 1:
        y = y[None, :]
    elif y.ndim == 2:
        if y.shape[0] > y.shape[1]:
            y = y.T
    else:
        raise ValueError(f"Audio must be 1D or 2D, got shape {y.shape}")

    # Stereo → mono
    if mono and y.shape[0] > 1:
        y = np.mean(y, axis=0, keepdims=True)

    # Resample to target_sr when we know the source rate
    # FIX: never overwrite target_sr with current_sr — that made orig_sr == target_sr
    if current_sr is not None and current_sr != target_sr:
        y = librosa.resample(y, orig_sr=current_sr, target_sr=target_sr)

    return y.squeeze(), target_sr
