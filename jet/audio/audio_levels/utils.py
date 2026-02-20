# audio_levels.py

from pathlib import Path
from typing import Tuple

import numpy as np
from jet.audio.audio_types import AudioInput
from scipy.io import wavfile

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


def load_audio_mono_float32(
    source: AudioInput, *, target_sr: int | None = None, normalize: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Unified audio loader supporting many input types.
    Returns (samples, sample_rate)
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        sample_rate, data = wavfile.read(path)
    elif isinstance(source, np.ndarray):
        data = source
        sample_rate = 44100
    elif HAS_TORCH and isinstance(source, torch.Tensor):
        data = source.numpy()
        sample_rate = 44100
    elif isinstance(source, bytes):
        from io import BytesIO

        sample_rate, data = wavfile.read(BytesIO(source))
    else:
        raise TypeError(f"Unsupported audio input type: {type(source)}")

    if data.ndim == 2:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    if len(data) == 0:
        return data, sample_rate

    # Normalize integer PCM formats to [-1.0, 1.0]
    if np.issubdtype(data.dtype, np.integer):
        if data.dtype not in (np.int16, np.int32):
            raise ValueError(
                f"Unsupported integer audio dtype: {data.dtype}. "
                "Only np.int16 and np.int32 are supported for automatic normalization."
            )
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    elif data.dtype == np.float64:
        data = data.astype(np.float32)

    if normalize and len(data) > 0:
        max_abs = np.max(np.abs(data))
        if max_abs > 0:
            data /= max_abs

    return data, sample_rate


def calculate_rms(samples: np.ndarray) -> float:
    """
    Calculate RMS value of audio samples (float32 [-1,1])
    Returns value in linear scale (0.0 → 1.0)
    """
    if len(samples) == 0:
        return 0.0

    # RMS = sqrt(mean(x²))
    return float(np.sqrt(np.mean(samples**2)))


def rms_to_dbfs(rms: float) -> float:
    """
    Convert RMS value to decibels.

    Returns
    """
    if rms < 0:
        raise ValueError("RMS value cannot be negative")
    if rms == 0:
        return float("-inf")
    # dBFS relative to full-scale sine (RMS=1/sqrt(2) -> 0 dBFS)
    return 20.0 * np.log10(rms * np.sqrt(2))


def get_audio_levels(
    audio: AudioInput, *, normalize: bool = False, **load_kwargs
) -> dict:
    """
    Most commonly used convenience function

    Returns:
    {
        "rms_linear": float,        # 0.0 to ~1.0
        "dbfs": float,              # -inf to 0.0
        "sample_rate": int,
        "duration_seconds": float
    }
    """
    samples, sr = load_audio_mono_float32(audio, normalize=normalize, **load_kwargs)

    rms = calculate_rms(samples)

    return {
        "rms_linear": rms,
        "dbfs": rms_to_dbfs(rms),
        "sample_rate": sr,
        "duration_seconds": len(samples) / sr,
    }


def has_sound(
    source: AudioInput,
    threshold_db: float = -60.0,
    min_duration_sec: float = 0.02,
    normalize: bool = False,
    **load_options,
) -> bool:
    """
    Check if audio contains meaningful sound.

    This is a high-level convenience function that accepts the same flexible
    AudioInput types as load_audio_mono_float32 and get_audio_levels.

    Args:
        source: AudioInput - path, bytes, numpy array, torch tensor, etc.
        threshold_db: RMS must be louder than this value to be considered sound
        min_duration_sec: Minimum length for signal to be considered meaningful
        normalize: If True, normalization to 1.0 peak is applied for detection
        **load_options: Passed directly to load_audio_mono_float32 (target_sr, ...)

    Returns:
        bool: True if the audio contains sound above threshold for sufficient duration
    """
    samples, sample_rate = load_audio_mono_float32(
        source, normalize=normalize, **load_options
    )

    if len(samples) == 0:
        return False

    duration_sec = len(samples) / sample_rate
    if duration_sec < min_duration_sec:
        return False

    rms = calculate_rms(samples)
    if rms <= 0:
        return False

    rms_db = rms_to_dbfs(rms)

    return bool(rms_db > threshold_db)
