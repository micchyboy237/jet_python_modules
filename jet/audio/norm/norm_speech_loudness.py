from __future__ import annotations

import io
import logging
import os
from typing import Optional, Tuple

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from jet.audio.audio_types import AudioInput

logger = logging.getLogger(__name__)

_SILERO_MODEL = None


def _load_silero_vad():
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _SILERO_MODEL = (model, utils)
    return _SILERO_MODEL


def _load_audio_input(
    audio_input: AudioInput,
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Load and normalize any AudioInput to float32 mono [-1,1] numpy array.
    Returns (audio, sample_rate) — sample_rate is None for pure array inputs.
    """
    if isinstance(audio_input, (str, os.PathLike)):
        data, sr = sf.read(str(audio_input), always_2d=True, dtype="float32")
        return data, int(sr)

    elif isinstance(audio_input, bytes):
        with io.BytesIO(audio_input) as buf:
            data, sr = sf.read(buf, always_2d=True, dtype="float32")
        return data, int(sr)

    elif isinstance(audio_input, torch.Tensor):
        return audio_input.numpy(), None

    elif isinstance(audio_input, np.ndarray):
        return audio_input, None

    else:
        raise TypeError(f"Unsupported AudioInput type: {type(audio_input).__name__}")


def _speech_probability(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Compute per-sample speech probability using Silero VAD.

    Silero requires fixed-size frames:
    - 512 samples @ 16kHz
    - 256 samples @ 8kHz
    """
    if sample_rate not in (8000, 16000):
        raise ValueError(
            f"Unsupported sample_rate={sample_rate}. "
            "Silero VAD supports only 8000 or 16000 Hz."
        )

    model, utils = _load_silero_vad()
    frame_size = 512 if sample_rate == 16000 else 256

    audio_tensor = torch.from_numpy(audio).float()

    num_samples = audio_tensor.shape[0]
    num_frames = int(np.ceil(num_samples / frame_size))

    # Pad to full frames
    padded_len = num_frames * frame_size
    if padded_len > num_samples:
        pad = padded_len - num_samples
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad))

    probs_per_frame = []

    with torch.no_grad():
        for i in range(num_frames):
            frame = audio_tensor[i * frame_size : (i + 1) * frame_size]
            frame = frame.unsqueeze(0)  # shape: (1, frame_size)
            prob = model(frame, sample_rate)
            probs_per_frame.append(prob.item())

    frame_probs = np.array(probs_per_frame, dtype=np.float32)

    # Upsample frame probabilities to sample-level
    sample_probs = np.repeat(frame_probs, frame_size)
    sample_probs = sample_probs[:num_samples]

    return sample_probs


def normalize_speech_loudness(
    audio: AudioInput,
    sample_rate: Optional[int] = None,
    target_lufs: float = -13.0,
    min_lufs_threshold: float = -70.0,
    max_loudness_threshold: float | None = -10.0,
    peak_target: float = 0.99,
    return_dtype=None,
) -> np.ndarray:
    """
    Normalize speech audio using speech-probability-weighted LUFS.
    Now accepts broader AudioInput types (files, bytes, torch, arrays).

    Returns
    -------
    np.ndarray
        Normalized audio in [-1, 1] range.
        dtype is np.float32 by default, or the requested return_dtype if provided.
    """

    # ── Load / convert input ───────────────────────────────────────
    audio_np, loaded_sr = _load_audio_input(audio)

    # Determine final sample rate
    if loaded_sr is not None:
        if sample_rate is not None and sample_rate != loaded_sr:
            raise ValueError(
                f"Provided sample_rate={sample_rate} does not match loaded rate {loaded_sr} "
                f"from {type(audio).__name__} input"
            )
        final_sr = loaded_sr
    else:
        # Array-like input (np.ndarray or torch.Tensor)
        if sample_rate is None:
            if isinstance(audio, torch.Tensor):
                raise NotImplementedError(
                    "torch.Tensor input requires explicit sample_rate argument. "
                    "Example: normalize_speech_loudness(audio_tensor, sample_rate=16000, ...)"
                )
            else:
                raise ValueError(
                    "When passing numpy array directly, sample_rate must be provided"
                )
        final_sr = sample_rate

    orig_dtype = audio_np.dtype
    meter = pyln.Meter(final_sr)

    probs = _speech_probability(audio_np, final_sr)

    if np.max(probs) < 0.1:
        result = audio_np.copy()
    else:
        weighted_audio = audio_np * probs

        try:
            speech_lufs = meter.integrated_loudness(weighted_audio)
        except Exception:
            peak = np.max(np.abs(audio_np))
            if peak == 0:
                result = audio_np.copy()
            else:
                result = audio_np / peak * peak_target
        else:
            if speech_lufs <= min_lufs_threshold:
                result = audio_np.copy()
            else:
                target = target_lufs

                if max_loudness_threshold is not None:
                    target = min(target, speech_lufs, max_loudness_threshold)

                normalized = pyln.normalize.loudness(
                    audio_np,
                    speech_lufs,
                    target,
                )

                peak = np.max(np.abs(normalized))
                if peak > 0:
                    normalized *= peak_target / peak

                result = np.clip(normalized, -1.0, 1.0)

    # ── Final dtype handling ───────────────────────────────────────
    if return_dtype is None:
        # Most common / expected case: return float32 [-1,1]
        return result.astype(np.float32, copy=False)

    # User explicitly requested a dtype → respect it
    return _cast_audio_dtype(result, return_dtype)


def _cast_audio_dtype(audio: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Cast normalized float audio back to target dtype.
    Integers are scaled from [-1, 1] to full-scale range.
    """
    if np.issubdtype(dtype, np.floating):
        return audio.astype(dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scaled = audio * info.max
        return np.clip(scaled, info.min, info.max).astype(dtype)

    raise TypeError(f"Unsupported audio dtype: {dtype}")
