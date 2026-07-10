from __future__ import annotations

from typing import Union

import numpy as np
import pyloudnorm as pyln
import torch

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


def _speech_probability(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute per-sample speech probability using Silero VAD.

    Silero requires fixed-size frames:
    - 512 samples @ 16kHz
    - 256 samples @ 8kHz

    Returns same type as input (numpy or torch).
    """
    if sample_rate not in (8000, 16000):
        raise ValueError(
            f"Unsupported sample_rate={sample_rate}. "
            "Silero VAD supports only 8000 or 16000 Hz."
        )

    model, utils = _load_silero_vad()
    frame_size = 512 if sample_rate == 16000 else 256

    is_torch = isinstance(audio, torch.Tensor)

    # Convert to torch for inference if needed
    if not is_torch:
        audio_tensor = torch.from_numpy(audio).float()
    else:
        audio_tensor = audio.float().clone()

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

    # Return as torch tensor if input was torch
    if is_torch:
        return torch.from_numpy(sample_probs)
    return sample_probs


def normalize_speech_loudness(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    target_lufs: float = -13.0,
    min_lufs_threshold: float = -70.0,
    max_loudness_threshold: float | None = -10.0,
    peak_target: float = 0.99,
    return_dtype=None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize speech audio using speech-probability-weighted LUFS.
    """
    is_torch = isinstance(audio, torch.Tensor)

    # Convert to numpy for processing
    if is_torch:
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = audio.copy() if isinstance(audio, np.ndarray) else np.array(audio)

    # Accept and repair common multichannel input
    if audio_np.ndim == 2:
        if audio_np.shape[1] == 1:
            audio_np = audio_np[:, 0]  # squeeze trivial stereo
        else:
            # Average channels → simple downmix
            audio_np = np.mean(audio_np.astype(np.float64), axis=1).astype(
                audio_np.dtype
            )
    elif audio_np.ndim > 2:
        raise ValueError(
            f"Unsupported audio shape {audio_np.shape} — "
            "expected 1D (mono) or 2D (frames, channels)"
        )

    orig_dtype = audio_np.dtype

    meter = pyln.Meter(sample_rate)

    # 1. Speech probabilities
    probs = _speech_probability(audio_np, sample_rate)

    # Convert probs to numpy if it's a torch tensor
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy()
    else:
        probs_np = probs

    if np.max(probs_np) < 0.1:
        result = audio_np
        if is_torch:
            return torch.from_numpy(
                result.astype(return_dtype or orig_dtype, copy=True)
            )
        return result.astype(return_dtype or orig_dtype, copy=True)

    # 2. Weighted audio for LUFS measurement
    weighted_audio = audio_np * probs_np

    try:
        speech_lufs = meter.integrated_loudness(weighted_audio)
    except Exception:
        peak = np.max(np.abs(audio_np))
        if peak == 0:
            result = audio_np.copy()
        else:
            result = audio_np / peak * peak_target

        target_dtype = return_dtype or orig_dtype
        result = _cast_audio_dtype(result, target_dtype)
        if is_torch:
            return torch.from_numpy(result)
        return result

    if speech_lufs <= min_lufs_threshold:
        result = audio_np
        target_dtype = return_dtype or orig_dtype
        result = _cast_audio_dtype(result, target_dtype)
        if is_torch:
            return torch.from_numpy(result)
        return result

    if max_loudness_threshold is not None:
        target_lufs = min(target_lufs, speech_lufs, max_loudness_threshold)

    # 3. Normalize ORIGINAL audio using speech LUFS
    normalized = pyln.normalize.loudness(
        audio_np,
        speech_lufs,
        target_lufs,
    )

    # 4. Speech peak normalization (AMPLIFICATION ALLOWED)
    peak = np.max(np.abs(normalized))
    if peak > 0:
        gain = peak_target / peak
        normalized *= gain

    normalized = np.clip(normalized, -1.0, 1.0)

    # 5. Respect return dtype
    target_dtype = return_dtype or orig_dtype
    result = _cast_audio_dtype(normalized, target_dtype)

    if is_torch:
        return torch.from_numpy(result)
    return result


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
