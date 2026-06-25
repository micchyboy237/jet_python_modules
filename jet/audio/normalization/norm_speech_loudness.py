from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import pyloudnorm as pyln
import torch
from jet.audio.helpers.energy_base import normalize_energy

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


def normalize_audio_for_vad(
    y: Union[np.ndarray, torch.Tensor],
    sr: Optional[int] = None,
    method: str = "hybrid",
    target_rms_db: float = -20.0,
    max_peak: float = 0.95,
    eps: float = 1e-8,
    min_signal_db: float = -60.0,
    remove_dc: bool = True,
) -> Tuple[Union[np.ndarray, torch.Tensor], dict]:
    """
    Normalize audio specifically for Voice Activity Detection (VAD).

    Uses normalize_energy() for consistent RMS measurement, aligned with
    rms_to_loudness_label() and has_sound() thresholds.

    Args:
        y:               Input audio array (any dtype; converted to float32).
                         Supports numpy.ndarray or torch.Tensor.
        sr:              Sample rate in Hz. Currently used for documentation
                         and future extensions (e.g., resampling, pre-emphasis
                         cutoff). Pass it for forward-compatibility.
        method:          Normalization strategy:
                           'peak'   – scale so the loudest sample hits ±1.0.
                           'rms'    – scale to target_rms_db; no peak limit.
                           'hybrid' – RMS target + peak ceiling (recommended).
        target_rms_db:   Desired RMS level in dBFS for 'rms' / 'hybrid'.
        max_peak:        Peak ceiling for 'hybrid' (0 < max_peak ≤ 1.0).
        eps:             Small constant to guard log/division of silent frames.
        min_signal_db:   Signals whose RMS is below this threshold are treated
                         as silent and returned unchanged (avoids boosting pure
                         noise by 50+ dB).
        remove_dc:       If True, subtract the mean before normalizing.
                         Recommended for energy-based and WebRTC VADs.

    Returns:
        y_norm:  Normalized float32 audio array (same type as input).
        info:    Diagnostic dict with original/final statistics.
    """
    is_torch = isinstance(y, torch.Tensor)

    # Convert to numpy for processing
    if is_torch:
        y_numpy = y.detach().cpu().numpy()
    else:
        y_numpy = y

    if len(y_numpy) == 0:
        empty_info = {
            "method": method,
            "original_rms_db": -np.inf,
            "final_rms_db": -np.inf,
            "original_peak": 0.0,
            "final_peak": 0.0,
            "applied_gain_db": 0.0,
            "skipped_reason": "empty_input",
        }
        if is_torch:
            return torch.tensor([], dtype=torch.float32), empty_info
        return y_numpy.astype(np.float32), empty_info

    y_norm = y_numpy.astype(np.float32).copy()
    if remove_dc:
        y_norm -= np.mean(y_norm)

    original_peak = float(np.max(np.abs(y_norm)))

    # --- Use normalize_energy for consistent RMS measurement ---
    # return_max=True gives us the effective max (the normalization anchor).
    # We pass [rms] as a single-element array so normalize_energy handles
    # the fallback_max / clip logic the same way as the rest of the codebase.
    raw_rms = float(np.sqrt(np.mean(y_norm.astype(np.float64) ** 2) + eps))
    _, effective_max = normalize_energy(
        [raw_rms],
        max_rms=None,  # let it auto-detect from the array
        fallback_max=raw_rms,  # anchor to the signal itself
        clip=False,
        return_max=True,
    )

    original_rms_db = float(20 * np.log10(raw_rms)) if raw_rms > eps else -np.inf

    if original_rms_db < min_signal_db:
        info = {
            "method": method,
            "original_rms_db": round(original_rms_db, 2),
            "final_rms_db": round(original_rms_db, 2),
            "original_peak": round(original_peak, 4),
            "final_peak": round(original_peak, 4),
            "applied_gain_db": 0.0,
            "skipped_reason": "silent_input",
        }
        if is_torch:
            return torch.from_numpy(y_norm), info
        return y_norm, info

    if method == "peak":
        y_norm = librosa.util.normalize(y_norm, norm=np.inf)
        final_peak = float(np.max(np.abs(y_norm)))

    elif method in ("rms", "hybrid"):
        target_rms = 10 ** (target_rms_db / 20.0)
        scale = target_rms / (raw_rms + eps)
        y_norm *= scale
        current_peak = float(np.max(np.abs(y_norm)))
        if method == "hybrid" and current_peak > max_peak:
            y_norm *= max_peak / current_peak
            final_peak = max_peak
        else:
            final_peak = current_peak

    else:
        raise ValueError(
            f"Unknown method: '{method}'. Choose from 'peak', 'rms', or 'hybrid'."
        )

    final_rms = float(np.sqrt(np.mean(y_norm.astype(np.float64) ** 2) + eps))
    final_rms_db = float(20 * np.log10(final_rms))

    info = {
        "method": method,
        "original_rms_db": round(original_rms_db, 2),
        "final_rms_db": round(final_rms_db, 2),
        "original_peak": round(original_peak, 4),
        "final_peak": round(final_peak, 4),
        "applied_gain_db": round(final_rms_db - original_rms_db, 2),
        "skipped_reason": None,
        "sr": sr,
    }

    if is_torch:
        return torch.from_numpy(y_norm), info
    return y_norm, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize audio for VAD.")
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    args = parser.parse_args()

    y, sr = librosa.load(args.audio_path, sr=None)

    y_norm, stats = normalize_audio_for_vad(y, sr=sr)

    print("Normalization applied:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
