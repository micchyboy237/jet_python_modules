from __future__ import annotations

import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional, Tuple, TypedDict

import numpy as np
import pyloudnorm as pyln
from jet.audio.audio_types import AudioInput
from jet.audio.helpers.config import HOP_SIZE, SAMPLE_RATE
from jet.audio.speech.vad_loaders import get_global_vad
from jet.audio.utils.loader import load_audio  # ← your exact loader

logger = logging.getLogger(__name__)


MODEL_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/VAD").expanduser().resolve()
)


class LoudnessNormInfo(TypedDict, total=False):
    """Diagnostic information from loudness normalization.

    Note: ``total=False`` makes all keys optional, as some fields
    are only populated in specific normalization paths (e.g.,
    ``speech_weighted_lufs`` is only set when VAD detected speech).
    """

    method: str
    """Normalization method used (always 'firered_lufs')."""

    target_lufs: float
    """Target LUFS level requested."""

    original_lufs: float
    """Integrated loudness of the original audio (dB LUFS)."""

    speech_weighted_lufs: float
    """FireRed VAD-weighted integrated loudness (dB LUFS).
    Only set when speech is detected (probs >= 0.1)."""

    final_lufs: float
    """Integrated loudness after normalization (dB LUFS)."""

    original_peak: float
    """Peak amplitude of original audio (0.0 to 1.0+)."""

    final_peak: float
    """Peak amplitude after normalization (0.0 to max_peak)."""

    applied_gain_db: float
    """Total gain applied in dB (negative = attenuation)."""

    speech_probability_mean: float
    """Mean FireRed VAD speech probability across all frames."""

    fallback_used: bool
    """Whether peak normalization fallback was triggered."""

    skipped_reason: Optional[str]
    """Reason normalization was skipped or fallback used, if any."""

    sr: int
    """Sample rate used for processing (always 16000)."""


def _get_firered_vad():
    vad = get_global_vad()
    return vad


def _numpy_to_temp_wav(audio_np: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    if audio_np.ndim != 1:
        audio_np = np.asarray(audio_np).squeeze()
    audio_int16 = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)

    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    return temp_path


def _speech_probability(
    audio: np.ndarray, sample_rate: int, hop_size: int = HOP_SIZE
) -> np.ndarray:
    vad = _get_firered_vad()
    frame_results, result = vad.detect_full(audio)

    probs = [r.smoothed_prob for r in frame_results]

    if len(probs) == 0:
        return np.zeros_like(audio, dtype=np.float32)

    num_samples = len(audio)
    num_frames = len(probs)
    frame_centers = np.arange(num_frames) * hop_size + hop_size / 2.0

    sample_indices = np.arange(num_samples, dtype=np.float32)
    sample_probs = np.interp(sample_indices, frame_centers, probs)
    return np.clip(sample_probs, 0.0, 1.0).astype(np.float32)


def normalize_speech_loudness_firered(
    audio: AudioInput,
    sr: Optional[int] = None,  # kept for backward compatibility
    original_sr: Optional[int] = None,  # ← required for raw arrays/tensors
    target_lufs: float = -13.0,
    min_lufs_threshold: float = -70.0,
    max_loudness_threshold: float | None = -10.0,
    peak_target: float = 0.99,
    return_dtype=None,
) -> Tuple[np.ndarray, LoudnessNormInfo]:
    """
    Normalize speech audio using FireRedVAD-weighted LUFS.
    Fully integrated with load_audio (including original_sr support).

    Args:
        audio:              Audio input (file path, numpy array, or torch tensor).
        sr:                 Target sample rate (deprecated; always 16000).
        original_sr:        Original sample rate for raw arrays/tensors.
        target_lufs:        Desired loudness level in LUFS.
        min_lufs_threshold: Speech-weighted LUFS below this is treated as silence.
        max_loudness_threshold: Cap target LUFS at this value to prevent
                                over-attenuation of already-loud speech.
        peak_target:        Peak amplitude ceiling (0.0 < peak_target <= 1.0).
        return_dtype:       Output dtype (default: float32).

    Returns:
        y_norm:  Normalized audio array.
        info:    Diagnostic dict with original/final statistics and
                 normalization details. See ``LoudnessNormInfo`` for fields.
    """
    target_sr = 16000  # FireRedVAD requirement
    eps = 1e-8

    audio_np, loaded_sr = load_audio(
        audio,
        sr=target_sr,
        mono=True,
    )

    # Helpful warning for the most common mistake
    if original_sr is None and not isinstance(audio, (str, bytes, os.PathLike)):
        logger.warning(
            "normalize_speech_loudness_firered: received raw np.ndarray/torch.Tensor with "
            "original_sr=None. Assuming input is already at 16 kHz (FireRedVAD requirement). "
            "Pass original_sr=... if your array is at a different rate."
        )

    if sr is not None and sr != target_sr:
        logger.warning(
            f"normalize_speech_loudness_firered: requested sr={sr} but forced to 16 kHz."
        )

    # ------------------------------------------------------------------ #
    # 0. Empty-array guard                                               #
    # ------------------------------------------------------------------ #
    if len(audio_np) == 0:
        info: LoudnessNormInfo = {
            "method": "firered_lufs",
            "target_lufs": target_lufs,
            "original_lufs": float("-inf"),
            "final_lufs": float("-inf"),
            "original_peak": 0.0,
            "final_peak": 0.0,
            "applied_gain_db": 0.0,
            "speech_probability_mean": 0.0,
            "fallback_used": True,
            "skipped_reason": "empty_input",
            "sr": target_sr,
        }
        return audio_np.astype(np.float32), info

    # ------------------------------------------------------------------ #
    # 1. Original statistics                                             #
    # ------------------------------------------------------------------ #
    original_peak = float(np.max(np.abs(audio_np)))
    meter = pyln.Meter(target_sr)

    try:
        original_lufs = meter.integrated_loudness(audio_np)
    except Exception:
        original_lufs = float("-inf")

    # ------------------------------------------------------------------ #
    # 2. FireRed VAD speech probabilities                                #
    # ------------------------------------------------------------------ #
    probs = _speech_probability(audio_np, target_sr)
    speech_prob_mean = float(np.mean(probs))

    # ------------------------------------------------------------------ #
    # 3. Normalization path                                              #
    # ------------------------------------------------------------------ #
    fallback_used = False
    skipped_reason = None
    speech_lufs = float("-inf")
    applied_gain_db = 0.0

    if np.max(probs) < 0.1:
        # Fallback: peak normalization when no speech detected
        fallback_used = True
        skipped_reason = "no_speech_detected"
        peak = np.max(np.abs(audio_np))
        if peak > 1e-8:
            result = audio_np / peak * peak_target
        else:
            result = audio_np.copy()
    else:
        weighted_audio = audio_np * probs
        try:
            speech_lufs = meter.integrated_loudness(weighted_audio)
        except Exception as e:
            logger.debug(f"LUFS failed: {e}. Falling back to peak norm.")
            fallback_used = True
            skipped_reason = f"lufs_measurement_failed: {e}"
            peak = np.max(np.abs(audio_np))
            if peak > 0:
                result = audio_np / peak * peak_target
            else:
                result = audio_np.copy()
        else:
            if speech_lufs <= min_lufs_threshold:
                # Too quiet to normalize safely
                skipped_reason = (
                    f"speech_lufs_below_threshold "
                    f"({speech_lufs:.1f} <= {min_lufs_threshold:.1f})"
                )
                result = audio_np.copy()
            else:
                target = (
                    min(target_lufs, speech_lufs, max_loudness_threshold)
                    if max_loudness_threshold is not None
                    else min(target_lufs, speech_lufs)
                )
                normalized = pyln.normalize.loudness(audio_np, speech_lufs, target)
                peak = np.max(np.abs(normalized))
                if peak > 0:
                    normalized = normalized * (peak_target / peak)
                result = np.clip(normalized, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    # 4. Calculate applied gain                                          #
    # ------------------------------------------------------------------ #
    final_peak = float(np.max(np.abs(result)))

    try:
        final_lufs = meter.integrated_loudness(result)
    except Exception:
        final_lufs = float("-inf")

    if original_lufs > float("-inf") and final_lufs > float("-inf"):
        applied_gain_db = round(final_lufs - original_lufs, 2)
    elif final_peak > 0 and original_peak > 0:
        # Fallback gain estimation from peaks
        applied_gain_db = round(20 * np.log10(final_peak / (original_peak + eps)), 2)
    else:
        applied_gain_db = 0.0

    # ------------------------------------------------------------------ #
    # 5. Build info dict                                                 #
    # ------------------------------------------------------------------ #
    info: LoudnessNormInfo = {
        "method": "firered_lufs",
        "target_lufs": target_lufs,
        "original_lufs": round(original_lufs, 2)
        if original_lufs > float("-inf")
        else float("-inf"),
        "speech_weighted_lufs": round(speech_lufs, 2)
        if speech_lufs > float("-inf")
        else float("-inf"),
        "final_lufs": round(final_lufs, 2)
        if final_lufs > float("-inf")
        else float("-inf"),
        "original_peak": round(original_peak, 4),
        "final_peak": round(final_peak, 4),
        "applied_gain_db": applied_gain_db,
        "speech_probability_mean": round(speech_prob_mean, 4),
        "fallback_used": fallback_used,
        "skipped_reason": skipped_reason,
        "sr": target_sr,
    }

    # ------------------------------------------------------------------ #
    # 6. Respect return dtype                                            #
    # ------------------------------------------------------------------ #
    if return_dtype is None:
        return result.astype(np.float32, copy=False), info
    return _cast_audio_dtype(result, return_dtype), info


def _cast_audio_dtype(audio: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return audio.astype(dtype, copy=False)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scaled = audio * info.max
        return np.clip(scaled, info.min, info.max).astype(dtype)
    raise TypeError(f"Unsupported target dtype: {dtype}")
