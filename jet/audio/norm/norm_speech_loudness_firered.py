from __future__ import annotations

import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import pyloudnorm as pyln
import torch
from jet.audio.audio_types import AudioInput
from jet.audio.utils.loader import load_audio  # ← your exact loader

logger = logging.getLogger(__name__)

_FIRE_RED_VAD = None

MODEL_DIR = str(
    Path("~/.cache/pretrained_models/FireRedVAD/VAD").expanduser().resolve()
)


def _get_firered_vad():
    global _FIRE_RED_VAD
    if _FIRE_RED_VAD is None:
        from fireredvad import FireRedVad, FireRedVadConfig

        vad_config = FireRedVadConfig(
            use_gpu=torch.cuda.is_available(),
            smooth_window_size=5,
            speech_threshold=0.4,
            min_speech_frame=20,
            max_speech_frame=2000,
            min_silence_frame=20,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        )
        _FIRE_RED_VAD = FireRedVad.from_pretrained(MODEL_DIR, vad_config)
    return _FIRE_RED_VAD


def _numpy_to_temp_wav(audio_np: np.ndarray, sr: int = 16000) -> str:
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


def _speech_probability(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 16000:
        raise ValueError(f"FireRedVAD requires exactly 16000 Hz. Got {sample_rate} Hz.")
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    temp_wav = _numpy_to_temp_wav(audio, sample_rate)
    try:
        vad = _get_firered_vad()
        _, probs = vad.detect(temp_wav)
    finally:
        if os.path.exists(temp_wav):
            os.unlink(temp_wav)

    if len(probs) == 0:
        return np.zeros_like(audio, dtype=np.float32)

    num_samples = len(audio)
    num_frames = len(probs)
    hop_size = num_samples / num_frames
    frame_centers = np.arange(num_frames) * hop_size + hop_size / 2.0

    sample_indices = np.arange(num_samples, dtype=np.float32)
    sample_probs = np.interp(sample_indices, frame_centers, probs)
    return np.clip(sample_probs, 0.0, 1.0).astype(np.float32)


def normalize_speech_loudness(
    audio: AudioInput,
    sr: Optional[int] = None,  # kept for backward compatibility
    original_sr: Optional[int] = None,  # ← required for raw arrays/tensors
    target_lufs: float = -13.0,
    min_lufs_threshold: float = -70.0,
    max_loudness_threshold: float | None = -10.0,
    peak_target: float = 0.99,
    return_dtype=None,
) -> np.ndarray:
    """
    Normalize speech audio using FireRedVAD-weighted LUFS.
    Fully integrated with load_audio (including original_sr support).
    """
    target_sr = 16000  # FireRedVAD requirement

    audio_np, loaded_sr = load_audio(
        audio,
        sr=target_sr,
        mono=True,
        normalize=True,
        original_sr=original_sr,  # ← critical line
    )

    # Helpful warning for the most common mistake
    if original_sr is None and not isinstance(audio, (str, bytes, os.PathLike)):
        logger.warning(
            "normalize_speech_loudness: received raw np.ndarray/torch.Tensor with "
            "original_sr=None. Assuming input is already at 16 kHz (FireRedVAD requirement). "
            "Pass original_sr=... if your array is at a different rate."
        )

    if sr is not None and sr != target_sr:
        logger.warning(
            f"normalize_speech_loudness: requested sr={sr} but forced to 16 kHz."
        )

    meter = pyln.Meter(target_sr)
    probs = _speech_probability(audio_np, target_sr)

    if np.max(probs) < 0.1:
        peak = np.max(np.abs(audio_np))
        result = audio_np / peak * peak_target if peak > 1e-8 else audio_np.copy()
    else:
        weighted_audio = audio_np * probs
        try:
            speech_lufs = meter.integrated_loudness(weighted_audio)
        except Exception as e:
            logger.debug(f"LUFS failed: {e}. Falling back to peak norm.")
            peak = np.max(np.abs(audio_np))
            result = audio_np / peak * peak_target if peak > 0 else audio_np.copy()
        else:
            if speech_lufs <= min_lufs_threshold:
                result = audio_np.copy()
            else:
                target = (
                    min(target_lufs, max_loudness_threshold)
                    if max_loudness_threshold is not None
                    else target_lufs
                )
                normalized = pyln.normalize.loudness(audio_np, speech_lufs, target)
                peak = np.max(np.abs(normalized))
                if peak > 0:
                    normalized = normalized * (peak_target / peak)
                result = np.clip(normalized, -1.0, 1.0)

    if return_dtype is None:
        return result.astype(np.float32, copy=False)
    return _cast_audio_dtype(result, return_dtype)


def _cast_audio_dtype(audio: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.floating):
        return audio.astype(dtype, copy=False)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scaled = audio * info.max
        return np.clip(scaled, info.min, info.max).astype(dtype)
    raise TypeError(f"Unsupported target dtype: {dtype}")
