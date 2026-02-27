from typing import Literal

import numpy as np
import torch
from jet.audio.utils import get_input_channels
from jet.logger import logger
from scipy.signal import resample

SAMPLE_RATE = 16000
DTYPE = "int16"

CHANNELS = min(2, get_input_channels())


def convert_audio_to_tensor(
    audio_data: np.ndarray | list[np.ndarray], sr: int = 16000
) -> torch.Tensor:
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
    assert sr == 16000, "Wrong sample rate for Silero VAD: must be 16000 Hz"

    return tensor  # shape: (N_samples,), float32, [-1, 1], 16kHz


PCMEncoding = Literal["pcm_s16le"]  # can extend later: "pcm_f32le", "mulaw", etc.


def _float_to_pcm(arr: np.ndarray, target_dtype: np.dtype = DTYPE) -> np.ndarray:
    """
    Convert floating-point audio [-1, 1] (or normalized) to integer PCM.
    Clips and scales safely.
    """
    if not np.issubdtype(arr.dtype, np.floating):
        return arr

    max_abs = np.max(np.abs(arr))
    if max_abs > 1.0001:
        logger.warning("Audio peak > 1.0 — normalizing")
        arr = arr / max_abs if max_abs > 0 else arr

    scaled = np.clip(arr, -1.0, 1.0) * np.iinfo(target_dtype).max
    return np.round(scaled).astype(target_dtype)


def to_raw_pcm_bytes_for_streaming(
    audio: np.ndarray | bytes,
    target_sample_rate: int = 16000,
    target_channels: int = 1,
    target_encoding: PCMEncoding = "pcm_s16le",
    input_sample_rate: int | None = None,
) -> bytes:
    """
    Prepare raw PCM bytes suitable for real-time STT WebSocket streaming.

    Most providers (AssemblyAI, Deepgram, ...) expect:
    - 16000 Hz, mono, signed 16-bit little-endian PCM (no header)

    Handles:
    - float → int16 conversion
    - channel reduction (avg) or expansion (duplicate)
    - resampling (using scipy — reasonable quality for most cases)

    For production/high-quality resampling, consider librosa.resample or torchaudio.
    """
    if target_encoding != "pcm_s16le":
        raise NotImplementedError("Only pcm_s16le supported currently")

    target_dtype = np.int16

    # ── Load / normalize input ─────────────────────────────────────────────
    if isinstance(audio, bytes):
        # Assume input bytes match global DTYPE + CHANNELS
        arr = np.frombuffer(audio, dtype=DTYPE).reshape(-1, CHANNELS)
    elif isinstance(audio, np.ndarray):
        arr = audio
    else:
        raise TypeError("audio must be np.ndarray or raw PCM bytes")

    current_sr = input_sample_rate or SAMPLE_RATE

    # ── Float to PCM ───────────────────────────────────────────────────────
    if np.issubdtype(arr.dtype, np.floating):
        arr = _float_to_pcm(arr, target_dtype=target_dtype)
    elif arr.dtype != target_dtype:
        arr = arr.astype(target_dtype)

    # ── Ensure 2D shape ────────────────────────────────────────────────────
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    current_channels = arr.shape[1]

    # ── Channels conversion ────────────────────────────────────────────────
    if current_channels != target_channels:
        if current_channels == 1 and target_channels > 1:
            arr = np.tile(arr, (1, target_channels))
        elif current_channels > 1 and target_channels == 1:
            arr = np.mean(arr, axis=1, keepdims=True).round().astype(target_dtype)
        else:
            raise ValueError(
                f"Cannot convert {current_channels} → {target_channels} channels"
            )

    # ── Resampling if needed ───────────────────────────────────────────────
    if current_sr != target_sample_rate:
        ratio = target_sample_rate / current_sr
        num_new_frames = int(arr.shape[0] * ratio)
        # Resample each channel separately
        resampled_channels = []
        for ch in range(arr.shape[1]):
            res_ch = resample(arr[:, ch], num_new_frames)
            resampled_channels.append(res_ch)
        arr = np.column_stack(resampled_channels).astype(target_dtype)
        logger.info(f"Resampled {current_sr} Hz → {target_sample_rate} Hz")

    return arr.tobytes()
