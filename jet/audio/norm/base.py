# preprocessors.py

"""
Standalone loudness normalization utility.

Provides a single, reusable function to normalize audio to a target LUFS value
using pyloudnorm (ITU-R BS.1770-4 compliant). Handles caching of meters for
performance, clipping prevention, and graceful fallback for silent/failed cases.
"""

# audio_types.py
from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Literal, Union

import numpy as np
import numpy.typing as npt
import pyloudnorm as pyln
import soundfile as sf
import torch

# Allow flexible input types
AudioInput = Union[
    str,
    bytes,
    os.PathLike,
    npt.NDArray[np.floating | np.integer],
    "torch.Tensor",
]

logger = logging.getLogger(__name__)

# Cache meters per sample rate to avoid expensive recreation
_METER_CACHE: dict[int, pyln.Meter] = {}

PEAK_TARGET_GENERAL = 0.95
PEAK_TARGET_SPEECH = 0.99

VALID_DTYPE_STRINGS = Literal["float32", "float64", "int16", "int32"]


def normalize_loudness(
    audio: AudioInput,
    sample_rate: int,
    target_lufs: float = -14.0,
    min_lufs_threshold: float = -70.0,
    headroom_factor: float = 1.05,
    mode: Literal["general", "speech"] | None = None,
    max_loudness_threshold: float | None = None,
    return_dtype: VALID_DTYPE_STRINGS | np.dtype | None = None,
) -> np.ndarray:
    """
    Normalize audio to a target integrated loudness (LUFS).

    Args:
        audio: Input audio – can be:
               - File path (str or os.PathLike)
               - Raw audio bytes
               - NumPy array (floating or integer dtype)
               - torch.Tensor
               When path/bytes provided, sample_rate is inferred from file.
        sample_rate: Sample rate of the audio in Hz.
        target_lufs: Desired integrated loudness in LUFS (default -14.0).
        min_lufs_threshold: If measured loudness is below this, skip normalization
                            to avoid amplifying pure noise/silence.
        headroom_factor: Multiplier applied after normalization to prevent clipping
                         (default 1.05 → ~0.87 peak).
        mode: Optional preset mode.
              - "speech": Optimized for spoken word – louder target (-13.0 LUFS)
                and minimal headroom (1.0) for maximum clarity.
              - "general" or None: Standard music/streaming settings.
        max_loudness_threshold: Optional upper bound (in LUFS).
            If provided and measured loudness exceeds this value,
            normalization will not amplify (only attenuate if needed).
            Useful to prevent boosting already-loud speech content.
        return_dtype: Desired dtype of returned array.
            - None: preserve input dtype if ndarray, else float32
            - "float32", "float64", np.float32, np.float64
            - "int16", "int32", np.int16, np.int32 → scales [-1,1] → integer range

    Returns:
        Normalized audio array with the same shape as input.

    Raises:
        ImportError: If pyloudnorm is not installed.
        ValueError: If input parameters are invalid.
        RuntimeError: If audio loading fails (for path/bytes inputs).
    """
    audio, sample_rate, original_dtype = _load_audio_input(audio, sample_rate)
    audio = _ensure_audio_shape(audio)

    meter = _get_meter(sample_rate)

    effective_target_lufs, effective_headroom_factor, apply_peak_norm = _resolve_mode(
        mode, target_lufs, headroom_factor
    )

    normalized_audio = _normalize_internal(
        audio=audio,
        meter=meter,
        effective_target_lufs=effective_target_lufs,
        min_lufs_threshold=min_lufs_threshold,
        max_loudness_threshold=max_loudness_threshold,
        effective_headroom_factor=effective_headroom_factor,
        apply_peak_norm=apply_peak_norm,
    )

    return _convert_output_dtype(
        normalized_audio,
        return_dtype=return_dtype,
        original_dtype=original_dtype,
    )


def _load_audio_input(audio: AudioInput, sample_rate: int):
    original_dtype = None
    if isinstance(audio, (str, os.PathLike)):
        audio_path = str(audio)
        loaded_audio, loaded_sr = sf.read(audio_path, always_2d=False)
        sample_rate = loaded_sr
    elif isinstance(audio, bytes):
        with BytesIO(audio) as buf:
            loaded_audio, loaded_sr = sf.read(buf, always_2d=False)
        sample_rate = loaded_sr
    elif isinstance(audio, torch.Tensor):
        loaded_audio = audio.cpu().numpy()
    elif isinstance(audio, np.ndarray):
        loaded_audio = audio
        original_dtype = loaded_audio.dtype
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)}")

    if loaded_audio.dtype != np.float32:
        loaded_audio = np.asarray(loaded_audio, dtype=np.float32)

    return loaded_audio, sample_rate, original_dtype


def _ensure_audio_shape(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[1] > audio.shape[0]:
            logger.warning("Transposing audio: channels-first detected")
            return audio.T
        return audio
    raise ValueError("Audio must be 1D (mono) or 2D (samples, channels)")


def _get_meter(sample_rate: int) -> pyln.Meter:
    return _METER_CACHE.setdefault(sample_rate, pyln.Meter(sample_rate))


def _resolve_mode(mode, target_lufs, headroom_factor):
    effective_target_lufs = target_lufs
    effective_headroom_factor = headroom_factor
    apply_peak_norm = False

    if mode == "speech":
        if target_lufs == -14.0:
            effective_target_lufs = -13.0
        effective_headroom_factor = 1.0
        apply_peak_norm = True
        logger.debug(
            "Speech mode activated: target_lufs=%.1f, headroom_factor=1.0",
            effective_target_lufs,
        )
    elif mode == "general":
        pass
    elif mode is not None:
        raise ValueError(
            f"Invalid mode: {mode!r}. Allowed: 'general', 'speech', or None."
        )
    return effective_target_lufs, effective_headroom_factor, apply_peak_norm


def _normalize_internal(
    audio,
    meter,
    effective_target_lufs,
    min_lufs_threshold,
    max_loudness_threshold,
    effective_headroom_factor,
    apply_peak_norm,
):
    try:
        measured_lufs = meter.integrated_loudness(audio)
        logger.debug(f"Measured integrated loudness: {measured_lufs:.2f} LUFS")
    except Exception as exc:
        # pyloudnorm raises ValueError when audio is shorter than one gating block (~0.4s)
        if "Audio must have length greater than the block size" in str(exc):
            logger.info(
                "Audio too short for reliable LUFS measurement (< ~0.4s). "
                "Falling back to peak normalization."
            )
            peak = np.max(np.abs(audio))
            if peak == 0:
                logger.debug("Silent audio detected in short clip fallback")
                return audio.copy()

            normalized = audio / peak
            normalized /= effective_headroom_factor
            if apply_peak_norm:
                normalized = _apply_peak_boost(normalized, PEAK_TARGET_SPEECH)

            return normalized.astype(np.float32).copy()
        else:
            logger.warning(
                f"Unexpected LUFS measurement failure ({exc}), returning original audio"
            )
            return audio.copy()
    else:
        if measured_lufs <= min_lufs_threshold:
            logger.debug("Audio too quiet – skipping loudness normalization")
            return audio.copy()

        # Optional cap: prevent amplification of already-loud content
        if max_loudness_threshold is not None:
            if measured_lufs > max_loudness_threshold:
                effective_target_lufs = min(effective_target_lufs, measured_lufs)
                logger.debug(
                    "Measured loudness %.2f exceeds max threshold %.2f – "
                    "preventing amplification (effective target: %.2f)",
                    measured_lufs,
                    max_loudness_threshold,
                    effective_target_lufs,
                )

        try:
            normalized = pyln.normalize.loudness(
                audio, measured_lufs, effective_target_lufs
            )
        except Exception:
            logger.warning("LUFS normalization failed, returning original audio")
            return audio.copy()

        normalized = _apply_peak_normalization(
            normalized, effective_headroom_factor, apply_peak_norm
        )

        return normalized.astype(np.float32).copy()


def _apply_peak_normalization(
    audio: np.ndarray, headroom_factor: float, apply_peak_norm: bool
) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / (peak * headroom_factor)
    elif apply_peak_norm:
        audio = _apply_peak_boost(audio, PEAK_TARGET_SPEECH)
    return np.clip(audio, -1.0, 1.0)


def _apply_peak_boost(audio: np.ndarray, target_peak: float) -> np.ndarray:
    current_peak = np.max(np.abs(audio))
    if current_peak == 0 or current_peak >= target_peak:
        return audio
    gain = target_peak / current_peak
    return audio * gain


def _convert_output_dtype(
    audio: np.ndarray, return_dtype, original_dtype
) -> np.ndarray:
    ALLOWED_OUTPUT_TYPES = {np.float32, np.float64, np.int16, np.int32}

    if return_dtype is not None:
        target_dtype = np.dtype(return_dtype)
        target_type = target_dtype.type

        if target_type not in ALLOWED_OUTPUT_TYPES:
            raise ValueError(
                f"Unsupported return_dtype: {return_dtype!r}. "
                "Supported: 'float32', 'float64', 'int16', 'int32' (or their np.dtype equivalents)."
            )

        if target_type is np.float64:
            audio = audio.astype(np.float64)
        elif target_type is np.int16:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32767.0).astype(np.int16)
        elif target_type is np.int32:
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 2147483647.0).astype(np.int32)
        # else: np.float32 is default
    else:
        if original_dtype is not None:
            if np.issubdtype(original_dtype, np.floating):
                audio = audio.astype(original_dtype)
            elif np.issubdtype(original_dtype, np.integer):
                if original_dtype == np.int16:
                    audio = np.clip(audio, -1.0, 1.0)
                    audio = (audio * 32767.0).astype(np.int16)
                elif original_dtype == np.int32:
                    audio = np.clip(audio, -1.0, 1.0)
                    audio = (audio * 2147483647.0).astype(np.int32)
                else:
                    audio = audio.astype(np.float32)
            else:
                audio = audio.astype(np.float32)
        else:
            audio = audio.astype(np.float32)

    return audio


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    import soundfile as sf
    from rich import print as rprint

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    # shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    DEFAULT_INPUT_AUDIO = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/audio/generated/run_record_mic/recording_missav_5mins.wav"

    parser = argparse.ArgumentParser(
        description="Normalize loudness of a WAV file to target LUFS (ITU-R BS.1770-4)."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",  # Make positional input optional
        default=DEFAULT_INPUT_AUDIO,
        help="Input WAV file path",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",  # Make it optional
        default=None,
        help="Output WAV file path (default: <input>_norm.wav)",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=float,
        default=-14.0,
        help="Target integrated loudness in LUFS (default: -14.0)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64", "int16", "int32"],
        default=None,
        help="Output data type: float32, float64, int16, int32. "
        "If not specified, automatically matches the input file's native subtype when possible "
        "(e.g., int16 input → int16 output). Falls back to float32.",
    )

    args = parser.parse_args()

    if args.input is None:
        rprint("[red]Error: input audio path is required[/red]")
        sys.exit(1)

    input_path: Path = args.input
    if not input_path.is_file():
        rprint(f"[red]Error: Input file not found: {input_path}[/red]")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        output_path = OUTPUT_DIR / f"{input_path.stem}_norm{input_path.suffix}"

    rprint(f"[bold]Loading audio:[/bold] {input_path}")
    audio, sr = sf.read(input_path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    # Measure original loudness
    meter = pyln.Meter(sr)
    try:
        original_lufs = meter.integrated_loudness(audio)
    except Exception:
        original_lufs = float("-inf")

    rprint(f"[bold]Normalizing[/bold] to {args.target} LUFS...")
    normalized_audio = normalize_loudness(
        audio,
        sr,
        target_lufs=args.target,
        return_dtype=args.dtype,
        mode="speech",
    )

    # Measure final loudness
    try:
        final_lufs = meter.integrated_loudness(normalized_audio)
    except Exception:
        final_lufs = float("-inf")

    rprint(
        f"[green]Original:[/green] {original_lufs:.2f} LUFS → [green]Normalized:[/green] {final_lufs:.2f} LUFS"
    )
    rprint(f"[bold]Writing output:[/bold] {output_path}")
    sf.write(output_path, normalized_audio, sr)
    rprint("[bold green]Done![/bold green]")
