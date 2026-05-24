from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, TypedDict

import librosa
import numpy as np
import pyloudnorm as pyln
import torch

logger = logging.getLogger(__name__)

_SILERO_MODEL = None

# --------------------------------------------------------------------------- #
# Type definitions
# --------------------------------------------------------------------------- #

NormalizationMethod = Literal["peak", "rms", "hybrid", "auto"]


class ClippingInfo(TypedDict):
    """Information about audio clipping and over-amplification."""

    is_clipped: bool
    clip_ratio: float
    is_over_amplified: bool
    over_amplification_factor: float
    peak_db: float


class DynamicRangeInfo(TypedDict):
    """Dynamic range and crest factor metrics."""

    crest_factor_db: float
    dynamic_range_db: float
    percentile_range_db: float


class VadNormalizationStats(TypedDict, total=False):
    """Complete normalization statistics for VAD."""

    # Core stats (always present)
    method: str
    original_rms_db: float
    final_rms_db: float
    original_peak: float
    final_peak: float
    applied_gain_db: float
    skipped_reason: Optional[str]
    sr: Optional[int]

    # Input diagnostics (always present)
    input_clipped: bool
    input_over_amplified: bool
    input_clip_ratio: float
    input_over_amplification_factor: float

    # Output quality metrics (always present)
    output_crest_factor_db: float
    output_dynamic_range_db: float
    output_percentile_range_db: float
    vad_suitability: Literal["excellent", "good", "marginal", "poor"]

    # Optional warnings
    warnings: List[str]


@dataclass
class NormalizationResult:
    """Container for normalization result with typed fields."""

    audio: np.ndarray
    stats: VadNormalizationStats

    @property
    def was_skipped(self) -> bool:
        """Check if normalization was skipped."""
        return self.stats.get("skipped_reason") is not None

    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were generated."""
        return len(self.stats.get("warnings", [])) > 0


# --------------------------------------------------------------------------- #
# Silero VAD model loading
# --------------------------------------------------------------------------- #


def _load_silero_vad() -> Tuple[torch.nn.Module, object]:
    """Load or retrieve cached Silero VAD model."""
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _SILERO_MODEL = (model, utils)
    return _SILERO_MODEL


# --------------------------------------------------------------------------- #
# Speech probability computation
# --------------------------------------------------------------------------- #


def _speech_probability(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Compute per-sample speech probability using Silero VAD.

    Silero requires fixed-size frames:
    - 512 samples @ 16kHz
    - 256 samples @ 8kHz

    Args:
        audio: 1D float32 audio array
        sample_rate: Sample rate (8000 or 16000 Hz)

    Returns:
        Array of speech probabilities per sample [0, 1]

    Raises:
        ValueError: If sample_rate is not 8000 or 16000
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

    probs_per_frame: List[float] = []

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


# --------------------------------------------------------------------------- #
# Audio diagnostics
# --------------------------------------------------------------------------- #


def detect_clipping(
    y: np.ndarray,
    threshold: float = 0.999,
    eps: float = 1e-8,
) -> ClippingInfo:
    """
    Detect clipping and over-amplification in audio.

    Args:
        y: Input audio array (float32 recommended)
        threshold: Clipping threshold relative to full scale [0, 1]
        eps: Small constant for numerical stability

    Returns:
        ClippingInfo dict with detection results
    """
    if len(y) == 0:
        return ClippingInfo(
            is_clipped=False,
            clip_ratio=0.0,
            is_over_amplified=False,
            over_amplification_factor=1.0,
            peak_db=-np.inf,
        )

    abs_y = np.abs(y)
    max_val = float(np.max(abs_y))

    # Standard clipping (samples at exactly ±1.0 or within threshold)
    clip_samples = int(np.sum(abs_y >= threshold))
    clip_ratio = float(clip_samples / len(y))

    # Over-amplification (values > 1.0, likely from integer overflow or bad gain staging)
    over_amplified = max_val > 1.0
    over_amplification_factor = float(max_val) if over_amplified else 1.0

    peak_db = float(20 * np.log10(max_val + eps))

    return ClippingInfo(
        is_clipped=clip_ratio > 0.01,  # More than 1% samples clipped
        clip_ratio=round(clip_ratio, 6),
        is_over_amplified=over_amplified,
        over_amplification_factor=round(over_amplification_factor, 2),
        peak_db=round(peak_db, 2),
    )


def estimate_dynamic_range(
    y: np.ndarray,
    sr: int,
    eps: float = 1e-8,
) -> DynamicRangeInfo:
    """
    Estimate dynamic range and crest factor of audio.

    Args:
        y: Input audio array
        sr: Sample rate in Hz
        eps: Small constant for numerical stability

    Returns:
        DynamicRangeInfo dict with dynamic range metrics
    """
    if len(y) == 0:
        return DynamicRangeInfo(
            crest_factor_db=-np.inf,
            dynamic_range_db=-np.inf,
            percentile_range_db=-np.inf,
        )

    # RMS and Peak for crest factor
    rms = float(np.sqrt(np.mean(y**2) + eps))
    peak = float(np.max(np.abs(y)))
    crest_factor_db = float(20 * np.log10(peak / rms)) if rms > eps else 0.0

    # Percentile-based dynamic range (more robust than min/max)
    abs_y = np.abs(y)
    p99 = float(np.percentile(abs_y, 99))
    p1 = float(np.percentile(abs_y, 1))
    percentile_range_db = float(20 * np.log10(p99 / p1)) if p1 > eps else 0.0

    # Estimate actual dynamic range (peak - noise floor)
    # Use bottom 10% of non-silent frames as noise floor estimate
    frame_length = int(sr * 0.025)  # 25ms frames
    if frame_length > 0 and len(y) >= frame_length:
        n_frames = len(y) // frame_length
        rms_per_frame = np.array(
            [
                np.sqrt(np.mean(y[i * frame_length : (i + 1) * frame_length] ** 2))
                for i in range(n_frames)
            ]
        )
        if len(rms_per_frame) > 0:
            rms_sorted = np.sort(rms_per_frame)
            noise_floor_idx = max(1, int(n_frames * 0.1))
            noise_floor = float(np.mean(rms_sorted[:noise_floor_idx]))
            signal_peak_rms = float(np.max(rms_per_frame))
            dynamic_range_db = (
                float(20 * np.log10(signal_peak_rms / noise_floor))
                if noise_floor > eps
                else 0.0
            )
        else:
            dynamic_range_db = crest_factor_db
    else:
        dynamic_range_db = crest_factor_db

    return DynamicRangeInfo(
        crest_factor_db=round(crest_factor_db, 2),
        dynamic_range_db=round(dynamic_range_db, 2),
        percentile_range_db=round(percentile_range_db, 2),
    )


def assess_vad_suitability(
    final_rms_db: float,
    crest_factor_db: float,
    is_clipped: bool,
    dynamic_range_db: float,
) -> Literal["excellent", "good", "marginal", "poor"]:
    """
    Assess how suitable the normalized audio is for VAD.

    Args:
        final_rms_db: Final RMS level in dBFS
        crest_factor_db: Crest factor in dB
        is_clipped: Whether output is clipped
        dynamic_range_db: Dynamic range in dB

    Returns:
        Suitability rating string
    """
    if is_clipped:
        return "poor"

    # Ideal VAD levels: -20 to -15 dBFS with moderate dynamics
    if -25 <= final_rms_db <= -15 and 10 <= crest_factor_db <= 25:
        return "excellent"
    elif -30 <= final_rms_db <= -10 and 5 <= crest_factor_db <= 35:
        return "good"
    elif -40 <= final_rms_db <= -5:
        return "marginal"
    else:
        return "poor"


# --------------------------------------------------------------------------- #
# Audio type casting
# --------------------------------------------------------------------------- #


def _cast_audio_dtype(audio: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Cast normalized float audio back to target dtype.
    Integers are scaled from [-1, 1] to full-scale range.

    Args:
        audio: Float audio array in range [-1, 1]
        dtype: Target numpy dtype

    Returns:
        Audio array cast to target dtype

    Raises:
        TypeError: If dtype is not float or integer
    """
    if np.issubdtype(dtype, np.floating):
        return audio.astype(dtype)

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scaled = audio * info.max
        return np.clip(scaled, info.min, info.max).astype(dtype)

    raise TypeError(f"Unsupported audio dtype: {dtype}")


# --------------------------------------------------------------------------- #
# Speech loudness normalization
# --------------------------------------------------------------------------- #


def normalize_speech_loudness(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -13.0,
    min_lufs_threshold: float = -70.0,
    max_loudness_threshold: Optional[float] = -10.0,
    peak_target: float = 0.99,
    return_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Normalize speech audio using speech-probability-weighted LUFS.

    This function uses Silero VAD to identify speech segments, then
    measures loudness only on those segments to avoid silence skewing
    the measurement.

    Args:
        audio: Input audio (1D mono or 2D with trivial channels)
        sample_rate: Sample rate in Hz (8000 or 16000)
        target_lufs: Target loudness in LUFS
        min_lufs_threshold: Below this, audio is returned unchanged
        max_loudness_threshold: Cap on target loudness
        peak_target: Target peak level (0-1)
        return_dtype: Output dtype (defaults to input dtype)

    Returns:
        Loudness-normalized audio array
    """
    # Accept and repair common multichannel input
    if audio.ndim == 2:
        if audio.shape[1] == 1:
            audio = audio[:, 0]  # squeeze trivial stereo
        else:
            # Average channels → simple downmix
            audio = np.mean(audio.astype(np.float64), axis=1).astype(audio.dtype)
    elif audio.ndim > 2:
        raise ValueError(
            f"Unsupported audio shape {audio.shape} — "
            "expected 1D (mono) or 2D (frames, channels)"
        )

    orig_dtype = audio.dtype

    meter = pyln.Meter(sample_rate)

    # 1. Speech probabilities
    probs = _speech_probability(audio, sample_rate)

    if np.max(probs) < 0.1:
        return audio.astype(return_dtype or orig_dtype, copy=True)

    # 2. Weighted audio for LUFS measurement
    weighted_audio = audio * probs

    try:
        speech_lufs = meter.integrated_loudness(weighted_audio)
    except Exception:
        peak = np.max(np.abs(audio))
        if peak == 0:
            result = audio.copy()
        else:
            result = audio / peak * peak_target

        target_dtype = return_dtype or orig_dtype
        return _cast_audio_dtype(result, target_dtype)

    if speech_lufs <= min_lufs_threshold:
        return audio.astype(return_dtype or orig_dtype, copy=True)

    if max_loudness_threshold is not None:
        target_lufs = min(target_lufs, speech_lufs, max_loudness_threshold)

    # 3. Normalize ORIGINAL audio using speech LUFS
    normalized = pyln.normalize.loudness(
        audio,
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
    return _cast_audio_dtype(normalized, target_dtype)


# --------------------------------------------------------------------------- #
# Main VAD normalization function
# --------------------------------------------------------------------------- #


def normalize_audio_for_vad(
    y: np.ndarray,
    sr: Optional[int] = None,
    method: NormalizationMethod = "hybrid",
    target_rms_db: float = -20.0,
    max_peak: float = 0.95,
    eps: float = 1e-8,
    min_signal_db: float = -60.0,
    remove_dc: bool = True,
) -> NormalizationResult:
    """
    Normalize audio specifically for Voice Activity Detection (VAD).

    Recommended for most pipelines: 'hybrid' method with target_rms_db=-20.
    This provides consistent levels for energy-based, WebRTC, Silero, and
    neural VADs.

    Args:
        y: Input audio array (any dtype; converted to float32 internally)
        sr: Sample rate in Hz. Used for diagnostics and future extensions
        method: Normalization strategy:
            'peak'   – scale so the loudest sample hits ±1.0
            'rms'    – scale to target_rms_db; no peak limit
            'hybrid' – RMS target + peak ceiling (recommended)
            'auto'   – intelligently select the best method
        target_rms_db: Desired RMS level in dBFS for 'rms'/'hybrid'
        max_peak: Peak ceiling for 'hybrid' (0 < max_peak ≤ 1.0)
        eps: Small constant to guard log/division of silent frames
        min_signal_db: Signals below this threshold are treated as silent
        remove_dc: If True, subtract the mean before normalizing

    Returns:
        NormalizationResult containing:
        - audio: Normalized float32 audio array
        - stats: Complete diagnostic dict with all metrics
    """
    # ------------------------------------------------------------------ #
    # 0. Empty-array guard
    # ------------------------------------------------------------------ #
    if len(y) == 0:
        empty_stats: VadNormalizationStats = {
            "method": method,
            "original_rms_db": -np.inf,
            "final_rms_db": -np.inf,
            "original_peak": 0.0,
            "final_peak": 0.0,
            "applied_gain_db": 0.0,
            "skipped_reason": "empty_input",
            "sr": sr,
            "input_clipped": False,
            "input_over_amplified": False,
            "input_clip_ratio": 0.0,
            "input_over_amplification_factor": 1.0,
            "output_crest_factor_db": -np.inf,
            "output_dynamic_range_db": -np.inf,
            "output_percentile_range_db": -np.inf,
            "vad_suitability": "poor",
        }
        return NormalizationResult(audio=y.astype(np.float32), stats=empty_stats)

    # ------------------------------------------------------------------ #
    # 1. Convert to float32 and check for clipping
    # ------------------------------------------------------------------ #
    y_norm = y.astype(np.float32).copy()

    # Check clipping BEFORE DC removal (DC offset can mask clipping)
    clip_info = detect_clipping(y_norm)

    # ------------------------------------------------------------------ #
    # 2. DC offset removal
    # ------------------------------------------------------------------ #
    if remove_dc:
        y_norm -= np.mean(y_norm)

    # ------------------------------------------------------------------ #
    # 3. Original statistics
    # ------------------------------------------------------------------ #
    original_rms = float(np.sqrt(np.mean(y_norm**2) + eps))
    original_peak = float(np.max(np.abs(y_norm)))
    original_rms_db = (
        float(20 * np.log10(original_rms)) if original_rms > eps else -np.inf
    )

    # Estimate input dynamic range
    input_dr_info = estimate_dynamic_range(y_norm, sr if sr else 16000)

    # ------------------------------------------------------------------ #
    # 4. Silence guard
    # ------------------------------------------------------------------ #
    if original_rms_db < min_signal_db:
        silent_stats: VadNormalizationStats = {
            "method": method,
            "original_rms_db": round(original_rms_db, 2),
            "final_rms_db": round(original_rms_db, 2),
            "original_peak": round(original_peak, 4),
            "final_peak": round(original_peak, 4),
            "applied_gain_db": 0.0,
            "skipped_reason": "silent_input",
            "sr": sr,
            "input_clipped": clip_info["is_clipped"],
            "input_over_amplified": clip_info["is_over_amplified"],
            "input_clip_ratio": clip_info["clip_ratio"],
            "input_over_amplification_factor": clip_info["over_amplification_factor"],
            "output_crest_factor_db": input_dr_info["crest_factor_db"],
            "output_dynamic_range_db": input_dr_info["dynamic_range_db"],
            "output_percentile_range_db": input_dr_info["percentile_range_db"],
            "vad_suitability": "poor",
        }
        return NormalizationResult(audio=y_norm, stats=silent_stats)

    # ------------------------------------------------------------------ #
    # 5. Auto-select method if requested
    # ------------------------------------------------------------------ #
    selected_method: str = method
    warnings: List[str] = []

    if method == "auto":
        selected_method, auto_warnings = _auto_select_method(
            y_norm=y_norm,
            original_rms_db=original_rms_db,
            original_peak=original_peak,
            clip_info=clip_info,
            input_dr_info=input_dr_info,
            target_rms_db=target_rms_db,
        )
        warnings.extend(auto_warnings)

    # ------------------------------------------------------------------ #
    # 6. Handle over-amplified audio
    # ------------------------------------------------------------------ #
    if clip_info["is_over_amplified"]:
        warnings.append(
            f"Input over-amplified by {clip_info['over_amplification_factor']:.1f}x. "
            f"Applying corrective attenuation before normalization."
        )
        y_norm = y_norm / clip_info["over_amplification_factor"]
        # Recalculate original stats after correction
        original_rms = float(np.sqrt(np.mean(y_norm**2) + eps))
        original_peak = float(np.max(np.abs(y_norm)))
        original_rms_db = (
            float(20 * np.log10(original_rms)) if original_rms > eps else -np.inf
        )

    # ------------------------------------------------------------------ #
    # 7. Normalization
    # ------------------------------------------------------------------ #
    if selected_method == "peak":
        # Scale so the loudest sample reaches ±1.0
        y_norm = librosa.util.normalize(y_norm, norm=np.inf)
        final_peak = float(np.max(np.abs(y_norm)))

    elif selected_method in ("rms", "hybrid"):
        target_rms = 10 ** (target_rms_db / 20.0)
        scale = target_rms / (original_rms + eps)
        y_norm *= scale

        # Post-scale peak
        current_peak = float(np.max(np.abs(y_norm)))

        if selected_method == "hybrid" and current_peak > max_peak:
            # current_peak > max_peak > 0, so division is safe
            y_norm *= max_peak / current_peak
            final_peak = max_peak
        else:
            final_peak = current_peak

    else:
        raise ValueError(
            f"Unknown method: '{selected_method}'. "
            "Choose from 'peak', 'rms', 'hybrid', or 'auto'."
        )

    # ------------------------------------------------------------------ #
    # 8. Final statistics
    # ------------------------------------------------------------------ #
    final_rms = float(np.sqrt(np.mean(y_norm**2) + eps))
    final_rms_db = float(20 * np.log10(final_rms)) if final_rms > eps else -np.inf

    output_dr_info = estimate_dynamic_range(y_norm, sr if sr else 16000)

    # Check for output clipping
    output_clip_info = detect_clipping(y_norm)

    # Assess VAD suitability
    vad_suitability = assess_vad_suitability(
        final_rms_db=final_rms_db,
        crest_factor_db=output_dr_info["crest_factor_db"],
        is_clipped=output_clip_info["is_clipped"],
        dynamic_range_db=output_dr_info["dynamic_range_db"],
    )

    # Add warnings for extreme conditions
    applied_gain = final_rms_db - original_rms_db
    if abs(applied_gain) > 40:
        warnings.append(
            f"Extreme gain correction ({applied_gain:.1f} dB) applied. "
            "Check input audio quality and gain staging."
        )
    if output_clip_info["is_clipped"]:
        warnings.append(
            f"Output contains {output_clip_info['clip_ratio']:.2%} clipped samples. "
            "Consider reducing target_rms_db or using a different method."
        )
    if final_peak > 1.0:
        warnings.append(
            f"Output peak ({final_peak:.4f}) exceeds 1.0. This may indicate a bug."
        )

    # Build complete stats
    stats: VadNormalizationStats = {
        "method": selected_method,
        "original_rms_db": round(original_rms_db, 2),
        "final_rms_db": round(final_rms_db, 2),
        "original_peak": round(original_peak, 4),
        "final_peak": round(final_peak, 4),
        "applied_gain_db": round(applied_gain, 2),
        "skipped_reason": None,
        "sr": sr,
        "input_clipped": clip_info["is_clipped"],
        "input_over_amplified": clip_info["is_over_amplified"],
        "input_clip_ratio": clip_info["clip_ratio"],
        "input_over_amplification_factor": clip_info["over_amplification_factor"],
        "output_crest_factor_db": output_dr_info["crest_factor_db"],
        "output_dynamic_range_db": output_dr_info["dynamic_range_db"],
        "output_percentile_range_db": output_dr_info["percentile_range_db"],
        "vad_suitability": vad_suitability,
    }

    if warnings:
        stats["warnings"] = warnings

    return NormalizationResult(audio=y_norm, stats=stats)


# --------------------------------------------------------------------------- #
# Auto method selection
# --------------------------------------------------------------------------- #


def _auto_select_method(
    y_norm: np.ndarray,
    original_rms_db: float,
    original_peak: float,
    clip_info: ClippingInfo,
    input_dr_info: DynamicRangeInfo,
    target_rms_db: float,
) -> Tuple[str, List[str]]:
    """
    Automatically select the best normalization method based on audio characteristics.

    Args:
        y_norm: Float32 audio array (DC removed)
        original_rms_db: Original RMS level in dBFS
        original_peak: Original peak amplitude
        clip_info: Clipping detection results
        input_dr_info: Input dynamic range metrics
        target_rms_db: Target RMS level

    Returns:
        Tuple of (selected_method, warnings_list)
    """
    warnings: List[str] = []

    crest_factor = input_dr_info["crest_factor_db"]

    if clip_info["is_over_amplified"]:
        # Over-amplified: use hybrid to control both level and peaks
        method = "hybrid"
        warnings.append(
            f"Auto-selected 'hybrid' method due to over-amplification "
            f"({clip_info['over_amplification_factor']:.1f}x)"
        )

    elif crest_factor > 30:
        # High dynamic range: use peak to preserve dynamics
        method = "peak"
        warnings.append(
            f"Auto-selected 'peak' method due to high crest factor "
            f"({crest_factor:.1f} dB)"
        )

    elif abs(original_rms_db - target_rms_db) < 10:
        # Already close to target: use RMS for precision
        method = "rms"
        logger.debug(
            f"Audio close to target level ({original_rms_db:.1f} dBFS). "
            "Using RMS normalization."
        )

    else:
        # Default: hybrid for best VAD performance
        method = "hybrid"
        logger.debug(
            f"Using hybrid normalization "
            f"(RMS target {target_rms_db:.1f} + peak ceiling)"
        )

    return method, warnings


# --------------------------------------------------------------------------- #
# Convenience function with full stats
# --------------------------------------------------------------------------- #


def normalize_audio_for_vad_verbose(
    y: np.ndarray,
    sr: Optional[int] = None,
    method: NormalizationMethod = "auto",
    target_rms_db: float = -20.0,
    max_peak: float = 0.95,
    eps: float = 1e-8,
    min_signal_db: float = -60.0,
    remove_dc: bool = True,
) -> NormalizationResult:
    """
    Normalize audio for VAD with automatic method selection and verbose stats.

    This is a convenience wrapper that defaults to 'auto' method selection
    and provides the most detailed diagnostics.

    Args:
        y: Input audio array
        sr: Sample rate in Hz
        method: Normalization strategy (default: 'auto')
        target_rms_db: Target RMS level in dBFS
        max_peak: Maximum peak level
        eps: Small constant for stability
        min_signal_db: Minimum signal level to process
        remove_dc: Remove DC offset before normalization

    Returns:
        NormalizationResult with normalized audio and complete stats
    """
    return normalize_audio_for_vad(
        y=y,
        sr=sr,
        method=method,
        target_rms_db=target_rms_db,
        max_peak=max_peak,
        eps=eps,
        min_signal_db=min_signal_db,
        remove_dc=remove_dc,
    )


# --------------------------------------------------------------------------- #
# Utility: print formatted stats
# --------------------------------------------------------------------------- #


def format_vad_stats(stats: VadNormalizationStats) -> str:
    """
    Format VAD normalization stats as a human-readable string.

    Args:
        stats: Normalization statistics dict

    Returns:
        Formatted multi-line string
    """
    lines: List[str] = []

    lines.append("=" * 60)
    lines.append("VAD NORMALIZATION STATISTICS")
    lines.append("=" * 60)

    # Core stats
    lines.append(f"Method:              {stats['method']}")
    lines.append(f"Sample Rate:         {stats.get('sr', 'N/A')} Hz")
    lines.append(f"Skipped:             {stats.get('skipped_reason') or 'No'}")
    lines.append("")

    # Level stats
    lines.append("LEVELS:")
    lines.append(f"  Original RMS:      {stats['original_rms_db']:>8.2f} dBFS")
    lines.append(f"  Final RMS:         {stats['final_rms_db']:>8.2f} dBFS")
    lines.append(f"  Applied Gain:      {stats['applied_gain_db']:>8.2f} dB")
    lines.append(f"  Original Peak:     {stats['original_peak']:>8.4f}")
    lines.append(f"  Final Peak:        {stats['final_peak']:>8.4f}")
    lines.append("")

    # Input diagnostics
    lines.append("INPUT DIAGNOSTICS:")
    lines.append(f"  Clipped:           {stats['input_clipped']}")
    lines.append(f"  Clip Ratio:        {stats['input_clip_ratio']:.4%}")
    lines.append(f"  Over-amplified:    {stats['input_over_amplified']}")
    if stats["input_over_amplified"]:
        lines.append(
            f"  Over-amp Factor:   {stats['input_over_amplification_factor']:.1f}x"
        )
    lines.append("")

    # Output quality
    lines.append("OUTPUT QUALITY:")
    lines.append(f"  Crest Factor:      {stats['output_crest_factor_db']:>8.2f} dB")
    lines.append(f"  Dynamic Range:     {stats['output_dynamic_range_db']:>8.2f} dB")
    lines.append(f"  Percentile Range:  {stats['output_percentile_range_db']:>8.2f} dB")
    lines.append(f"  VAD Suitability:   {stats['vad_suitability']}")
    lines.append("")

    # Warnings
    if stats.get("warnings"):
        lines.append("WARNINGS:")
        for i, warning in enumerate(stats["warnings"], 1):
            lines.append(f"  {i}. {warning}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Normalize audio for VAD with comprehensive diagnostics."
    )
    parser.add_argument("audio_path", type=str, help="Path to input audio file")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["peak", "rms", "hybrid", "auto"],
        help="Normalization method (default: auto)",
    )
    parser.add_argument(
        "--target-rms-db",
        type=float,
        default=-20.0,
        help="Target RMS level in dBFS (default: -20.0)",
    )
    parser.add_argument(
        "--max-peak",
        type=float,
        default=0.95,
        help="Maximum peak level (default: 0.95)",
    )
    parser.add_argument(
        "--no-dc-remove", action="store_true", help="Skip DC offset removal"
    )

    args = parser.parse_args()

    # Load audio
    y, sr = librosa.load(args.audio_path, sr=None)

    print(f"Loaded: {args.audio_path}")
    print(f"  Samples: {len(y)}")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Duration: {len(y) / sr:.2f}s")
    print()

    # Normalize
    result = normalize_audio_for_vad_verbose(
        y=y,
        sr=sr,
        method=args.method,  # type: ignore
        target_rms_db=args.target_rms_db,
        max_peak=args.max_peak,
        remove_dc=not args.no_dc_remove,
    )

    # Print stats
    print(format_vad_stats(result.stats))

    # Print warnings to stderr if any
    if result.has_warnings:
        import sys

        print("\nWarnings detected:", file=sys.stderr)
        for warning in result.stats.get("warnings", []):
            print(f"  ⚠ {warning}", file=sys.stderr)
