"""
VAD Segment Scoring Module
Scores voice activity detection segments considering peak height, width, and global distribution.
"""

import json
import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

# Import silence-related functions from energy_base for true audio silence detection
from jet.audio.helpers.config import FRAME_LENGTH_MS, FRAME_SHIFT_MS, SAMPLE_RATE
from jet.audio.helpers.energy_base import (
    has_sound,  # RMS-based silence detection
    trim_silent,  # Operates on raw audio samples
)
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.traceback import install

install(show_locals=True)
console = Console()

# VAD-specific silence threshold for probability values
# This is conceptually parallel to SILENCE_MAX_THRESHOLD but for VAD probabilities
VAD_SILENCE_THRESHOLD = 0.01


class ScoringMethod(Enum):
    """Available scoring methods."""

    BALANCED = "balanced"
    WIDTH_HEAVY = "width_heavy"
    PEAK_HEAVY = "peak_heavy"
    SIMPLE = "simple"


def trim_silent_edges(
    segment_probs: Union[List[float], np.ndarray],
    silence_threshold: float = VAD_SILENCE_THRESHOLD,
    audio_samples: Optional[np.ndarray] = None,
    sample_rate: int = SAMPLE_RATE,
    frame_length_ms: float = FRAME_LENGTH_MS,
    hop_length_ms: float = FRAME_SHIFT_MS,
    return_audio: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Trim leading and trailing silent frames from segment probabilities.

    This function is the VAD-probability equivalent of energy_base.trim_silent().
    While trim_silent() operates on raw audio samples using RMS energy detection,
    this function operates on VAD probability scores (0-1 scale).

    When audio_samples are provided, it will use the actual audio-based silence
    detection from energy_base to determine which frames to trim, providing
    true silence detection rather than just probability thresholding.

    Args:
        segment_probs: List or numpy array of voice activity probabilities per segment
        silence_threshold: Probabilities below this value are considered silence
                          (default: VAD_SILENCE_THRESHOLD = 0.01)
        audio_samples: Optional raw audio samples for true RMS-based silence detection.
                      When provided, uses energy_base.trim_silent() for detection.
        sample_rate: Audio sample rate in Hz (used only with audio_samples)
        frame_length_ms: Frame length for RMS analysis in ms (used only with audio_samples)
        hop_length_ms: Hop length for RMS analysis in ms (used only with audio_samples)
        return_audio: If True and audio_samples provided, returns tuple of (probs, trimmed_audio)

    Returns:
        If return_audio=False: Numpy array of trimmed probabilities
        If return_audio=True and audio provided: Tuple of (trimmed_probs, trimmed_audio)
        If all values are below threshold, returns empty array (or empty tuple if return_audio=True)

    Examples:
        >>> trim_silent_edges([0.001, 0.002, 0.8, 0.9, 0.003, 0.001])
        array([0.8, 0.9])
        >>> trim_silent_edges([0.9, 0.8, 0.7])
        array([0.9, 0.8, 0.7])
        >>> trim_silent_edges([0.001, 0.002, 0.003])
        array([], dtype=float64)
        >>> # Works with numpy arrays too
        >>> trim_silent_edges(np.array([0.001, 0.8, 0.9, 0.001]))
        array([0.8, 0.9])
    """
    if isinstance(segment_probs, np.ndarray):
        probs = segment_probs.astype(float)
    else:
        probs = np.array(segment_probs, dtype=float)

    if len(probs) == 0:
        if return_audio:
            return np.array([], dtype=float), None
        return np.array([], dtype=float)

    trimmed_audio = None

    if audio_samples is not None and len(audio_samples) > 0:
        # Use audio-based trimming
        try:
            trimmed_audio = trim_silent(
                audio_samples,
                sample_rate=sample_rate,
                frame_length_ms=frame_length_ms,
                hop_length_ms=hop_length_ms,
            )
        except Exception as e:
            # Fall back to probability-based trimming if audio trim fails
            console.print(
                f"[yellow]Warning:[/yellow] Audio-based trimming failed ({e}), "
                f"falling back to probability-based trimming"
            )
            trimmed_audio = None

        if trimmed_audio is not None and len(trimmed_audio) > 0:
            # Calculate frame parameters
            frame_length_samples = int(round(frame_length_ms * sample_rate / 1000.0))
            hop_length_samples = int(round(hop_length_ms * sample_rate / 1000.0))

            # Find first and last frames with sound in original audio
            start_sample = 0
            for i in range(
                0, len(audio_samples) - frame_length_samples + 1, hop_length_samples
            ):
                frame = audio_samples[i : i + frame_length_samples]
                if has_sound(frame):
                    start_sample = i
                    break

            end_sample = len(audio_samples)
            for i in range(
                len(audio_samples) - frame_length_samples, -1, -hop_length_samples
            ):
                frame = audio_samples[i : i + frame_length_samples]
                if has_sound(frame):
                    end_sample = i + frame_length_samples
                    break

            # Map audio sample indices to VAD frame indices
            # VAD frames typically have a 1:1 mapping with hop_length
            start_frame = max(0, start_sample // hop_length_samples)
            end_frame = min(
                len(probs), (end_sample + hop_length_samples - 1) // hop_length_samples
            )

            # Ensure we don't exceed array bounds
            start_frame = min(start_frame, len(probs))
            end_frame = min(end_frame, len(probs))

            if start_frame >= end_frame:
                # No speech detected in audio
                if return_audio:
                    return np.array([], dtype=float), np.array(
                        [], dtype=audio_samples.dtype
                    )
                return np.array([], dtype=float)

            trimmed_probs = probs[start_frame:end_frame]

            if return_audio:
                return trimmed_probs, trimmed_audio

            return trimmed_probs
        else:
            # Audio trimming returned empty or failed, fall through to probability-based
            if return_audio:
                return np.array([], dtype=float), np.array(
                    [], dtype=audio_samples.dtype
                )
            return np.array([], dtype=float)

    # Probability-based trimming (original logic)
    above_threshold = probs >= silence_threshold
    valid_indices = np.where(above_threshold)[0]

    if len(valid_indices) == 0:
        if return_audio and audio_samples is not None:
            return np.array([], dtype=float), np.array([], dtype=audio_samples.dtype)
        return np.array([], dtype=float)

    start_idx = valid_indices[0]
    end_idx = valid_indices[-1] + 1

    trimmed_probs = probs[start_idx:end_idx]

    if return_audio and audio_samples is not None:
        hop_length = int(round(hop_length_ms * sample_rate / 1000.0))
        audio_start = start_idx * hop_length
        audio_end = min(end_idx * hop_length, len(audio_samples))
        trimmed_audio = audio_samples[audio_start:audio_end]
        return trimmed_probs, trimmed_audio

    return trimmed_probs


def score_vad_segments(
    segment_probs: Union[List[float], np.ndarray],
    method: str = "balanced",
    trim_edges: bool = True,
    silence_threshold: float = VAD_SILENCE_THRESHOLD,
    audio_samples: Optional[np.ndarray] = None,
    sample_rate: int = SAMPLE_RATE,
    frame_length_ms: float = FRAME_LENGTH_MS,
    hop_length_ms: float = FRAME_SHIFT_MS,
) -> float:
    """
    Score VAD segments considering peak height, width, and global distribution.

    Args:
        segment_probs: List or numpy array of voice activity probabilities per segment
        method: "balanced" (default), "width_heavy", "peak_heavy", or "simple"
        trim_edges: Whether to trim silent edges before scoring (default: True)
        silence_threshold: Threshold for silence detection (default: VAD_SILENCE_THRESHOLD = 0.01)
        audio_samples: Optional raw audio for true RMS-based silence detection
        sample_rate: Audio sample rate (used only with audio_samples)
        frame_length_ms: Frame length for analysis (used only with audio_samples)
        hop_length_ms: Hop length for analysis (used only with audio_samples)

    Returns:
        Score between 0 and 1, higher for sustained high-confidence speech

    Examples:
        >>> score_vad_segments([0.9, 0.9, 0.9, 0.9, 0.1])  # Wide peak
        0.868
        >>> score_vad_segments([0.1, 0.1, 0.95, 0.1, 0.1])  # Narrow peak
        0.559
        >>> score_vad_segments([0.001, 0.9, 0.9, 0.001])  # Trimmed edges
        0.9
    """
    # Convert to numpy array if needed
    if isinstance(segment_probs, np.ndarray):
        probs = segment_probs.astype(float)
    else:
        probs = np.array(segment_probs, dtype=float)

    if len(probs) == 0:
        return 0.0

    # Trim silent edges using appropriate method
    if trim_edges:
        probs = trim_silent_edges(
            probs,
            silence_threshold=silence_threshold,
            audio_samples=audio_samples,
            sample_rate=sample_rate,
            frame_length_ms=frame_length_ms,
            hop_length_ms=hop_length_ms,
        )
        if len(probs) == 0:
            return 0.0

    # Compute scoring components
    max_prob = float(np.max(probs))
    mean_prob = float(np.mean(probs))

    # Width score: mean of top 80% of probabilities
    sorted_probs = np.sort(probs)[::-1]
    top_k = max(1, int(len(sorted_probs) * 0.8))
    width_score = float(np.mean(sorted_probs[:top_k]))

    # Apply scoring formula based on method
    if method == "balanced":
        score = 0.4 * width_score + 0.4 * max_prob + 0.2 * mean_prob
    elif method == "width_heavy":
        score = 0.6 * width_score + 0.3 * max_prob + 0.1 * mean_prob
    elif method == "peak_heavy":
        score = 0.6 * max_prob + 0.3 * width_score + 0.1 * mean_prob
    elif method == "simple":
        score = max_prob * mean_prob
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'balanced', 'width_heavy', 'peak_heavy', or 'simple'"
        )

    return float(np.clip(score, 0.0, 1.0))


def score_sustained_speech(
    segment_probs: Union[List[float], np.ndarray],
    trim_edges: bool = True,
    silence_threshold: float = VAD_SILENCE_THRESHOLD,
    **kwargs,
) -> float:
    """Optimized for detecting sustained speech (rewards width)."""
    return score_vad_segments(
        segment_probs,
        method="width_heavy",
        trim_edges=trim_edges,
        silence_threshold=silence_threshold,
        **kwargs,
    )


def score_balanced_speech(
    segment_probs: Union[List[float], np.ndarray],
    trim_edges: bool = True,
    silence_threshold: float = VAD_SILENCE_THRESHOLD,
    **kwargs,
) -> float:
    """Balanced scoring for general voice activity."""
    return score_vad_segments(
        segment_probs,
        method="balanced",
        trim_edges=trim_edges,
        silence_threshold=silence_threshold,
        **kwargs,
    )


def score_peak_confidence(
    segment_probs: Union[List[float], np.ndarray],
    trim_edges: bool = True,
    silence_threshold: float = VAD_SILENCE_THRESHOLD,
    **kwargs,
) -> float:
    """Emphasizes peak confidence (useful for keyword spotting)."""
    return score_vad_segments(
        segment_probs,
        method="peak_heavy",
        trim_edges=trim_edges,
        silence_threshold=silence_threshold,
        **kwargs,
    )


def load_probs_from_json(file_path: str) -> List[float]:
    """
    Load probabilities array from JSON file.

    Args:
        file_path: Path to JSON file containing array of floats

    Returns:
        List of probabilities

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        ValueError: If data is not a list of floats
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"JSON data must be an array of floats, got {type(data).__name__}"
        )

    probs = []
    for i, val in enumerate(data):
        try:
            prob = float(val)
            if not (0.0 <= prob <= 1.0):
                console.print(
                    f"[yellow]Warning:[/yellow] Value at index {i} ({prob}) outside [0,1], clipping"
                )
                prob = np.clip(prob, 0.0, 1.0)
            probs.append(prob)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value at index {i}: {val} (must be numeric)")

    return probs


def get_score_components(
    segment_probs: Union[List[float], np.ndarray],
    trim_edges: bool = True,
    silence_threshold: float = VAD_SILENCE_THRESHOLD,
    audio_samples: Optional[np.ndarray] = None,
    sample_rate: int = SAMPLE_RATE,
    frame_length_ms: float = FRAME_LENGTH_MS,
    hop_length_ms: float = FRAME_SHIFT_MS,
) -> dict:
    """
    Get detailed scoring components for analysis.

    Args:
        segment_probs: List or numpy array of voice activity probabilities
        trim_edges: Whether to trim silent edges
        silence_threshold: Threshold for silence detection
        audio_samples: Optional raw audio for true RMS-based silence detection
        sample_rate: Audio sample rate (used only with audio_samples)
        frame_length_ms: Frame length for analysis (used only with audio_samples)
        hop_length_ms: Hop length for analysis (used only with audio_samples)

    Returns:
        Dictionary with detailed scoring components
    """
    # Convert to numpy array if needed
    if isinstance(segment_probs, np.ndarray):
        probs_input = segment_probs.astype(float)
    else:
        probs_input = np.array(segment_probs, dtype=float)

    if len(probs_input) == 0:
        return {}

    trim_kwargs = {
        "silence_threshold": silence_threshold,
        "audio_samples": audio_samples,
        "sample_rate": sample_rate,
        "frame_length_ms": frame_length_ms,
        "hop_length_ms": hop_length_ms,
    }

    if trim_edges:
        trimmed_probs = trim_silent_edges(probs_input, **trim_kwargs)
        if len(trimmed_probs) == 0:
            return {
                "num_segments": len(probs_input),
                "trimmed_segments": len(probs_input),
                "all_trimmed": True,
                "balanced_score": 0.0,
                "sustained_score": 0.0,
                "peak_score": 0.0,
            }
        trimmed_count = len(probs_input) - len(trimmed_probs)
    else:
        trimmed_probs = probs_input
        trimmed_count = 0

    probs = trimmed_probs
    sorted_probs = np.sort(probs)[::-1]
    top_k = max(1, int(len(sorted_probs) * 0.8))

    above_06 = np.sum(probs >= 0.6)
    above_08 = np.sum(probs >= 0.8)

    return {
        "num_segments": len(probs_input),
        "trimmed_segments": trimmed_count,
        "remaining_segments": len(trimmed_probs),
        "max_probability": float(np.max(probs)),
        "min_probability": float(np.min(probs)),
        "mean_probability": float(np.mean(probs)),
        "median_probability": float(np.median(probs)),
        "std_deviation": float(np.std(probs)),
        "width_score": float(np.mean(sorted_probs[:top_k])),
        "segments_above_0.6": above_06,
        "segments_above_0.8": above_08,
        "percentile_90": float(np.percentile(probs, 90)),
        "percentile_75": float(np.percentile(probs, 75)),
        "balanced_score": score_balanced_speech(
            probs_input, trim_edges=trim_edges, **trim_kwargs
        ),
        "sustained_score": score_sustained_speech(
            probs_input, trim_edges=trim_edges, **trim_kwargs
        ),
        "peak_score": score_peak_confidence(
            probs_input, trim_edges=trim_edges, **trim_kwargs
        ),
    }


def display_results(probs: Union[List[float], np.ndarray], components: dict) -> None:
    """Display scoring results with rich formatting."""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]VAD Segment Scoring Results[/bold cyan]", border_style="cyan"
        )
    )

    if "trimmed_segments" in components and components["trimmed_segments"] > 0:
        console.print(
            f"[yellow]✂️  Trimmed {components['trimmed_segments']} silent edge frames "
            f"({components['remaining_segments']} remaining)[/yellow]"
        )

    stats_table = Table(title="📊 Probability Statistics", title_style="bold blue")
    stats_table.add_column("Metric", style="cyan", no_wrap=True)
    stats_table.add_column("Value", style="green")
    stats_table.add_column("Interpretation", style="white")

    stats_table.add_row(
        "Number of segments", str(components["num_segments"]), "Total analysis windows"
    )
    if "trimmed_segments" in components:
        stats_table.add_row(
            "Trimmed segments",
            str(components["trimmed_segments"]),
            "Silent edges removed",
        )
    stats_table.add_row(
        "Max probability",
        f"{components['max_probability']:.3f}",
        "Peak confidence"
        if components["max_probability"] > 0.8
        else "Moderate peak"
        if components["max_probability"] > 0.6
        else "Low peak",
    )
    stats_table.add_row(
        "Mean probability",
        f"{components['mean_probability']:.3f}",
        "Overall voice activity",
    )
    stats_table.add_row(
        "Median probability",
        f"{components['median_probability']:.3f}",
        "Typical segment confidence",
    )
    stats_table.add_row(
        "Std deviation",
        f"{components['std_deviation']:.3f}",
        "High = variable" if components["std_deviation"] > 0.3 else "Low = consistent",
    )
    stats_table.add_row(
        "Width score", f"{components['width_score']:.3f}", "Sustained activity measure"
    )
    console.print(stats_table)

    threshold_table = Table(title="🎯 Threshold Analysis", title_style="bold blue")
    threshold_table.add_column("Threshold", style="cyan", no_wrap=True)
    threshold_table.add_column("Segments Above", style="green")
    threshold_table.add_column("Percentage", style="yellow")

    total = components.get("remaining_segments", components["num_segments"])
    above_06_pct = (components["segments_above_0.6"] / total) * 100 if total > 0 else 0
    above_08_pct = (components["segments_above_0.8"] / total) * 100 if total > 0 else 0

    threshold_table.add_row(
        "≥ 0.6 (moderate confidence)",
        str(components["segments_above_0.6"]),
        f"{above_06_pct:.1f}%",
    )
    threshold_table.add_row(
        "≥ 0.8 (high confidence)",
        str(components["segments_above_0.8"]),
        f"{above_08_pct:.1f}%",
    )
    console.print(threshold_table)

    score_table = Table(title="🎯 Scoring Results", title_style="bold blue")
    score_table.add_column("Method", style="cyan", no_wrap=True)
    score_table.add_column("Score", style="green", justify="right")
    score_table.add_column("Rating", style="white")
    score_table.add_column("Best For", style="dim")

    def get_rating(score: float) -> str:
        if score >= 0.85:
            return "🟢 Excellent"
        elif score >= 0.70:
            return "🔵 Good"
        elif score >= 0.55:
            return "🟡 Marginal"
        elif score >= 0.40:
            return "🟠 Poor"
        else:
            return "🔴 Invalid"

    score_table.add_row(
        "Balanced",
        f"{components['balanced_score']:.3f}",
        get_rating(components["balanced_score"]),
        "General VAD",
    )
    score_table.add_row(
        "Width-heavy (Sustained)",
        f"{components['sustained_score']:.3f}",
        get_rating(components["sustained_score"]),
        "Sustained speech",
    )
    score_table.add_row(
        "Peak-heavy",
        f"{components['peak_score']:.3f}",
        get_rating(components["peak_score"]),
        "Keyword spotting",
    )
    console.print(score_table)

    console.print("\n[bold blue]📈 Probability Distribution[/bold blue]")
    max_bar_width = 50
    probs_array = np.array(probs) if not isinstance(probs, np.ndarray) else probs

    for i, prob in enumerate(probs_array[:20]):
        bar_length = int(prob * max_bar_width)
        bar = "█" * bar_length
        color = "green" if prob >= 0.8 else "yellow" if prob >= 0.6 else "red"
        console.print(
            f"  [{color}]{bar:<{max_bar_width}}[/{color}] {prob:.3f} (seg {i})"
        )

    if len(probs_array) > 20:
        console.print(f"  [dim]... and {len(probs_array) - 20} more segments[/dim]")

    highest_score = max(
        components["balanced_score"],
        components["sustained_score"],
        components["peak_score"],
    )

    if highest_score >= 0.85:
        summary_color = "green"
        summary_icon = "✅"
        summary_text = "High-quality voice activity detected"
    elif highest_score >= 0.70:
        summary_color = "blue"
        summary_icon = "📢"
        summary_text = "Good voice activity detected"
    elif highest_score >= 0.55:
        summary_color = "yellow"
        summary_icon = "⚠️"
        summary_text = "Marginal voice activity - may need review"
    else:
        summary_color = "red"
        summary_icon = "❌"
        summary_text = "No significant voice activity detected"

    console.print()
    console.print(
        Panel(
            f"[bold {summary_color}]{summary_icon} {summary_text}[/bold {summary_color}]\n"
            f"Best score: [bold]{highest_score:.3f}[/bold] using {
                max(
                    [
                        (components['balanced_score'], 'balanced'),
                        (components['sustained_score'], 'sustained'),
                        (components['peak_score'], 'peak'),
                    ],
                    key=lambda x: x[0],
                )[1]
            } method",
            border_style=summary_color,
        )
    )


def save_vad_score(
    segment_probs: Union[List[float], np.ndarray],
    seg_dir: Path,
    seg_number: int,
    trim_edges: bool = True,
    silence_threshold: float = VAD_SILENCE_THRESHOLD,
    **kwargs,
) -> Path:
    """
    Save detailed VAD scoring components to a JSON file.

    Args:
        segment_probs: List or numpy array of VAD probability scores per frame
        seg_dir: Directory to save the score file
        seg_number: Segment number for logging
        trim_edges: Whether to trim silent edges
        silence_threshold: Threshold for silence detection
        **kwargs: Additional arguments passed to get_score_components

    Returns:
        Path to the saved vad_score.json file
    """
    score_components = get_score_components(
        segment_probs,
        trim_edges=trim_edges,
        silence_threshold=silence_threshold,
        **kwargs,
    )

    vad_score_data = {
        "segment_number": seg_number,
        "num_frames": len(segment_probs) if hasattr(segment_probs, "__len__") else 0,
        "trim_settings": {
            "enabled": trim_edges,
            "threshold": silence_threshold,
            "using_audio_trim": kwargs.get("audio_samples") is not None,
        },
        "scoring_components": score_components,
        "interpretation": _interpret_vad_score(score_components),
    }

    vad_score_path = seg_dir / "vad_score.json"
    vad_score_path.write_text(
        json.dumps(vad_score_data, indent=2, default=float), encoding="utf-8"
    )
    return vad_score_path


def _interpret_vad_score(components: dict) -> dict:
    """
    Provide human-readable interpretation of VAD scoring components.

    Args:
        components: Dictionary of scoring components from get_score_components

    Returns:
        Dictionary with quality assessment and recommendations
    """
    balanced_score = components.get("balanced_score", 0.0)
    sustained_score = components.get("sustained_score", 0.0)
    peak_score = components.get("peak_score", 0.0)

    if balanced_score >= 0.85:
        quality = "excellent"
        confidence = "high"
    elif balanced_score >= 0.70:
        quality = "good"
        confidence = "moderate"
    elif balanced_score >= 0.55:
        quality = "marginal"
        confidence = "low"
    elif balanced_score >= 0.40:
        quality = "poor"
        confidence = "very_low"
    else:
        quality = "invalid"
        confidence = "none"

    scores = {
        "sustained_speech": sustained_score,
        "keyword_spotting": peak_score,
        "general_vad": balanced_score,
    }
    best_use_case = max(scores, key=scores.get)

    is_sustained = sustained_score > balanced_score
    has_peaks = peak_score > balanced_score

    interpretation = {
        "quality": quality,
        "confidence_level": confidence,
        "best_use_case": best_use_case.replace("_", " "),
        "speech_characteristics": {
            "is_sustained_speech": is_sustained,
            "has_clear_peaks": has_peaks,
            "consistency": "high"
            if components.get("std_deviation", 1.0) < 0.3
            else "variable",
        },
        "recommendations": _get_recommendations(components, quality),
    }
    return interpretation


def _get_recommendations(components: dict, quality: str) -> list[str]:
    """
    Generate actionable recommendations based on VAD scoring.

    Args:
        components: Scoring components dictionary
        quality: Overall quality assessment

    Returns:
        List of recommendation strings
    """
    recommendations = []

    if quality == "excellent":
        recommendations.append("Segment contains clear, sustained speech")
        recommendations.append("Suitable for transcription or speaker recognition")
    elif quality == "good":
        recommendations.append("Speech is clearly present but may have brief gaps")
        recommendations.append("Review segment boundaries for potential trimming")
    elif quality == "marginal":
        recommendations.append("Speech presence is uncertain - may contain noise")
        recommendations.append("Consider adjusting VAD thresholds")
        recommendations.append("Manual review recommended for critical applications")
    elif quality == "poor":
        recommendations.append("Limited voice activity detected")
        recommendations.append("May be noise, music, or very quiet speech")
        recommendations.append("Consider discarding or lowering confidence thresholds")
    else:
        recommendations.append("No significant voice activity detected")
        recommendations.append("Segment likely contains silence or non-speech audio")

    mean_prob = components.get("mean_probability", 0.0)
    std_dev = components.get("std_deviation", 0.0)
    trimmed_segments = components.get("trimmed_segments", 0)

    if trimmed_segments > 0:
        recommendations.append(f"Trimmed {trimmed_segments} silent edge frames")
    if mean_prob < 0.3:
        recommendations.append(
            "Very low mean probability suggests minimal speech content"
        )
    if std_dev > 0.3:
        recommendations.append(
            "High variability indicates inconsistent speech presence"
        )
    if (
        components.get("segments_above_0.8", 0)
        > components.get("remaining_segments", components.get("num_segments", 1)) * 0.5
    ):
        recommendations.append(
            "Strong high-confidence regions present - focus analysis here"
        )

    return recommendations


def save_plots(
    original_probs: np.ndarray,
    trimmed_probs: Optional[np.ndarray],
    output_dir: Path,
    silence_threshold: float,
    all_scores: Optional[dict] = None,
    components: Optional[dict] = None,
) -> list[tuple[str, Path]]:
    """
    Generate and save visualization plots for VAD scoring results.

    Creates up to four plots:
    1. Side-by-side original and trimmed probability plots
    2. Overlay comparison plot (if trimming was performed)
    3. Scores bar chart (if scores are provided)
    4. Distribution histogram (if trimmed probs available)

    Args:
        original_probs: Original probability array before trimming
        trimmed_probs: Trimmed probability array after silence removal
        output_dir: Directory to save plot files
        silence_threshold: Threshold used for silence detection
        all_scores: Dictionary of scoring method names to scores
        components: Optional scoring components dictionary

    Returns:
        List of (file_type, file_path) tuples for saved plot files
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved_files = []

    # Plot 1: Side-by-side original and trimmed probabilities
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Original probabilities
    ax1.plot(original_probs, "b-", linewidth=1, alpha=0.7)
    ax1.fill_between(range(len(original_probs)), 0, original_probs, alpha=0.3)
    ax1.axhline(
        y=silence_threshold,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Silence threshold ({silence_threshold})",
    )
    ax1.set_title(f"Original VAD Probabilities ({len(original_probs)} frames)")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Statistics box for original
    orig_stats = (
        f"Max: {np.max(original_probs):.3f}\n"
        f"Mean: {np.mean(original_probs):.3f}\n"
        f"Min: {np.min(original_probs):.3f}\n"
        f"Std: {np.std(original_probs):.3f}\n"
        f"Median: {np.median(original_probs):.3f}"
    )
    ax1.text(
        0.02,
        0.98,
        orig_stats,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Trimmed probabilities
    if trimmed_probs is not None and len(trimmed_probs) > 0:
        ax2.plot(trimmed_probs, "g-", linewidth=1, alpha=0.7)
        ax2.fill_between(
            range(len(trimmed_probs)), 0, trimmed_probs, alpha=0.3, color="green"
        )
        ax2.axhline(
            y=silence_threshold,
            color="r",
            linestyle="--",
            alpha=0.5,
            label=f"Silence threshold ({silence_threshold})",
        )

        trimmed_count = len(original_probs) - len(trimmed_probs)
        ax2.set_title(
            f"Trimmed VAD Probabilities ({len(trimmed_probs)} frames, {trimmed_count} trimmed)"
        )
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        # Statistics box for trimmed
        trim_stats = (
            f"Max: {np.max(trimmed_probs):.3f}\n"
            f"Mean: {np.mean(trimmed_probs):.3f}\n"
            f"Min: {np.min(trimmed_probs):.3f}\n"
            f"Std: {np.std(trimmed_probs):.3f}\n"
            f"Median: {np.median(trimmed_probs):.3f}"
        )
        ax2.text(
            0.02,
            0.98,
            trim_stats,
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )
    else:
        ax2.text(
            0.5,
            0.5,
            "All frames trimmed\n(no speech detected)",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax2.set_title("Trimmed VAD Probabilities (empty)")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(0, 1)

    plt.tight_layout()
    plot_path = output_dir / "probabilities_plot.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    saved_files.append(("Probabilities plot", plot_path))

    # Plot 2: Overlay comparison (only if trimming was performed and changed anything)
    if (
        trimmed_probs is not None
        and len(trimmed_probs) > 0
        and len(trimmed_probs) < len(original_probs)
    ):
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot original in blue with low alpha
        x_original = np.arange(len(original_probs))
        ax.plot(
            x_original,
            original_probs,
            "b-",
            linewidth=1.5,
            alpha=0.4,
            label=f"Original ({len(original_probs)} frames)",
        )

        # Find where trimmed probs start in the original array
        # The trimmed probs are a contiguous slice from original
        # We need to find the matching region
        above_threshold = original_probs >= silence_threshold
        valid_indices = np.where(above_threshold)[0]

        if len(valid_indices) > 0:
            start_idx = valid_indices[0]
            end_idx = valid_indices[-1] + 1

            # Verify the slice matches trimmed length
            expected_trimmed_len = end_idx - start_idx
            if expected_trimmed_len == len(trimmed_probs):
                # Plot trimmed in green over original at the correct position
                x_trimmed = np.arange(start_idx, end_idx)
                ax.plot(
                    x_trimmed,
                    trimmed_probs,
                    "g-",
                    linewidth=2,
                    alpha=0.9,
                    label=f"Trimmed ({len(trimmed_probs)} frames)",
                )

                # Highlight trimmed leading silence
                if start_idx > 0:
                    ax.axvspan(
                        0,
                        start_idx,
                        alpha=0.2,
                        color="red",
                        label=f"Leading silence ({start_idx} frames)",
                    )

                # Highlight trimmed trailing silence
                trailing_start = end_idx
                trailing_count = len(original_probs) - trailing_start
                if trailing_count > 0:
                    ax.axvspan(
                        trailing_start,
                        len(original_probs),
                        alpha=0.2,
                        color="orange",
                        label=f"Trailing silence ({trailing_count} frames)",
                    )

                # Add boundary lines
                ax.axvline(
                    x=start_idx,
                    color="green",
                    linestyle=":",
                    alpha=0.7,
                    label=f"Speech start (frame {start_idx})",
                )
                ax.axvline(
                    x=end_idx,
                    color="red",
                    linestyle=":",
                    alpha=0.7,
                    label=f"Speech end (frame {end_idx})",
                )
            else:
                # If lengths don't match (audio-based trimming), just overlay at start
                ax.plot(
                    x_original[: len(trimmed_probs)],
                    trimmed_probs,
                    "g-",
                    linewidth=2,
                    alpha=0.9,
                    label=f"Trimmed ({len(trimmed_probs)} frames)",
                )

                if len(trimmed_probs) < len(original_probs):
                    ax.axvspan(
                        len(trimmed_probs),
                        len(original_probs),
                        alpha=0.2,
                        color="orange",
                        label=f"Removed ({len(original_probs) - len(trimmed_probs)} frames)",
                    )

        ax.axhline(
            y=silence_threshold,
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Silence threshold",
        )
        ax.set_title("Original vs Trimmed VAD Probabilities - Trimming Visualization")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        comp_path = output_dir / "probabilities_comparison.png"
        plt.savefig(str(comp_path), dpi=150, bbox_inches="tight")
        plt.close()
        saved_files.append(("Comparison plot", comp_path))

    # Plot 3: Scores bar chart
    if all_scores:
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = list(all_scores.keys())
        scores = list(all_scores.values())

        # Use distinct colors for each method
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
        bars = ax.bar(
            methods, scores, color=colors[: len(methods)], alpha=0.8, edgecolor="white"
        )

        # Add score labels on bars
        for bar, score_val in zip(bars, scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{score_val:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        # Quality threshold lines
        thresholds = {
            0.85: ("Excellent", "green"),
            0.70: ("Good", "blue"),
            0.55: ("Marginal", "orange"),
            0.40: ("Poor", "red"),
        }

        for thresh, (label, color) in thresholds.items():
            ax.axhline(y=thresh, color=color, linestyle="--", alpha=0.4, linewidth=1)
            ax.text(
                len(methods) - 0.5,
                thresh + 0.01,
                f"{label} ({thresh})",
                fontsize=7,
                color=color,
                alpha=0.7,
                ha="right",
            )

        ax.set_title("VAD Scoring Method Comparison")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticklabels(methods, rotation=15)

        plt.tight_layout()
        scores_path = output_dir / "scores_plot.png"
        plt.savefig(str(scores_path), dpi=150, bbox_inches="tight")
        plt.close()
        saved_files.append(("Scores plot", scores_path))

    # Plot 4: Distribution histogram (if trimmed probs available)
    if trimmed_probs is not None and len(trimmed_probs) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))

        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        bin_colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]

        counts = [
            int(np.sum((trimmed_probs >= bins[i]) & (trimmed_probs < bins[i + 1])))
            for i in range(len(bins) - 1)
        ]

        bars = ax.bar(
            bin_labels, counts, color=bin_colors, alpha=0.8, edgecolor="white"
        )

        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max(counts) * 0.01,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        ax.set_title(f"Probability Distribution ({len(trimmed_probs)} frames)")
        ax.set_xlabel("Probability Range")
        ax.set_ylabel("Frame Count")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        hist_path = output_dir / "distribution_plot.png"
        plt.savefig(str(hist_path), dpi=150, bbox_inches="tight")
        plt.close()
        saved_files.append(("Distribution plot", hist_path))

    return saved_files


def main():
    """Main entry point with argument parsing."""
    import argparse

    import soundfile as sf

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

    parser = argparse.ArgumentParser(
        description="Score VAD segments from a JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s probs.json
  %(prog)s probs.json --method width_heavy
  %(prog)s probs.json --components
  %(prog)s probs.json --no-trim
  %(prog)s probs.json --silence-threshold 0.05
  %(prog)s probs.json --audio audio.wav  # Use true RMS-based silence detection
  %(prog)s --test
        """,
    )
    parser.add_argument(
        "probs_path",
        type=str,
        nargs="?",
        help="Path to JSON file containing array of probabilities",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="balanced",
        choices=["balanced", "width_heavy", "peak_heavy", "simple"],
        help="Scoring method (default: balanced)",
    )
    parser.add_argument(
        "--components",
        "-c",
        action="store_true",
        help="Show detailed component analysis",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=Path,
        help=f"output directory (default: '{OUTPUT_DIR}')",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress rich output, print only the score",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable trimming of silent edges",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=VAD_SILENCE_THRESHOLD,
        help=f"Threshold for silence detection in VAD probabilities (default: {VAD_SILENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file for true RMS-based silence detection",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run with built-in test cases"
    )

    args = parser.parse_args()
    trim_edges = not args.no_trim

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio if provided
    audio_samples = None
    original_audio_path = None
    if args.audio:
        try:
            audio_samples, sample_rate = sf.read(args.audio)
            original_audio_path = Path(args.audio)
            if len(audio_samples.shape) > 1:
                audio_samples = np.mean(audio_samples, axis=1)  # Convert to mono
            console.print(
                f"[green]✓ Loaded audio from {args.audio} ({len(audio_samples)} samples)[/green]"
            )
        except ImportError:
            console.print(
                "[red]Error: soundfile library required for audio loading[/red]"
            )
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error loading audio: {e}[/red]")
            sys.exit(1)
    else:
        sample_rate = 22050  # Default

    if args.test:
        console.print("[bold cyan]Running built-in tests...[/bold cyan]\n")
        test_cases = {
            "narrow_high": [0.1, 0.1, 0.95, 0.1, 0.1],
            "wide_high": [0.9, 0.9, 0.9, 0.9, 0.1],
            "wide_moderate": [0.6, 0.6, 0.6, 0.6, 0.2],
            "with_silent_edges": [0.001, 0.002, 0.8, 0.9, 0.85, 0.003, 0.001],
            "all_silent": [0.001, 0.002, 0.003, 0.004],
        }

        results = []
        for name, probs in test_cases.items():
            components = get_score_components(
                probs,
                trim_edges=trim_edges,
                silence_threshold=args.silence_threshold,
                audio_samples=audio_samples,
                sample_rate=sample_rate,
            )
            results.append({"name": name, "probs": probs, "components": components})

            if not args.quiet:
                console.print(f"[bold]{name.upper()}[/bold]: {probs}")
                if "trimmed_segments" in components:
                    console.print(f"  Trimmed: {components['trimmed_segments']} frames")
                console.print(f"  Balanced: {components['balanced_score']:.3f}")
                console.print(f"  Sustained: {components['sustained_score']:.3f}")
                console.print(f"  Peak-heavy: {components['peak_score']:.3f}\n")

            # Generate plots for each test case
            trimmed = trim_silent_edges(probs, silence_threshold=args.silence_threshold)
            all_scores = {
                "balanced": components["balanced_score"],
                "width_heavy": components["sustained_score"],
                "peak_heavy": components["peak_score"],
            }
            test_output_dir = output_dir / name
            test_output_dir.mkdir(parents=True, exist_ok=True)
            save_plots(
                original_probs=np.array(probs, dtype=float),
                trimmed_probs=trimmed,
                output_dir=test_output_dir,
                silence_threshold=args.silence_threshold,
                all_scores=all_scores,
                components=components,
            )

        # Save test results
        test_results_path = output_dir / "test_results.json"
        test_results_path.write_text(
            json.dumps(results, indent=2, default=float), encoding="utf-8"
        )
        console.print(f"\n[green]✓ Test results saved to {test_results_path}[/green]")
        console.print(f"[green]✓ Test plots saved to {output_dir}[/green]")
        return

    if not args.probs_path:
        parser.print_help()
        console.print(
            "\n[red]Error: Either provide a JSON file path or use --test[/red]"
        )
        sys.exit(1)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Loading probabilities...", total=None)
            probs = load_probs_from_json(args.probs_path)
            progress.update(task, completed=True)

        console.print(
            f"[green]✓ Loaded {len(probs)} probabilities from {args.probs_path}[/green]"
        )

        trim_kwargs = {
            "silence_threshold": args.silence_threshold,
            "audio_samples": audio_samples,
            "sample_rate": sample_rate,
        }

        original_probs_array = np.array(probs, dtype=float)
        trimmed_probs = None
        trimmed_audio = None

        if trim_edges:
            # Get trimmed probabilities and audio if available
            if audio_samples is not None:
                trimmed_probs, trimmed_audio = trim_silent_edges(
                    original_probs_array, **trim_kwargs, return_audio=True
                )
            else:
                trimmed_probs = trim_silent_edges(original_probs_array, **trim_kwargs)

            trim_method = (
                "RMS-based audio" if audio_samples is not None else "probability-based"
            )
            trimmed_count = (
                len(original_probs_array) - len(trimmed_probs)
                if len(trimmed_probs) > 0
                else len(original_probs_array)
            )
            console.print(
                f"[yellow]✂️  Trimming enabled ({trim_method}, threshold: {args.silence_threshold})"
                f"{' - trimmed ' + str(trimmed_count) + ' frames' if trimmed_count > 0 else ''}[/yellow]"
            )
        else:
            trimmed_probs = original_probs_array.copy()
            if audio_samples is not None:
                trimmed_audio = audio_samples.copy()

        # Compute scores
        if args.components:
            components = get_score_components(
                probs, trim_edges=trim_edges, **trim_kwargs
            )
        else:
            components = None

        score = score_vad_segments(
            probs,
            method=args.method,
            trim_edges=trim_edges,
            **trim_kwargs,
        )

        all_scores = {
            "balanced": score_vad_segments(
                probs, method="balanced", trim_edges=trim_edges, **trim_kwargs
            ),
            "width_heavy": score_vad_segments(
                probs, method="width_heavy", trim_edges=trim_edges, **trim_kwargs
            ),
            "peak_heavy": score_vad_segments(
                probs, method="peak_heavy", trim_edges=trim_edges, **trim_kwargs
            ),
            "simple": score_vad_segments(
                probs, method="simple", trim_edges=trim_edges, **trim_kwargs
            ),
        }

        if not args.quiet:
            if components:
                display_results(probs, components)
            else:
                console.print(
                    f"\n[bold cyan]VAD Score ({args.method}):[/bold cyan] [bold green]{score:.3f}[/bold green]"
                )
                if score >= 0.85:
                    console.print(
                        "[green]✅ Excellent - Sustained high-confidence speech[/green]"
                    )
                elif score >= 0.70:
                    console.print("[blue]📢 Good - Clear voice activity[/blue]")
                elif score >= 0.55:
                    console.print(
                        "[yellow]⚠️ Marginal - Possible voice or noisy speech[/yellow]"
                    )
                elif score >= 0.40:
                    console.print(
                        "[orange1]❌ Poor - Unlikely voice activity[/orange1]"
                    )
                else:
                    console.print("[red]🔴 Invalid - No voice activity detected[/red]")

        # --- Save all result files to output directory ---
        saved_files = []

        # 1. Save original probabilities
        original_probs_path = output_dir / "original_probs.json"
        original_probs_path.write_text(
            json.dumps(original_probs_array.tolist(), indent=2), encoding="utf-8"
        )
        saved_files.append(("Original probabilities", original_probs_path))

        # 2. Save trimmed probabilities
        if trimmed_probs is not None and len(trimmed_probs) > 0:
            trimmed_probs_path = output_dir / "trimmed_probs.json"
            trimmed_probs_path.write_text(
                json.dumps(trimmed_probs.tolist(), indent=2), encoding="utf-8"
            )
            saved_files.append(("Trimmed probabilities", trimmed_probs_path))

        # 3. Save trimmed audio if available
        if trimmed_audio is not None and len(trimmed_audio) > 0:
            trimmed_audio_path = output_dir / "trimmed_audio.wav"
            sf.write(str(trimmed_audio_path), trimmed_audio, sample_rate)
            saved_files.append(("Trimmed audio", trimmed_audio_path))

        # 4. Save original audio copy if provided
        if audio_samples is not None:
            original_copy_path = output_dir / "original_audio.wav"
            sf.write(str(original_copy_path), audio_samples, sample_rate)
            saved_files.append(("Original audio", original_copy_path))

        # 5. Save scoring result JSON
        result = {
            "file": args.probs_path,
            "method": args.method,
            "score": score,
            "trim_settings": {
                "enabled": trim_edges,
                "threshold": args.silence_threshold,
                "using_audio_trim": audio_samples is not None,
                "trimmed_frames": len(original_probs_array) - len(trimmed_probs)
                if trimmed_probs is not None and len(trimmed_probs) > 0
                else 0,
            },
            "num_segments": len(original_probs_array),
            "original_statistics": {
                "max": float(np.max(original_probs_array)),
                "mean": float(np.mean(original_probs_array)),
                "min": float(np.min(original_probs_array)),
                "std": float(np.std(original_probs_array)),
                "median": float(np.median(original_probs_array)),
            },
            "trimmed_statistics": {
                "max": float(np.max(trimmed_probs)),
                "mean": float(np.mean(trimmed_probs)),
                "min": float(np.min(trimmed_probs)),
                "std": float(np.std(trimmed_probs)),
                "median": float(np.median(trimmed_probs)),
            }
            if trimmed_probs is not None and len(trimmed_probs) > 0
            else None,
            "input_audio": str(original_audio_path) if original_audio_path else None,
        }
        if components:
            result["components"] = components

        result_path = output_dir / "scoring_result.json"
        result_path.write_text(
            json.dumps(result, indent=2, default=float), encoding="utf-8"
        )
        saved_files.append(("Scoring result", result_path))

        # 6. Save all scores comparison
        scores_path = output_dir / "all_scores.json"
        scores_path.write_text(json.dumps(all_scores, indent=2), encoding="utf-8")
        saved_files.append(("All scores", scores_path))

        # 7. Save probability distribution data
        if trimmed_probs is not None and len(trimmed_probs) > 0:
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            dist_data = {
                "probabilities": trimmed_probs.tolist()
                if isinstance(trimmed_probs, np.ndarray)
                else list(trimmed_probs),
                "histogram": {
                    "bins": bins,
                    "counts": [
                        int(
                            np.sum(
                                (trimmed_probs >= bins[i])
                                & (trimmed_probs < bins[i + 1])
                            )
                        )
                        for i in range(len(bins) - 1)
                    ],
                },
            }
            dist_path = output_dir / "probability_distribution.json"
            dist_path.write_text(json.dumps(dist_data, indent=2), encoding="utf-8")
            saved_files.append(("Probability distribution", dist_path))

        # 8. Generate and save plots
        plot_files = save_plots(
            original_probs=original_probs_array,
            trimmed_probs=trimmed_probs,
            output_dir=output_dir,
            silence_threshold=args.silence_threshold
            if trim_edges
            else VAD_SILENCE_THRESHOLD,
            all_scores=all_scores,
            components=components,
        )
        saved_files.extend(plot_files)

        # Print summary of saved files
        output_dir_link = f"file://{output_dir.resolve()}"
        console.print(
            f"\n[bold green]✅ Results saved to [link={output_dir_link}]{output_dir}[/link][/bold green]"
        )
        for file_type, file_path in saved_files:
            # Use 'file://' link for clickable path if terminal supports it
            file_link = f"file://{file_path.resolve()}"
            console.print(
                f"  • {file_type}: [cyan][link={file_link}]{file_path.name}[/link][/cyan]"
            )

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON: {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
