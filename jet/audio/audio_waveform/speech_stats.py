from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np

# Reuse existing energy helpers
from jet.audio.helpers.energy_base import compute_rms

if TYPE_CHECKING:
    from .speech_events import SpeechSegmentEndEvent
from .speech_types import SpeechFrame


@dataclass
class SpeechSegmentStats:
    avg_smoothed_prob: float
    max_smoothed_prob: float
    min_smoothed_prob: float
    prob_std: float
    total_frames: int
    speech_frames: int
    speech_frame_ratio: float
    silence_ratio: float
    internal_silence_frames: int
    max_consecutive_silence_frames: int
    has_internal_pause: bool
    max_consecutive_speech_frames: int
    avg_consecutive_speech_frames: float
    speech_segments_count: int
    long_speech_ratio: float
    avg_rms: float
    rms_std: float
    speech_rms_mean: float
    silence_rms_mean: float
    energy_speech_correlation: float
    low_energy_speech_ratio: float
    high_energy_non_speech_ratio: float
    prob_velocity_mean: float
    prob_velocity_std: float
    prob_acceleration_mean: float
    speech_transition_rate: float
    estimated_speech_duration_sec: float
    confidence_level: str
    quality_score: float


def derive_segment_stats(event: SpeechSegmentEndEvent) -> SpeechSegmentStats:
    """Compute comprehensive statistics for a speech segment."""
    prob_frames: List[SpeechFrame] = event.prob_frames or []
    if not prob_frames:
        return _empty_stats()

    # Extract arrays
    smoothed_probs = np.array(
        [f["smoothed_prob"] for f in prob_frames], dtype=np.float32
    )
    is_speech = np.array([f["is_speech"] for f in prob_frames], dtype=bool)

    rms_values = _extract_rms(prob_frames, event)

    # Basic probability stats
    avg_prob = float(np.mean(smoothed_probs))
    max_prob = float(np.max(smoothed_probs))
    min_prob = float(np.min(smoothed_probs))
    prob_std = float(np.std(smoothed_probs)) if len(smoothed_probs) > 1 else 0.0

    total_frames = len(prob_frames)
    speech_frames = int(np.sum(is_speech))
    speech_ratio = speech_frames / total_frames if total_frames else 0.0
    silence_ratio = 1.0 - speech_ratio

    # Pause and run length analysis
    max_silence_run = _max_consecutive_run(~is_speech)
    has_pause = max_silence_run >= 5

    speech_runs = _collect_runs(is_speech)
    max_speech_run = max(speech_runs) if speech_runs else 0
    avg_speech_run = float(np.mean(speech_runs)) if speech_runs else 0.0
    speech_segments_count = len(speech_runs)

    long_speech_ratio = (
        sum(r for r in speech_runs if r >= 20) / total_frames if total_frames else 0.0
    )

    # RMS / Energy stats
    avg_rms = float(np.mean(rms_values))
    rms_std = float(np.std(rms_values)) if len(rms_values) > 1 else 0.0

    speech_rms_mean = (
        float(np.mean(rms_values[is_speech])) if np.any(is_speech) else 0.0
    )
    silence_rms_mean = (
        float(np.mean(rms_values[~is_speech])) if np.any(~is_speech) else 0.0
    )

    energy_corr = _safe_corr(smoothed_probs, rms_values)

    low_energy_speech_ratio = float(np.mean((is_speech) & (rms_values < avg_rms * 0.5)))
    high_energy_non_speech_ratio = float(
        np.mean((~is_speech) & (rms_values > avg_rms * 1.5))
    )

    # Probability dynamics
    velocity = np.diff(smoothed_probs)
    acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([])

    prob_velocity_mean = float(np.mean(np.abs(velocity))) if len(velocity) else 0.0
    prob_velocity_std = float(np.std(velocity)) if len(velocity) else 0.0
    prob_acceleration_mean = (
        float(np.mean(np.abs(acceleration))) if len(acceleration) else 0.0
    )

    transitions = np.sum(np.abs(np.diff(is_speech.astype(int))))
    transition_rate = float(transitions / total_frames) if total_frames else 0.0

    estimated_speech_sec = speech_frames * 0.010
    confidence_level = _confidence_from_prob(avg_prob)

    quality_score = _compute_quality_score(
        avg_prob=avg_prob,
        speech_ratio=speech_ratio,
        prob_std=prob_std,
        energy_corr=energy_corr,
        transition_rate=transition_rate,
        forced_split=event.forced_split,
    )

    return SpeechSegmentStats(
        avg_smoothed_prob=round(avg_prob, 3),
        max_smoothed_prob=round(max_prob, 3),
        min_smoothed_prob=round(min_prob, 3),
        prob_std=round(prob_std, 3),
        total_frames=total_frames,
        speech_frames=speech_frames,
        speech_frame_ratio=round(speech_ratio, 3),
        silence_ratio=round(silence_ratio, 3),
        internal_silence_frames=total_frames - speech_frames,
        max_consecutive_silence_frames=max_silence_run,
        has_internal_pause=has_pause,
        max_consecutive_speech_frames=max_speech_run,
        avg_consecutive_speech_frames=round(avg_speech_run, 3),
        speech_segments_count=speech_segments_count,
        long_speech_ratio=round(long_speech_ratio, 3),
        avg_rms=round(avg_rms, 5),
        rms_std=round(rms_std, 5),
        speech_rms_mean=round(speech_rms_mean, 5),
        silence_rms_mean=round(silence_rms_mean, 5),
        energy_speech_correlation=round(energy_corr, 3),
        low_energy_speech_ratio=round(low_energy_speech_ratio, 3),
        high_energy_non_speech_ratio=round(high_energy_non_speech_ratio, 3),
        prob_velocity_mean=round(prob_velocity_mean, 4),
        prob_velocity_std=round(prob_velocity_std, 4),
        prob_acceleration_mean=round(prob_acceleration_mean, 4),
        speech_transition_rate=round(transition_rate, 4),
        estimated_speech_duration_sec=round(estimated_speech_sec, 3),
        confidence_level=confidence_level,
        quality_score=quality_score,
    )


def _extract_rms(
    prob_frames: List[SpeechFrame], event: SpeechSegmentEndEvent
) -> np.ndarray:
    """Fallback-safe RMS extraction using canonical helper from jet.audio.helpers."""
    if prob_frames and "rms" in prob_frames[0]:
        return np.array([f["rms"] for f in prob_frames], dtype=np.float32)

    # Reuse existing code from jet.audio.helpers.energy_base
    if len(prob_frames) > 0 and event.audio is not None and len(event.audio) > 0:
        full_rms = compute_rms(event.audio)
        return np.full(len(prob_frames), full_rms, dtype=np.float32)

    return (
        np.zeros(len(prob_frames), dtype=np.float32)
        if prob_frames
        else np.array([], dtype=np.float32)
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Robust Pearson correlation that handles constant arrays safely."""
    if len(a) < 2 or len(b) < 2:
        return 0.0

    std_a = float(np.std(a))
    std_b = float(np.std(b))

    # Guard against near-constant signals (common when using full-segment RMS)
    if std_a < 1e-8 or std_b < 1e-8:
        return 0.0

    try:
        corr = np.corrcoef(a, b)[0, 1]
        if np.isnan(corr) or np.isinf(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def _max_consecutive_run(mask: np.ndarray) -> int:
    """Return length of longest consecutive True values in boolean mask."""
    if len(mask) == 0:
        return 0

    max_run = 0
    current = 0
    for val in mask:
        if val:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _collect_runs(mask: np.ndarray) -> List[int]:
    """Collect lengths of all consecutive True runs."""
    runs = []
    current = 0
    for val in mask:
        if val:
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0
    if current > 0:
        runs.append(current)
    return runs


def _confidence_from_prob(avg_prob: float) -> str:
    if avg_prob >= 0.75:
        return "high"
    elif avg_prob >= 0.5:
        return "medium"
    return "low"


def _compute_quality_score(
    *,
    avg_prob: float,
    speech_ratio: float,
    prob_std: float,
    energy_corr: float,
    transition_rate: float,
    forced_split: bool,
) -> float:
    stability = max(0.3, 1.0 - prob_std)
    alignment = max(0.3, energy_corr)
    continuity = max(0.3, 1.0 - transition_rate)
    penalty = 0.7 if forced_split else 1.0

    score = avg_prob * speech_ratio * stability * alignment * continuity * penalty
    return round(float(score), 3)


def _empty_stats() -> SpeechSegmentStats:
    return SpeechSegmentStats(
        avg_smoothed_prob=0.0,
        max_smoothed_prob=0.0,
        min_smoothed_prob=0.0,
        prob_std=0.0,
        total_frames=0,
        speech_frames=0,
        speech_frame_ratio=0.0,
        silence_ratio=0.0,
        internal_silence_frames=0,
        max_consecutive_silence_frames=0,
        has_internal_pause=False,
        max_consecutive_speech_frames=0,
        avg_consecutive_speech_frames=0.0,
        speech_segments_count=0,
        long_speech_ratio=0.0,
        avg_rms=0.0,
        rms_std=0.0,
        speech_rms_mean=0.0,
        silence_rms_mean=0.0,
        energy_speech_correlation=0.0,
        low_energy_speech_ratio=0.0,
        high_energy_non_speech_ratio=0.0,
        prob_velocity_mean=0.0,
        prob_velocity_std=0.0,
        prob_acceleration_mean=0.0,
        speech_transition_rate=0.0,
        estimated_speech_duration_sec=0.0,
        confidence_level="low",
        quality_score=0.0,
    )
