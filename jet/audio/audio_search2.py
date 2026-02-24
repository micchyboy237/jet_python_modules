from __future__ import annotations

from typing import Optional, TypedDict

import numpy as np


class AudioMatchResult(TypedDict):
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    confidence: float


def _validate_inputs(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
) -> None:
    if long_signal.ndim != 1 or short_signal.ndim != 1:
        raise ValueError("Signals must be 1D mono arrays.")

    if short_signal.size == 0 or long_signal.size == 0:
        raise ValueError("Signals must not be empty.")

    if short_signal.size > long_signal.size:
        raise ValueError("Short signal cannot be longer than long signal.")


def _compute_normalized_cross_correlation(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
) -> np.ndarray:
    long_signal = long_signal.astype(np.float64)
    short_signal = short_signal.astype(np.float64)

    m = short_signal.size

    short_mean = short_signal.mean()
    short_zero = short_signal - short_mean
    short_energy = np.sum(short_zero**2)

    if short_energy == 0:
        return np.zeros(long_signal.size - m + 1)

    numerator = np.correlate(long_signal, short_zero, mode="valid")

    window = np.ones(m)
    long_sum = np.convolve(long_signal, window, mode="valid")
    long_sq_sum = np.convolve(long_signal**2, window, mode="valid")

    long_mean = long_sum / m
    long_energy = long_sq_sum - m * (long_mean**2)

    denominator = np.sqrt(long_energy * short_energy)
    denominator[denominator == 0] = np.inf

    ncc = numerator / denominator
    return np.clip(ncc, -1.0, 1.0)


def find_audio_offset(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    sample_rate: int,
    confidence_threshold: float = 0.8,
    tie_break_epsilon: float = 1e-9,
) -> Optional[AudioMatchResult]:
    _validate_inputs(long_signal, short_signal)

    ncc = _compute_normalized_cross_correlation(
        long_signal=long_signal,
        short_signal=short_signal,
    )

    max_score = float(np.max(ncc))

    if max_score < confidence_threshold:
        return None

    # Find all indices close to max score
    candidate_indices = np.where(np.abs(ncc - max_score) <= tie_break_epsilon)[0]

    # Choose earliest match deterministically
    best_index = int(candidate_indices[0])

    start_sample = best_index
    end_sample = best_index + short_signal.size

    return AudioMatchResult(
        start_sample=start_sample,
        end_sample=end_sample,
        start_time=start_sample / sample_rate,
        end_time=end_sample / sample_rate,
        confidence=max_score,
    )
