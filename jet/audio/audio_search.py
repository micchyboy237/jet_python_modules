from __future__ import annotations

import time
from typing import Optional, TypedDict

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from scipy.signal import correlate, fftconvolve


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
    verbose: bool = True,
) -> np.ndarray:
    """Compute normalized cross-correlation using FFT for speed."""
    # Use float64 to avoid precision issues in FFT-based NCC
    long_signal = long_signal.astype(np.float64)
    short_signal = short_signal.astype(np.float64)

    m = short_signal.size
    short_mean = short_signal.mean()
    short_zero = short_signal - short_mean
    short_energy = np.sum(short_zero**2)
    if short_energy == 0:
        return np.zeros(long_signal.size - m + 1, dtype=np.float64)

    if verbose:
        console = Console()
        console.rule("Computing normalized cross-correlation (FFT-based)")
        start_t = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]FFT-based NCC...", total=3)

        # Use correlate with fft method → handles flip internally
        numerator = correlate(long_signal, short_zero, mode="valid", method="fft")
        progress.advance(task)

        long_sum = fftconvolve(long_signal, np.ones(m), mode="valid")
        progress.advance(task)

        long_sq_sum = fftconvolve(long_signal**2, np.ones(m), mode="valid")
        progress.advance(task)

    long_mean = long_sum / m
    # Prevent negative energy due to floating-point errors
    long_energy = np.maximum(long_sq_sum - m * (long_mean**2), 0.0)

    denominator = np.sqrt(long_energy * short_energy)
    denominator[denominator == 0] = np.inf

    ncc = numerator / denominator
    ncc = np.clip(ncc, -1.0, 1.0)

    if verbose:
        elapsed = time.perf_counter() - start_t
        console.print(f"[dim]NCC computation took {elapsed:.1f} seconds[/dim]")

    return ncc


def find_audio_offset(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    sample_rate: int,
    verbose: bool = True,
    confidence_threshold: float = 0.8,
    tie_break_epsilon: float = 1e-9,
) -> Optional[AudioMatchResult]:
    _validate_inputs(long_signal, short_signal)

    ncc = _compute_normalized_cross_correlation(
        long_signal=long_signal,
        short_signal=short_signal,
        verbose=verbose,
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
