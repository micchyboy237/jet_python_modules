from __future__ import annotations

import time
from dataclasses import dataclass
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


def find_audio_offsets(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    sample_rate: int,
    verbose: bool = True,
    confidence_threshold: float = 0.8,
    min_distance_samples: Optional[int] = None,
    tie_break_epsilon: float = 1e-9,
) -> list[AudioMatchResult]:
    """
    Find all high-confidence occurrences of short_signal inside long_signal.

    Returns a list of matches sorted by starting position.
    Uses greedy non-maximum suppression to avoid reporting near-duplicate/strongly overlapping detections.
    """
    _validate_inputs(long_signal, short_signal)

    if min_distance_samples is None:
        # Allow very small overlaps only (e.g. 50 samples), forbid most near-duplicates
        min_distance_samples = short_signal.size - 50

    ncc = _compute_normalized_cross_correlation(
        long_signal=long_signal,
        short_signal=short_signal,
        verbose=verbose,
    )

    # Find all positions above threshold
    candidate_mask = (
        ncc >= confidence_threshold - tie_break_epsilon
    )  # slight relaxation for ties
    if not np.any(candidate_mask):
        return []

    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_scores = ncc[candidate_indices]

    # Sort candidates by descending score (best first)
    sorted_order = np.argsort(-candidate_scores)
    sorted_candidates = candidate_indices[sorted_order]
    sorted_scores = candidate_scores[sorted_order]

    selected: list[int] = []
    used = np.zeros(len(ncc), dtype=bool)

    # Greedy NMS: take best remaining, suppress neighborhood
    for idx, score in zip(sorted_candidates, sorted_scores):
        if used[idx]:
            continue

        selected.append(idx)

        # Suppress nearby peaks (symmetric window)
        start_suppress = max(0, idx - min_distance_samples)
        end_suppress = min(len(ncc), idx + min_distance_samples + 1)
        used[start_suppress:end_suppress] = True

    # Build final results sorted by start position
    results = [
        AudioMatchResult(
            start_sample=start,
            end_sample=start + short_signal.size,
            start_time=start / sample_rate,
            end_time=(start + short_signal.size) / sample_rate,
            confidence=float(score),  # use the original peak score
        )
        for start, score in zip(
            sorted(selected),
            [
                ncc[s] for s in sorted(selected)
            ],  # or keep sorted_scores if order preserved
        )
    ]

    return results


@dataclass
class PartialAudioMatch:
    start_sample: int
    end_sample: int
    confidence: float
    match_length_samples: int  # how many samples of short were used


def extract_sliding_subsignals(
    signal: np.ndarray,
    min_length: int,
    max_length: int,
    step: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """
    Generate overlapping sub-signals from signal[start:start+length].
    Returns list of (sub_signal, original_start_idx)
    """
    if step is None:
        step = max(1, (max_length - min_length) // 8)
    subs = []
    for length in range(min_length, max_length + 1, step):
        for start in range(0, signal.size - length + 1, step):
            subs.append((signal[start : start + length], start))
    return subs


def find_partial_audio_matches(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    sample_rate: int,
    verbose: bool = True,
    confidence_threshold: float = 0.75,  # usually lower for partial
    min_match_fraction: float = 0.5,
    max_match_fraction: float = 1.0,
    length_step_fraction: float = 0.1,
    min_distance_samples: int | None = None,
    tie_break_epsilon: float = 1e-9,
    max_subclips: int | None = 60,
) -> list[PartialAudioMatch]:
    """
    Find partial matches by trying multiple sub-clips of short_signal.
    Returns sorted list of best non-overlapping partial matches.
    """
    _validate_inputs(long_signal, short_signal)

    min_len = max(64, int(len(short_signal) * min_match_fraction))
    max_len = int(len(short_signal) * max_match_fraction)
    step = max(1, int(len(short_signal) * length_step_fraction))

    sub_clips = extract_sliding_subsignals(short_signal, min_len, max_len, step=step)

    # Optional: early exit if full-length match is very confident
    full_matches = find_audio_offsets(
        long_signal,
        short_signal,
        sample_rate,
        verbose=False,
        confidence_threshold=0.90,
        min_distance_samples=min_distance_samples,
    )
    if full_matches and max(m["confidence"] for m in full_matches) >= 0.92:
        # Convert to PartialAudioMatch format
        return [
            PartialAudioMatch(
                start_sample=m["start_sample"],
                end_sample=m["end_sample"],
                confidence=m["confidence"],
                match_length_samples=len(short_signal),
            )
            for m in full_matches
        ]

    from joblib import Parallel, delayed

    def process_one(sub_clip: np.ndarray, sub_start: int):
        matches = find_audio_offsets(
            long_signal=long_signal,
            short_signal=sub_clip,
            sample_rate=sample_rate,
            verbose=False,
            confidence_threshold=confidence_threshold,
            min_distance_samples=min_distance_samples,
            tie_break_epsilon=tie_break_epsilon,
        )
        return [(m, sub_start) for m in matches]

    if max_subclips is not None and len(sub_clips) > max_subclips:
        # could subsample here, but for simplicity just truncate
        sub_clips = sub_clips[:max_subclips]

    results = Parallel(n_jobs=-1)(
        delayed(process_one)(clip, start) for clip, start in sub_clips
    )
    all_matches: list[tuple[AudioMatchResult, int]] = [
        item for sublist in results for item in sublist
    ]

    # Sort by start position, then prefer longer + higher conf
    all_matches.sort(
        key=lambda x: (
            x[0]["start_sample"],
            -x[0]["end_sample"] + x[0]["start_sample"],
            -x[0]["confidence"],
        )
    )

    # Simple greedy non-overlap suppression (can be refined)
    selected: list[PartialAudioMatch] = []
    used_ranges = []

    for match, sub_start in all_matches:
        s, e = match["start_sample"], match["end_sample"]
        if any(s < ue and e > us for us, ue in used_ranges):
            continue
        selected.append(
            PartialAudioMatch(
                start_sample=s,
                end_sample=e,
                confidence=match["confidence"],
                match_length_samples=e - s,
            )
        )
        used_ranges.append((s, e))

    return sorted(selected, key=lambda x: x.start_sample)
