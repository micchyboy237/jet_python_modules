from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, TypedDict

import numpy as np
from rich.console import Console
from scipy.io import wavfile
from scipy.ndimage import maximum_filter
from scipy.signal import correlate, stft

console = Console()


# ============================================================
# Result Types
# ============================================================


class AudioMatchResult(TypedDict):
    reference_start_sample: int
    reference_end_sample: int
    reference_start_time: float
    reference_end_time: float

    query_start_sample: int
    query_end_sample: int
    query_start_time: float
    query_end_time: float

    offset_frames: int
    votes: int
    confidence: float


class Fingerprint(TypedDict):
    hash: int
    time_bin: int


@dataclass
class Peak:
    time_bin: int
    freq_bin: int
    magnitude: float


# ============================================================
# Utilities
# ============================================================


def _validate_signal(signal: np.ndarray) -> None:
    if signal.ndim != 1:
        raise ValueError("Signal must be mono.")
    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")


def _normalize_order(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Ensure first signal is longer.
    """
    if a.size >= b.size:
        return a, b, False
    return b, a, True


# ============================================================
# Spectrogram
# ============================================================


def compute_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    window_size: int = 2048,
    hop_length: int = 512,
):
    f, t, Z = stft(
        signal,
        fs=sample_rate,
        nperseg=window_size,
        noverlap=window_size - hop_length,
        padded=False,
        boundary=None,
    )

    return np.abs(Z)


# ============================================================
# Peak Detection
# ============================================================


def detect_spectral_peaks(
    spectrogram: np.ndarray,
    freq_radius: int = 10,
    time_radius: int = 10,
    threshold: float = 0.1,
) -> List[Peak]:
    neighborhood = (freq_radius * 2 + 1, time_radius * 2 + 1)

    local_max = maximum_filter(spectrogram, size=neighborhood) == spectrogram

    mask = local_max & (spectrogram > threshold)

    peaks: List[Peak] = []

    freq_bins, time_bins = np.where(mask)

    for f, t in zip(freq_bins, time_bins):
        peaks.append(
            Peak(
                time_bin=int(t),
                freq_bin=int(f),
                magnitude=float(spectrogram[f, t]),
            )
        )

    return peaks


# ============================================================
# Fingerprints
# ============================================================


def _hash_pair(f1: int, f2: int, dt: int) -> int:
    payload = f"{f1}|{f2}|{dt}".encode()
    digest = hashlib.sha1(payload).digest()
    return int.from_bytes(digest[:4], "little")


def generate_fingerprints(
    peaks: List[Peak],
    fanout: int = 10,
    max_time_delta: int = 200,
) -> List[Fingerprint]:
    fingerprints: List[Fingerprint] = []

    peaks = sorted(peaks, key=lambda p: p.time_bin)

    for i, anchor in enumerate(peaks):
        for j in range(1, fanout + 1):
            if i + j >= len(peaks):
                break

            target = peaks[i + j]

            dt = target.time_bin - anchor.time_bin

            if dt <= 0:
                continue

            if dt > max_time_delta:
                break

            h = _hash_pair(anchor.freq_bin, target.freq_bin, dt)

            fingerprints.append(
                Fingerprint(
                    hash=h,
                    time_bin=anchor.time_bin,
                )
            )

    return fingerprints


# ============================================================
# Index
# ============================================================


def build_index(
    fingerprints: Iterable[Fingerprint],
) -> Dict[int, List[int]]:
    index: Dict[int, List[int]] = {}

    for fp in fingerprints:
        index.setdefault(fp["hash"], []).append(fp["time_bin"])

    return index


# ============================================================
# Matching
# ============================================================


def fingerprint_vote(
    query_fp: List[Fingerprint],
    reference_index: Dict[int, List[int]],
) -> Dict[int, int]:
    votes: Dict[int, int] = {}

    for fp in query_fp:
        ref_times = reference_index.get(fp["hash"])

        if not ref_times:
            continue

        for t in ref_times:
            offset = t - fp["time_bin"]

            votes[offset] = votes.get(offset, 0) + 1

    return votes


# ============================================================
# NCC Refinement
# ============================================================


def _normalized_cross_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    b = b - b.mean()
    a = a - a.mean()

    corr = correlate(a, b, mode="valid", method="fft")

    denom = np.sqrt(np.sum(b**2) * np.convolve(a**2, np.ones(len(b)), "valid"))

    denom[denom == 0] = np.inf

    return corr / denom


def refine_offset_with_ncc(
    long_signal: np.ndarray,
    short_signal: np.ndarray,
    approx_offset_samples: int,
    search_radius: int = 44100,
) -> tuple[int, float]:
    start = max(0, approx_offset_samples - search_radius)

    end = min(
        long_signal.size,
        approx_offset_samples + search_radius + short_signal.size,
    )

    segment = long_signal[start:end]

    ncc = _normalized_cross_correlation(segment, short_signal)

    best = int(np.argmax(ncc))

    best_score = float(ncc[best])

    refined_offset = start + best

    return refined_offset, best_score


# ============================================================
# Main Hybrid API
# ============================================================


def find_audio_match_hybrid(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    sample_rate: int,
    min_match_duration_sec: float = 2.0,
    threshold: float = 0.65,
    top_k: int = 20,
    verbose: bool = True,
) -> List[AudioMatchResult]:
    _validate_signal(signal_a)
    _validate_signal(signal_b)

    long_signal, short_signal, swapped = _normalize_order(signal_a, signal_b)

    start_time = time.perf_counter()

    if verbose:
        console.rule("Hybrid Fingerprint + NCC Matching")

    # -------------------------
    # reference fingerprints
    # -------------------------

    spec = compute_spectrogram(long_signal, sample_rate)

    peaks = detect_spectral_peaks(spec)

    ref_fp = generate_fingerprints(peaks)

    ref_index = build_index(ref_fp)

    # -------------------------
    # query fingerprints
    # -------------------------

    spec_q = compute_spectrogram(short_signal, sample_rate)

    peaks_q = detect_spectral_peaks(spec_q)

    query_fp = generate_fingerprints(peaks_q)

    # -------------------------
    # vote offsets
    # -------------------------

    votes = fingerprint_vote(query_fp, ref_index)

    if not votes:
        return []

    sorted_offsets = sorted(
        votes.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    hop_length = 512

    results: List[AudioMatchResult] = []

    for offset_frames, vote_count in sorted_offsets:
        approx_offset_samples = offset_frames * hop_length

        refined_offset, confidence = refine_offset_with_ncc(
            long_signal,
            short_signal,
            approx_offset_samples,
        )

        if confidence < threshold:
            continue

        ref_start = refined_offset
        ref_end = refined_offset + short_signal.size

        result: AudioMatchResult = AudioMatchResult(
            reference_start_sample=ref_start,
            reference_end_sample=ref_end,
            reference_start_time=ref_start / sample_rate,
            reference_end_time=ref_end / sample_rate,
            query_start_sample=0,
            query_end_sample=short_signal.size,
            query_start_time=0.0,
            query_end_time=short_signal.size / sample_rate,
            offset_frames=offset_frames,
            votes=vote_count,
            confidence=confidence,
        )

        results.append(result)

    results.sort(key=lambda x: x["confidence"], reverse=True)

    if verbose:
        elapsed = time.perf_counter() - start_time

        console.print(f"[green]{len(results)} matches found in {elapsed:.2f}s[/green]")

    return results


# ============================================================
# CLI
# ============================================================


def load_audio(path: str, sample_rate_override: int | None = None):
    sr, data = wavfile.read(path)

    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    if sample_rate_override and sample_rate_override != sr:
        raise ValueError(
            f"Sample rate mismatch: file={sr}, override={sample_rate_override}"
        )

    return sr, data


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid fingerprint + NCC audio matcher"
    )

    parser.add_argument("reference", type=Path, help="reference audio file")

    parser.add_argument("query", type=Path, help="query audio file")

    parser.add_argument(
        "-m",
        "--min-duration",
        type=float,
        default=1.0,
        help="minimum match duration seconds",
    )

    parser.add_argument(
        "-s",
        "--sample-rate",
        type=int,
        default=None,
        help="override sample rate",
    )

    parser.add_argument(
        "-r",
        "--radius",
        type=int,
        default=44100,
        help="NCC refinement search radius in samples",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.65,
        help="minimum NCC confidence threshold",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="disable verbose output",
    )

    return parser.parse_args()


def main() -> None:
    args = get_args()

    if not args.reference.exists():
        raise FileNotFoundError(args.reference)

    if not args.query.exists():
        raise FileNotFoundError(args.query)

    sr_a, signal_a = load_audio(args.reference, args.sample_rate)
    sr_b, signal_b = load_audio(args.query, args.sample_rate)

    if sr_a != sr_b:
        raise ValueError("Sample rates must match")

    results = find_audio_match_hybrid(
        signal_a,
        signal_b,
        sample_rate=sr_a,
        min_match_duration_sec=args.min_duration,
        threshold=args.threshold,
        verbose=not args.quiet,
    )

    if not results:
        console.print("[red]No match found[/red]")
        return

    console.print("\n[bold green]Matches[/bold green]")

    for i, r in enumerate(results):
        console.print(
            f"[cyan]#{i + 1}[/cyan] "
            f"ref {r['reference_start_time']:.2f}s → {r['reference_end_time']:.2f}s | "
            f"query {r['query_start_time']:.2f}s → {r['query_end_time']:.2f}s | "
            f"votes={r['votes']} "
            f"conf={r['confidence']:.3f}"
        )


if __name__ == "__main__":
    main()
