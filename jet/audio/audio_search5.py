from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, TypedDict

import numpy as np
from rich.console import Console
from scipy.io import wavfile
from scipy.ndimage import maximum_filter
from scipy.signal import correlate, stft

try:
    import numba
except ImportError:
    numba = None

console = Console()

WINDOW_SIZE = 2048
HOP_LENGTH = 512


class AudioMatchResult(TypedDict):
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    confidence: float


@dataclass
class Peak:
    time_bin: int
    freq_bin: int
    magnitude: float


# ============================================================
# Audio loading
# ============================================================


def load_audio(path: Path):
    sr, data = wavfile.read(path)

    if data.ndim > 1:
        data = data.mean(axis=1)

    return sr, data.astype(np.float32)


# ============================================================
# Spectrogram
# ============================================================


def compute_spectrogram(signal, sr):
    _, _, Z = stft(
        signal,
        fs=sr,
        nperseg=WINDOW_SIZE,
        noverlap=WINDOW_SIZE - HOP_LENGTH,
        padded=False,
    )

    return np.abs(Z)


# ============================================================
# Peak detection
# ============================================================


if numba:

    @numba.njit(parallel=True)
    def _peak_mask(spec, threshold):
        rows, cols = spec.shape
        mask = np.zeros(spec.shape, dtype=np.uint8)

        for i in numba.prange(1, rows - 1):
            for j in range(1, cols - 1):
                v = spec[i, j]

                if v < threshold:
                    continue

                if (
                    v > spec[i - 1, j]
                    and v > spec[i + 1, j]
                    and v > spec[i, j - 1]
                    and v > spec[i, j + 1]
                ):
                    mask[i, j] = 1

        return mask


def detect_peaks(spec, threshold=0.1):
    if numba:
        mask = _peak_mask(spec, threshold)
    else:
        local_max = maximum_filter(spec, size=(5, 5)) == spec
        mask = local_max & (spec > threshold)

    freq, time = np.where(mask)

    return [Peak(int(t), int(f), float(spec[f, t])) for f, t in zip(freq, time)]


# ============================================================
# Bit-packed fingerprint
# ============================================================


def hash_pair(f1, f2, dt):
    return ((f1 & 1023) << 22) | ((f2 & 1023) << 12) | (dt & 4095)


def generate_fingerprints(peaks, fanout=8, max_dt=200):
    peaks = sorted(peaks, key=lambda p: p.time_bin)

    hashes = []
    times = []

    for i, a in enumerate(peaks):
        for j in range(1, fanout + 1):
            if i + j >= len(peaks):
                break

            b = peaks[i + j]

            dt = b.time_bin - a.time_bin

            if dt <= 0 or dt > max_dt:
                continue

            h = hash_pair(a.freq_bin, b.freq_bin, dt)

            hashes.append(h)
            times.append(a.time_bin)

    return np.array(hashes, np.uint32), np.array(times, np.uint32)


# ============================================================
# LSH bucket index
# ============================================================


class FingerprintDB:
    def __init__(self, db_dir: Path):
        self.hash_file = db_dir / "hashes.memmap"
        self.time_file = db_dir / "times.memmap"

        if not self.hash_file.exists():
            raise RuntimeError("Fingerprint DB not found. Use --build-db")

        self.hashes = np.memmap(self.hash_file, dtype=np.uint32, mode="r")
        self.times = np.memmap(self.time_file, dtype=np.uint32, mode="r")

        # build LSH buckets
        self.buckets: Dict[int, np.ndarray] = {}

        for i, h in enumerate(self.hashes):
            key = int(h >> 12)
            self.buckets.setdefault(key, []).append(i)

        for k in self.buckets:
            self.buckets[k] = np.array(self.buckets[k])

    def lookup(self, h):
        key = int(h >> 12)

        idx = self.buckets.get(key)

        if idx is None:
            return []

        matches = idx[self.hashes[idx] == h]

        return self.times[matches]


# ============================================================
# DB Builder
# ============================================================


def build_database(input_path: Path, db_dir: Path):
    db_dir.mkdir(exist_ok=True)

    files: List[Path] = []

    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.rglob("*.wav"))

    if not files:
        raise RuntimeError("No WAV files found to build database.")

    console.print(f"[cyan]Indexing {len(files)} audio files...")

    all_hashes = []
    all_times = []

    for file in files:
        sr, sig = load_audio(file)

        spec = compute_spectrogram(sig, sr)

        peaks = detect_peaks(spec)

        hashes, times = generate_fingerprints(peaks)

        if hashes.size == 0:
            continue

        all_hashes.append(hashes)
        all_times.append(times)

    if not all_hashes:
        raise RuntimeError("No fingerprints extracted.")

    hashes = np.concatenate(all_hashes)
    times = np.concatenate(all_times)

    np.memmap(db_dir / "hashes.memmap", dtype=np.uint32, mode="w+", shape=hashes.shape)[
        :
    ] = hashes
    np.memmap(db_dir / "times.memmap", dtype=np.uint32, mode="w+", shape=times.shape)[
        :
    ] = times

    console.print(f"[green]DB built with {len(hashes):,} fingerprints[/green]")


# ============================================================
# NCC refinement
# ============================================================


def normalized_xcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()

    corr = correlate(a, b, mode="valid", method="fft")

    denom = np.sqrt(np.sum(b**2) * np.convolve(a**2, np.ones(len(b)), "valid"))

    denom[denom == 0] = np.inf

    return corr / denom


def refine_offset(long_signal, short_signal, approx, radius):
    start = max(0, approx - radius)
    end = min(len(long_signal), approx + radius + len(short_signal))

    segment = long_signal[start:end]

    ncc = normalized_xcorr(segment, short_signal)

    idx = int(np.argmax(ncc))

    return start + idx, float(ncc[idx])


# ============================================================
# Matching
# ============================================================


def match_audio(sig_a, sig_b, db, sr, threshold, radius):
    if len(sig_a) >= len(sig_b):
        long_sig, short_sig = sig_a, sig_b
    else:
        long_sig, short_sig = sig_b, sig_a

    spec = compute_spectrogram(short_sig, sr)

    peaks = detect_peaks(spec)

    hashes, times = generate_fingerprints(peaks)

    votes: Dict[int, int] = {}

    for h, t in zip(hashes, times):
        for ref_t in db.lookup(h):
            off = int(ref_t) - int(t)

            votes[off] = votes.get(off, 0) + 1

    if not votes:
        return None

    best_offset_frames = max(votes, key=votes.get)

    approx_samples = best_offset_frames * HOP_LENGTH

    offset, conf = refine_offset(long_sig, short_sig, approx_samples, radius)

    if conf < threshold:
        return None

    return AudioMatchResult(
        start_sample=offset,
        end_sample=offset + len(short_sig),
        start_time=offset / sr,
        end_time=(offset + len(short_sig)) / sr,
        confidence=conf,
    )


# ============================================================
# CLI
# ============================================================


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument("reference", type=Path)
    p.add_argument("query", type=Path)

    p.add_argument("-d", "--db", type=Path, required=True)

    p.add_argument("-t", "--threshold", type=float, default=0.75)

    p.add_argument("-r", "--radius", type=int, default=44100)

    p.add_argument("--build-db", action="store_true")

    p.add_argument("-q", "--quiet", action="store_true")

    return p.parse_args()


# ============================================================
# Main
# ============================================================


def main():
    args = get_args()

    if args.build_db:
        build_database(args.reference, args.db)
        return

    sr_a, sig_a = load_audio(args.reference)
    sr_b, sig_b = load_audio(args.query)

    if sr_a != sr_b:
        raise ValueError("Sample rate mismatch")

    db = FingerprintDB(args.db)

    start = time.perf_counter()

    result = match_audio(
        sig_a,
        sig_b,
        db,
        sr_a,
        args.threshold,
        args.radius,
    )

    elapsed = time.perf_counter() - start

    if result:
        console.print("\n[bold green]Match Found[/bold green]")
        console.print(result)

    else:
        console.print("[red]No match[/red]")

    console.print(f"[dim]Search time: {elapsed:.3f}s[/dim]")


if __name__ == "__main__":
    main()
