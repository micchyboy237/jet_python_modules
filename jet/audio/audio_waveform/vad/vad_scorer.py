"""
vad_scorer.py
=============
Reusable scoring module for VAD speech-probability lists.
No ground-truth labels required.

Usage
-----
    from vad_scorer import VADScorer

    probs = [0.01, 0.02, 0.95, 0.97, 0.98, 0.50, 0.03, 0.99]
    scorer = VADScorer(probs)

    print(scorer.summary())          # dict of all metrics + quality label
    print(scorer.label())            # "Good", "Very bad", etc.
    scorer.plot()                    # requires matplotlib
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Quality bands
# ---------------------------------------------------------------------------

QUALITY_BANDS: List[Tuple[float, str, str]] = [
    # (min_confidence_exclusive, label, description)
    (
        0.80,
        "Very good",
        "Probs are highly bimodal — strong, confident speech/silence separation.",
    ),
    (
        0.60,
        "Good",
        "Mostly crisp decisions. A few borderline frames but overall well-separated.",
    ),
    (0.40, "Fair", "Mixed confidence. Some clear frames, many near 0.5."),
    (
        0.20,
        "Bad",
        "Weak bimodality. Many frames ambiguous. Consider de-noising or re-tuning.",
    ),
    (
        0.00,
        "Very bad",
        "Probs cluster near 0.5 throughout — model is uncertain. High noise or poor audio.",
    ),
]


def _quality_from_confidence(confidence: float) -> Tuple[str, str]:
    """Return (label, description) for a given confidence score (0–1)."""
    for threshold, label, description in QUALITY_BANDS:
        if confidence > threshold:
            return label, description
    return QUALITY_BANDS[-1][1], QUALITY_BANDS[-1][2]


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass
class VADMetrics:
    """All computed metrics for a probability list."""

    # Raw descriptors
    n_frames: int
    mean_prob: float
    median_prob: float
    std_prob: float

    # No-GT metrics
    confidence: float  # mean(|p - 0.5| * 2)  →  0=uncertain, 1=crisp
    speech_ratio: float  # fraction of frames above threshold
    jitter: float  # mean frame-to-frame |delta|
    bimodality_coeff: float  # Sarle's BC: (skew²+1) / kurtosis  →  >0.555 = bimodal

    # Segment stats
    n_speech_segments: int
    n_silence_segments: int
    mean_speech_segment_len: float  # in frames
    mean_silence_segment_len: float  # in frames

    # Quality label
    quality_label: str
    quality_description: str

    # Config
    threshold: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class VADScorer:
    """
    Score a list of VAD speech probabilities without ground-truth labels.

    Parameters
    ----------
    probs     : list of floats in [0, 1]
    threshold : decision boundary for speech/silence (default 0.5)
    """

    def __init__(self, probs: List[float], threshold: float = 0.5):
        if not probs:
            raise ValueError("probs list must not be empty.")
        for i, p in enumerate(probs):
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"probs[{i}] = {p} is outside [0, 1].")

        self._probs = list(probs)
        self._threshold = threshold
        self._metrics: Optional[VADMetrics] = None  # lazy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def metrics(self) -> VADMetrics:
        """Return the full VADMetrics dataclass (cached)."""
        if self._metrics is None:
            self._metrics = self._compute()
        return self._metrics

    def summary(self) -> dict:
        """Return all metrics as a plain dict."""
        return self.metrics().as_dict()

    def label(self) -> str:
        """Return just the quality label string."""
        return self.metrics().quality_label

    def confidence(self) -> float:
        """Return the confidence score (0–1)."""
        return self.metrics().confidence

    def speech_ratio(self) -> float:
        """Fraction of frames classified as speech."""
        return self.metrics().speech_ratio

    def report(self) -> str:
        """Human-readable report."""
        m = self.metrics()
        lines = [
            "=" * 52,
            f"  VAD Score Report  (n={m.n_frames} frames, τ={m.threshold})",
            "=" * 52,
            "",
            "── Distribution ─────────────────────────────────",
            f"  Mean prob        : {m.mean_prob:.4f}",
            f"  Median prob      : {m.median_prob:.4f}",
            f"  Std dev          : {m.std_prob:.4f}",
            "",
            "── No-GT Metrics ────────────────────────────────",
            f"  Confidence       : {m.confidence:.4f}   (0=uncertain → 1=crisp)",
            f"  Speech ratio     : {m.speech_ratio:.1%}",
            f"  Jitter           : {m.jitter:.4f}   (frame-to-frame |delta|)",
            f"  Bimodality coeff : {m.bimodality_coeff:.4f}  (>0.555 = bimodal)",
            "",
            "── Segment Stats ────────────────────────────────",
            f"  Speech segments  : {m.n_speech_segments}  (avg len {m.mean_speech_segment_len:.1f} frames)",
            f"  Silence segments : {m.n_silence_segments}  (avg len {m.mean_silence_segment_len:.1f} frames)",
            "",
            "── Quality ──────────────────────────────────────",
            f"  Label            : {m.quality_label}",
            f"  Note             : {m.quality_description}",
            "=" * 52,
        ]
        return "\n".join(lines)

    def plot(self, title: str = "VAD probability list") -> None:
        """
        Plot the probability list with threshold line and quality annotation.
        Requires matplotlib.
        """
        try:
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot(). pip install matplotlib"
            )

        m = self.metrics()
        probs = self._probs

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.fill_between(range(len(probs)), probs, alpha=0.25, color="#3B8BD4")
        ax.plot(probs, color="#185FA5", linewidth=1.2, label="speech prob")
        ax.axhline(
            self._threshold,
            color="#E24B4A",
            linewidth=1,
            linestyle="--",
            label=f"threshold ({self._threshold})",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Frame")
        ax.set_ylabel("P(speech)")
        ax.set_title(
            f"{title}  —  Quality: {m.quality_label}  |  Confidence: {m.confidence:.3f}"
        )
        ax.legend(loc="upper right", fontsize=9)

        # Shade speech regions
        in_speech = False
        start = 0
        for i, p in enumerate(probs):
            is_speech = p >= self._threshold
            if is_speech and not in_speech:
                start = i
                in_speech = True
            elif not is_speech and in_speech:
                ax.axvspan(start, i, alpha=0.12, color="#3B8BD4")
                in_speech = False
        if in_speech:
            ax.axvspan(start, len(probs), alpha=0.12, color="#3B8BD4")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(self) -> VADMetrics:
        probs = self._probs
        tau = self._threshold
        n = len(probs)

        # Basic stats
        mean_p = sum(probs) / n
        median_p = statistics.median(probs)
        std_p = statistics.pstdev(probs)  # population std

        # Confidence: mean(|p - 0.5| * 2)
        confidence = sum(abs(p - 0.5) * 2 for p in probs) / n

        # Speech ratio
        speech_ratio = sum(1 for p in probs if p >= tau) / n

        # Jitter: mean frame-to-frame absolute delta
        if n > 1:
            jitter = sum(abs(probs[i] - probs[i - 1]) for i in range(1, n)) / (n - 1)
        else:
            jitter = 0.0

        # Bimodality coefficient (Sarle 1981): (skew²+1) / kurtosis
        # Values > 0.555 indicate bimodal distribution
        bimodality_coeff = self._bimodality_coeff(probs, mean_p, std_p, n)

        # Segment analysis
        n_speech_segs, n_sil_segs, mean_sp_len, mean_sil_len = self._segment_stats(
            probs, tau
        )

        # Quality label
        quality_label, quality_desc = _quality_from_confidence(confidence)

        return VADMetrics(
            n_frames=n,
            mean_prob=round(mean_p, 6),
            median_prob=round(median_p, 6),
            std_prob=round(std_p, 6),
            confidence=round(confidence, 6),
            speech_ratio=round(speech_ratio, 6),
            jitter=round(jitter, 6),
            bimodality_coeff=round(bimodality_coeff, 6),
            n_speech_segments=n_speech_segs,
            n_silence_segments=n_sil_segs,
            mean_speech_segment_len=round(mean_sp_len, 2),
            mean_silence_segment_len=round(mean_sil_len, 2),
            quality_label=quality_label,
            quality_description=quality_desc,
            threshold=tau,
        )

    @staticmethod
    def _bimodality_coeff(
        probs: List[float], mean_p: float, std_p: float, n: int
    ) -> float:
        """Sarle's Bimodality Coefficient. >0.555 → bimodal."""
        if std_p == 0 or n < 3:
            return 0.0
        # skewness
        skew = sum((p - mean_p) ** 3 for p in probs) / (n * std_p**3)
        # excess kurtosis
        kurt = sum((p - mean_p) ** 4 for p in probs) / (n * std_p**4) - 3
        # Sarle's formula; avoid div-by-zero with a small epsilon
        denominator = kurt + 3 if abs(kurt + 3) > 1e-9 else 1e-9
        bc = (skew**2 + 1) / denominator
        return bc

    @staticmethod
    def _segment_stats(probs: List[float], tau: float) -> Tuple[int, int, float, float]:
        """Count and measure speech/silence run-lengths."""
        speech_runs: List[int] = []
        silence_runs: List[int] = []

        current_is_speech = probs[0] >= tau
        run_len = 1

        for p in probs[1:]:
            is_speech = p >= tau
            if is_speech == current_is_speech:
                run_len += 1
            else:
                (speech_runs if current_is_speech else silence_runs).append(run_len)
                current_is_speech = is_speech
                run_len = 1
        (speech_runs if current_is_speech else silence_runs).append(run_len)

        mean_sp = sum(speech_runs) / len(speech_runs) if speech_runs else 0.0
        mean_sil = sum(silence_runs) / len(silence_runs) if silence_runs else 0.0
        return len(speech_runs), len(silence_runs), mean_sp, mean_sil


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def score(probs: List[float], threshold: float = 0.5) -> VADMetrics:
    """One-liner: score a prob list and return VADMetrics."""
    return VADScorer(probs, threshold=threshold).metrics()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import numpy as np

    parser = argparse.ArgumentParser(description="Run VAD scoring on probability file")
    parser.add_argument(
        "probs_path",
        type=Path,
        help="Path to the probabilities file (e.g. .npy, .pkl, etc.)",
    )

    args = parser.parse_args()

    # Load probabilities
    probs_path = args.probs_path
    print(f"Loading probabilities from: {probs_path}")

    if probs_path.suffix == ".npy":
        probs = np.load(probs_path)
    elif probs_path.suffix == ".pkl":
        import pickle

        with open(probs_path, "rb") as f:
            probs = pickle.load(f)
    else:
        # Add more formats as needed
        probs = np.load(probs_path)  # default fallback

    # Run scorer
    scorer = VADScorer(probs)
    print(scorer.report())
