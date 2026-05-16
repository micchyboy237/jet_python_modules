"""
vad_scorer.py
=============
Reusable scoring module for VAD speech-probability lists.
No ground-truth labels required.

Key insight from real-world data
---------------------------------
A prob list that is *entirely below the detection threshold* (e.g. max=0.44)
but dominated by zeros scores artificially high on raw *confidence* because
the zeros are confidently silent — but there is no speech at all.

This module fixes that with a **composite score** built from four components:

  1. peak_score       — how far above threshold the peak rose           (w=0.35)
                        0 if max_prob < threshold; scales to 1 at max=1.0
  2. activity_score   — fraction of frames with any notable VAD signal  (w=0.30)
                        uses a sub-threshold band to catch "almost speech"
  3. clarity_score    — how bimodal/unambiguous the distribution is     (w=0.20)
                        std-based proxy, avoids Sarle BC edge-cases
  4. smoothness_score — how smooth the signal is (real speech is smooth)(w=0.15)
                        inverse of normalised frame-to-frame jitter

Usage
-----
    from vad_scorer import VADScorer

    probs = [0.01, 0.02, 0.95, 0.97, 0.98, 0.50, 0.03, 0.99]
    scorer = VADScorer(probs)

    print(scorer.report())     # full human-readable report
    print(scorer.label())      # "Good", "Very bad", etc.
    print(scorer.score())      # composite float in [0, 1]
    print(scorer.summary())    # dict (serialisable)
    scorer.plot()              # requires matplotlib
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Quality bands  (driven by composite_score)
# ---------------------------------------------------------------------------

QUALITY_BANDS: List[Tuple[float, str, str]] = [
    (
        0.80,
        "Very good",
        "Strong, confident speech detection — clear bimodal separation.",
    ),
    (0.60, "Good", "Mostly crisp speech/silence decisions. A few borderline frames."),
    (0.40, "Fair", "Partial or weak speech signal. Consider lowering the threshold."),
    (0.20, "Bad", "Probs barely rise above silence. Likely very quiet or non-speech."),
    (0.00, "Very bad", "No credible speech detected. Segment is silent or noise-only."),
]


def _quality_from_score(composite_score: float) -> Tuple[str, str]:
    for threshold, label, description in QUALITY_BANDS:
        if composite_score > threshold:
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
    max_prob: float
    min_prob: float

    # Classic no-GT metrics (reference / downstream use)
    confidence: float  # mean(|p - 0.5| * 2)       0=uncertain → 1=crisp
    speech_ratio: float  # fraction of frames >= threshold
    subthresh_ratio: float  # fraction in (subthresh_low, threshold)
    jitter: float  # mean frame-to-frame |delta|
    bimodality_coeff: float  # Sarle's BC; >0.555 = bimodal

    # Composite score components  [0, 1]
    peak_score: float  # margin above threshold at the peak
    activity_score: float  # fraction with any notable activity
    clarity_score: float  # distribution bimodality / separation
    smoothness_score: float  # inverse jitter (smooth = likely real speech)

    # The single best number
    composite_score: float

    # Segment stats
    n_speech_segments: int
    n_silence_segments: int
    mean_speech_segment_len: float  # frames
    mean_silence_segment_len: float  # frames

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
    probs           : list of floats in [0, 1]
    threshold       : hard speech/silence boundary (default 0.5)
    subthresh_low   : lower edge of the "almost speech" band (default 0.2)
    """

    # Composite weights — must sum to 1.0
    _W_PEAK = 0.35
    _W_ACTIVITY = 0.30
    _W_CLARITY = 0.20
    _W_SMOOTHNESS = 0.15

    def __init__(
        self,
        probs: List[float],
        threshold: float = 0.5,
        subthresh_low: float = 0.2,
    ):
        if not probs:
            raise ValueError("probs list must not be empty.")
        for i, p in enumerate(probs):
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"probs[{i}] = {p} is outside [0, 1].")
        self._probs = list(probs)
        self._threshold = threshold
        self._subthresh = subthresh_low
        self._metrics: Optional[VADMetrics] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def metrics(self) -> VADMetrics:
        """Return the full VADMetrics dataclass (cached after first call)."""
        if self._metrics is None:
            self._metrics = self._compute()
        return self._metrics

    def summary(self) -> dict:
        """All metrics as a plain serialisable dict."""
        return self.metrics().as_dict()

    def label(self) -> str:
        """Quality label: 'Very bad' / 'Bad' / 'Fair' / 'Good' / 'Very good'."""
        return self.metrics().quality_label

    def score(self) -> float:
        """Composite score in [0, 1]. The single best number for this prob list."""
        return self.metrics().composite_score

    def confidence(self) -> float:
        """Raw confidence score mean(|p-0.5|*2). Kept for backwards compatibility."""
        return self.metrics().confidence

    def speech_ratio(self) -> float:
        """Fraction of frames at or above threshold."""
        return self.metrics().speech_ratio

    def report(self) -> str:
        """Human-readable multi-section report."""
        m = self.metrics()
        tau = self._threshold
        sub = self._subthresh
        sep = "=" * 56
        div = "-" * 52
        lines = [
            sep,
            f"  VAD Score Report  (n={m.n_frames} frames, tau={tau})",
            sep,
            "",
            f"  {'Distribution':─<50}",
            f"  Mean prob          : {m.mean_prob:.4f}",
            f"  Median prob        : {m.median_prob:.4f}",
            f"  Std dev            : {m.std_prob:.4f}",
            f"  Max prob           : {m.max_prob:.4f}",
            f"  Min prob           : {m.min_prob:.4f}",
            "",
            f"  {'Classic Metrics':─<50}",
            f"  Confidence         : {m.confidence:.4f}   (mean|p-0.5|x2; 0=uncertain->1=crisp)",
            f"  Speech ratio       : {m.speech_ratio:.1%}   (frames >= {tau})",
            f"  Sub-thresh ratio   : {m.subthresh_ratio:.1%}   (frames in ({sub}, {tau}))",
            f"  Jitter             : {m.jitter:.4f}   (frame-to-frame |delta|)",
            f"  Bimodality coeff   : {m.bimodality_coeff:.4f}  (>0.555 = bimodal)",
            "",
            f"  {'Composite Score Components':─<50}",
            f"  Peak score         : {m.peak_score:.4f}   (w={self._W_PEAK})  margin above tau at peak",
            f"  Activity score     : {m.activity_score:.4f}   (w={self._W_ACTIVITY})  fraction with notable VAD activity",
            f"  Clarity score      : {m.clarity_score:.4f}   (w={self._W_CLARITY})  bimodality / separation",
            f"  Smoothness score   : {m.smoothness_score:.4f}   (w={self._W_SMOOTHNESS})  inverse jitter",
            f"  {div}",
            f"  COMPOSITE SCORE    : {m.composite_score:.4f}   <- the single best metric",
            "",
            f"  {'Segment Stats':─<50}",
            f"  Speech segments    : {m.n_speech_segments}  (avg {m.mean_speech_segment_len:.1f} frames)",
            f"  Silence segments   : {m.n_silence_segments}  (avg {m.mean_silence_segment_len:.1f} frames)",
            "",
            f"  {'Quality':─<50}",
            f"  Label              : {m.quality_label}",
            f"  Note               : {m.quality_description}",
            sep,
        ]
        return "\n".join(lines)

    def plot(self, title: str = "VAD probability list") -> None:
        """
        Plot the probability list with threshold lines and speech shading.
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")

        m = self.metrics()
        probs = self._probs
        n = len(probs)

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.fill_between(range(n), probs, alpha=0.20, color="#3B8BD4")
        ax.plot(probs, color="#185FA5", linewidth=1.2, label="speech prob")
        ax.axhline(
            self._threshold,
            color="#E24B4A",
            linewidth=1,
            linestyle="--",
            label=f"threshold ({self._threshold})",
        )
        ax.axhline(
            self._subthresh,
            color="#F5A623",
            linewidth=0.8,
            linestyle=":",
            label=f"sub-thresh ({self._subthresh})",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Frame")
        ax.set_ylabel("P(speech)")
        ax.set_title(
            f"{title}  —  [{m.quality_label}]  "
            f"composite={m.composite_score:.3f}  peak={m.max_prob:.3f}"
        )
        ax.legend(loc="upper right", fontsize=9)

        in_speech, start = False, 0
        for i, p in enumerate(probs):
            if p >= self._threshold and not in_speech:
                start, in_speech = i, True
            elif p < self._threshold and in_speech:
                ax.axvspan(start, i, alpha=0.15, color="#3B8BD4")
                in_speech = False
        if in_speech:
            ax.axvspan(start, n, alpha=0.15, color="#3B8BD4")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(self) -> VADMetrics:
        probs = self._probs
        tau = self._threshold
        sub = self._subthresh
        n = len(probs)

        # Basic stats
        mean_p = sum(probs) / n
        median_p = statistics.median(probs)
        std_p = statistics.pstdev(probs)
        max_p = max(probs)
        min_p = min(probs)

        # Classic metrics
        confidence = sum(abs(p - 0.5) * 2 for p in probs) / n
        speech_ratio = sum(1 for p in probs if p >= tau) / n
        subthresh_ratio = sum(1 for p in probs if sub < p < tau) / n
        jitter = (
            sum(abs(probs[i] - probs[i - 1]) for i in range(1, n)) / (n - 1)
            if n > 1
            else 0.0
        )
        bimodality_coeff = self._sarle_bc(probs, mean_p, std_p, n)

        # --- Composite score components ---

        # 1. Peak score: how far above threshold did the peak reach?
        #    0 if max never crossed tau; scales linearly to 1.0 at max_prob=1.0.
        if max_p >= tau and tau < 1.0:
            peak_score = (max_p - tau) / (1.0 - tau)
        elif max_p >= tau:
            peak_score = 1.0
        else:
            peak_score = 0.0

        # 2. Activity score: fraction of frames with any notable VAD signal
        #    (above subthresh_low), rewards "almost speech" too.
        activity_score = sum(1 for p in probs if p >= sub) / n

        # 3. Clarity score: std / 0.5 (max possible std for a 0/1 distribution).
        #    Robust proxy for bimodality that doesn't break on edge cases.
        clarity_score = min(std_p / 0.5, 1.0)

        # 4. Smoothness score: real speech transitions are gradual.
        #    Normalise jitter against 0.5 (worst case = alternating 0↔1).
        smoothness_score = max(0.0, 1.0 - jitter / 0.5)

        composite_score = (
            self._W_PEAK * peak_score
            + self._W_ACTIVITY * activity_score
            + self._W_CLARITY * clarity_score
            + self._W_SMOOTHNESS * smoothness_score
        )

        n_sp, n_sil, mean_sp, mean_sil = self._segment_stats(probs, tau)
        quality_label, quality_desc = _quality_from_score(composite_score)

        return VADMetrics(
            n_frames=n,
            mean_prob=round(mean_p, 6),
            median_prob=round(median_p, 6),
            std_prob=round(std_p, 6),
            max_prob=round(max_p, 6),
            min_prob=round(min_p, 6),
            confidence=round(confidence, 6),
            speech_ratio=round(speech_ratio, 6),
            subthresh_ratio=round(subthresh_ratio, 6),
            jitter=round(jitter, 6),
            bimodality_coeff=round(bimodality_coeff, 6),
            peak_score=round(peak_score, 6),
            activity_score=round(activity_score, 6),
            clarity_score=round(clarity_score, 6),
            smoothness_score=round(smoothness_score, 6),
            composite_score=round(composite_score, 6),
            n_speech_segments=n_sp,
            n_silence_segments=n_sil,
            mean_speech_segment_len=round(mean_sp, 2),
            mean_silence_segment_len=round(mean_sil, 2),
            quality_label=quality_label,
            quality_description=quality_desc,
            threshold=tau,
        )

    @staticmethod
    def _sarle_bc(probs: List[float], mean_p: float, std_p: float, n: int) -> float:
        """Sarle's Bimodality Coefficient. Values >0.555 indicate bimodal distribution."""
        if std_p == 0 or n < 3:
            return 0.0
        skew = sum((p - mean_p) ** 3 for p in probs) / (n * std_p**3)
        kurt = sum((p - mean_p) ** 4 for p in probs) / (n * std_p**4) - 3
        denom = kurt + 3 if abs(kurt + 3) > 1e-9 else 1e-9
        return (skew**2 + 1) / denom

    @staticmethod
    def _segment_stats(probs: List[float], tau: float) -> Tuple[int, int, float, float]:
        """Count run-lengths of speech and silence segments."""
        speech_runs: List[int] = []
        silence_runs: List[int] = []
        current, run = probs[0] >= tau, 1
        for p in probs[1:]:
            sp = p >= tau
            if sp == current:
                run += 1
            else:
                (speech_runs if current else silence_runs).append(run)
                current, run = sp, 1
        (speech_runs if current else silence_runs).append(run)
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
# CLI / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Score a VAD probability file.")
    parser.add_argument(
        "probs_path", type=Path, help="Path to .json / .npy / .pkl probability file"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--subthresh", type=float, default=0.2)
    args = parser.parse_args()

    p = args.probs_path
    if p.suffix == ".json":
        probs = json.loads(p.read_text())
    elif p.suffix == ".npy":
        import numpy as np

        probs = np.load(p).tolist()
    elif p.suffix == ".pkl":
        import pickle

        probs = pickle.loads(p.read_bytes())
    else:
        raise ValueError(f"Unsupported format: {p.suffix}")
    scorer = VADScorer(probs, threshold=args.threshold, subthresh_low=args.subthresh)
    print(scorer.report())
