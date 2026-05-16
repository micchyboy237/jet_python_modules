"""
vad_scorer.py
=============
Reusable scoring module for VAD speech-probability lists.
No ground-truth labels required.

Frame-rate independence
------------------------
Three metrics were previously frame-rate-sensitive.  They are now normalised
using ``frame_shift_s`` (seconds per frame, default 0.010 = 10 ms):

  * ``jitter_per_s``      — frame-to-frame |delta| divided by frame_shift_s
                            → probability units per second; comparable across VADs
  * ``smoothness_score``  — derived from jitter_per_s, not raw jitter
  * ``mean_speech_segment_len_s`` / ``mean_silence_segment_len_s``
                          — run-lengths multiplied by frame_shift_s → seconds

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
                        inverse of normalised jitter_per_s

Usage
-----
    from vad_scorer import VADScorer

    probs = [0.01, 0.02, 0.95, 0.97, 0.98, 0.50, 0.03, 0.99]

    # Default: 10 ms frame shift (FireRedVAD / Silero default)
    scorer = VADScorer(probs)

    # Explicit frame shift for a different VAD
    scorer = VADScorer(probs, frame_shift_s=0.025)   # 25 ms shift

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
# Smoothness normalisation anchor
# ---------------------------------------------------------------------------
# Maximum "expected" jitter in probability-units per second.
# At 10 ms shift, the worst possible frame-to-frame delta is 1.0 / frame →
# 1.0 / 0.010 = 100 prob-units/s.  Observed real-world ceiling is ~50.
# We use 50 as the normalisation ceiling so the score stays in [0, 1]
# with sensible headroom.  Anything at or above 50 prob-units/s → score = 0.
_MAX_JITTER_PER_S: float = 50.0

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


@dataclass
class VADMetrics:
    """All computed metrics for a probability list."""

    n_frames: int
    mean_prob: float
    median_prob: float
    std_prob: float
    max_prob: float
    min_prob: float
    confidence: float
    speech_ratio: float
    subthresh_ratio: float

    # --- frame-rate-aware jitter -------------------------------------------
    jitter: float  # raw mean |delta| per frame  (kept for compat.)
    jitter_per_s: float  # jitter / frame_shift_s  (comparable across VADs)

    bimodality_coeff: float
    peak_score: float
    activity_score: float
    clarity_score: float
    smoothness_score: float  # now derived from jitter_per_s
    composite_score: float

    n_speech_segments: int
    n_silence_segments: int

    # --- frame-rate-aware segment lengths ----------------------------------
    mean_speech_segment_len: float  # frames  (kept for compat.)
    mean_silence_segment_len: float  # frames  (kept for compat.)
    mean_speech_segment_len_s: float  # seconds
    mean_silence_segment_len_s: float  # seconds

    quality_label: str
    quality_description: str
    threshold: float = 0.5
    frame_shift_s: float = 0.010

    def as_dict(self) -> dict:
        return asdict(self)


class VADScorer:
    """
    Score a list of VAD speech probabilities without ground-truth labels.

    Parameters
    ----------
    probs           : list of floats in [0, 1]
    threshold       : hard speech/silence boundary (default 0.5)
    subthresh_low   : lower edge of the "almost speech" band (default 0.2)
    frame_shift_s   : seconds per frame (default 0.010 = 10 ms).
                      Set this to match your VAD's hop size so that
                      smoothness_score and segment lengths are comparable
                      across different frame rates.
    """

    _W_PEAK = 0.35
    _W_ACTIVITY = 0.30
    _W_CLARITY = 0.20
    _W_SMOOTHNESS = 0.15

    def __init__(
        self,
        probs: List[float],
        threshold: float = 0.5,
        subthresh_low: float = 0.2,
        frame_shift_s: float = 0.010,
    ):
        if not probs:
            raise ValueError("probs list must not be empty.")
        for i, p in enumerate(probs):
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"probs[{i}] = {p} is outside [0, 1].")
        if frame_shift_s <= 0:
            raise ValueError("frame_shift_s must be positive.")

        self._probs = list(probs)
        self._threshold = threshold
        self._subthresh = subthresh_low
        self._frame_shift_s = frame_shift_s
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
        fs = self._frame_shift_s

        sep = "=" * 60
        div = "-" * 56

        lines = [
            sep,
            f"  VAD Score Report  (n={m.n_frames} frames, tau={tau}, shift={fs * 1000:.1f} ms)",
            sep,
            "",
            f"  {'Distribution':─<55}",
            f"  Mean prob          : {m.mean_prob:.4f}",
            f"  Median prob        : {m.median_prob:.4f}",
            f"  Std dev            : {m.std_prob:.4f}",
            f"  Max prob           : {m.max_prob:.4f}",
            f"  Min prob           : {m.min_prob:.4f}",
            "",
            f"  {'Classic Metrics':─<55}",
            f"  Confidence         : {m.confidence:.4f}   (mean|p-0.5|x2; 0=uncertain->1=crisp)",
            f"  Speech ratio       : {m.speech_ratio:.1%}   (frames >= {tau})",
            f"  Sub-thresh ratio   : {m.subthresh_ratio:.1%}   (frames in ({sub}, {tau}))",
            f"  Jitter             : {m.jitter:.4f}   (mean frame-to-frame |delta|, raw)",
            f"  Jitter/s           : {m.jitter_per_s:.4f}  (prob-units/s — frame-rate invariant)",
            f"  Bimodality coeff   : {m.bimodality_coeff:.4f}  (>0.555 = bimodal)",
            "",
            f"  {'Composite Score Components':─<55}",
            f"  Peak score         : {m.peak_score:.4f}   (w={self._W_PEAK})  margin above tau at peak",
            f"  Activity score     : {m.activity_score:.4f}   (w={self._W_ACTIVITY})  fraction with notable VAD activity",
            f"  Clarity score      : {m.clarity_score:.4f}   (w={self._W_CLARITY})  bimodality / separation",
            f"  Smoothness score   : {m.smoothness_score:.4f}   (w={self._W_SMOOTHNESS})  1 - jitter_per_s/{_MAX_JITTER_PER_S}",
            f"  {div}",
            f"  COMPOSITE SCORE    : {m.composite_score:.4f}   <- the single best metric",
            "",
            f"  {'Segment Stats':─<55}",
            f"  Speech segments    : {m.n_speech_segments}  "
            f"(avg {m.mean_speech_segment_len:.1f} frames / {m.mean_speech_segment_len_s * 1000:.0f} ms)",
            f"  Silence segments   : {m.n_silence_segments}  "
            f"(avg {m.mean_silence_segment_len:.1f} frames / {m.mean_silence_segment_len_s * 1000:.0f} ms)",
            "",
            f"  {'Quality':─<55}",
            f"  Label              : {m.quality_label}",
            f"  Note               : {m.quality_description}",
            sep,
        ]
        return "\n".join(lines)

    def plot(self, title: str = "VAD probability list") -> None:
        """
        Plot the probability list with threshold lines and speech shading.
        X-axis is shown in seconds (using frame_shift_s).
        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")

        m = self.metrics()
        probs = self._probs
        n = len(probs)
        fs = self._frame_shift_s
        times = [i * fs for i in range(n)]

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.fill_between(times, probs, alpha=0.20, color="#3B8BD4")
        ax.plot(times, probs, color="#185FA5", linewidth=1.2, label="speech prob")
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
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("P(speech)")
        ax.set_title(
            f"{title}  —  [{m.quality_label}]  "
            f"composite={m.composite_score:.3f}  peak={m.max_prob:.3f}  "
            f"shift={fs * 1000:.0f} ms"
        )
        ax.legend(loc="upper right", fontsize=9)

        in_speech, start = False, 0.0
        for i, p in enumerate(probs):
            t = i * fs
            if p >= self._threshold and not in_speech:
                start, in_speech = t, True
            elif p < self._threshold and in_speech:
                ax.axvspan(start, t, alpha=0.15, color="#3B8BD4")
                in_speech = False
        if in_speech:
            ax.axvspan(start, n * fs, alpha=0.15, color="#3B8BD4")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute(self) -> VADMetrics:
        probs = self._probs
        tau = self._threshold
        sub = self._subthresh
        fs = self._frame_shift_s
        n = len(probs)

        # --- distribution stats ------------------------------------------
        mean_p = sum(probs) / n
        median_p = statistics.median(probs)
        std_p = statistics.pstdev(probs)
        max_p = max(probs)
        min_p = min(probs)

        # --- classic metrics ---------------------------------------------
        confidence = sum(abs(p - 0.5) * 2 for p in probs) / n
        speech_ratio = sum(1 for p in probs if p >= tau) / n
        subthresh_ratio = sum(1 for p in probs if sub < p < tau) / n

        # Raw jitter (mean |delta| per frame) — kept for backwards compat.
        jitter = (
            sum(abs(probs[i] - probs[i - 1]) for i in range(1, n)) / (n - 1)
            if n > 1
            else 0.0
        )

        # Frame-rate-invariant jitter: probability units per second.
        # Dividing by frame_shift_s converts "delta per frame" →
        # "delta per second", making it comparable across VAD types.
        jitter_per_s = jitter / fs

        bimodality_coeff = self._sarle_bc(probs, mean_p, std_p, n)

        # --- composite component scores ----------------------------------
        # peak_score: 0 if never crossed tau, else linear up to 1.
        if max_p >= tau and tau < 1.0:
            peak_score = (max_p - tau) / (1.0 - tau)
        elif max_p >= tau:
            peak_score = 1.0
        else:
            peak_score = 0.0

        activity_score = sum(1 for p in probs if p >= sub) / n
        clarity_score = min(std_p / 0.5, 1.0)

        # smoothness_score: derived from jitter_per_s so it is independent
        # of frame shift.  A jitter_per_s of 0 → score 1.0; at or above
        # _MAX_JITTER_PER_S → score 0.0.
        smoothness_score = max(0.0, 1.0 - jitter_per_s / _MAX_JITTER_PER_S)

        composite_score = (
            self._W_PEAK * peak_score
            + self._W_ACTIVITY * activity_score
            + self._W_CLARITY * clarity_score
            + self._W_SMOOTHNESS * smoothness_score
        )

        # --- segment stats -----------------------------------------------
        n_sp, n_sil, mean_sp_f, mean_sil_f = self._segment_stats(probs, tau)
        mean_sp_s = mean_sp_f * fs
        mean_sil_s = mean_sil_f * fs

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
            jitter_per_s=round(jitter_per_s, 6),
            bimodality_coeff=round(bimodality_coeff, 6),
            peak_score=round(peak_score, 6),
            activity_score=round(activity_score, 6),
            clarity_score=round(clarity_score, 6),
            smoothness_score=round(smoothness_score, 6),
            composite_score=round(composite_score, 6),
            n_speech_segments=n_sp,
            n_silence_segments=n_sil,
            mean_speech_segment_len=round(mean_sp_f, 2),
            mean_silence_segment_len=round(mean_sil_f, 2),
            mean_speech_segment_len_s=round(mean_sp_s, 4),
            mean_silence_segment_len_s=round(mean_sil_s, 4),
            quality_label=quality_label,
            quality_description=quality_desc,
            threshold=tau,
            frame_shift_s=fs,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

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
        """
        Count run-lengths of speech and silence segments (in frames).
        Callers multiply by frame_shift_s to get seconds.
        """
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
# Convenience one-liner
# ---------------------------------------------------------------------------


def score(
    probs: List[float],
    threshold: float = 0.5,
    frame_shift_s: float = 0.010,
) -> VADMetrics:
    """One-liner: score a prob list and return VADMetrics."""
    return VADScorer(probs, threshold=threshold, frame_shift_s=frame_shift_s).metrics()


# ---------------------------------------------------------------------------
# CLI entry-point
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
    parser.add_argument(
        "--frame-shift-ms",
        type=float,
        default=10.0,
        help="Frame shift in milliseconds (default: 10 ms)",
    )
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

    scorer = VADScorer(
        probs,
        threshold=args.threshold,
        subthresh_low=args.subthresh,
        frame_shift_s=args.frame_shift_ms / 1000.0,
    )
    print(scorer.report())
