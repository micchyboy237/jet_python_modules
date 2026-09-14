"""Score fusion and normalization utilities for multi-signal retrieval.

Pure functions for combining and normalizing ranking signals.
No I/O, no logging, no API calls. All functions accept list or ndarray
inputs and return np.ndarray outputs for consistent downstream processing.

Strategies:
    - weighted_sum: Linear combination with configurable weights (default)
    - rrf: Reciprocal Rank Fusion for heterogeneous signals
    - max_score: Conservative aggregation using maximum signal per document

Normalization:
    - sigmoid: Maps raw scores to (0, 1) with temperature control
    - minmax: Scales to [0, 1] based on observed range
    - zscore: Standardizes to zero mean, unit variance
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

# Type alias for flexible input
ScoreArray = Union[Sequence[float], np.ndarray]


def _to_array(scores: ScoreArray) -> np.ndarray:
    """Convert input to float32 numpy array."""
    if isinstance(scores, np.ndarray):
        return scores.astype(np.float32, copy=False)
    return np.array(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Normalization Functions
# ---------------------------------------------------------------------------


def normalize_sigmoid(
    scores: ScoreArray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Sigmoid normalization mapping raw scores to (0, 1).

    Args:
        scores: Raw scores (any scale).
        temperature: Controls curve steepness. Lower = more extreme.

    Returns:
        Normalized scores in range (0, 1).
    """
    arr = _to_array(scores)
    if arr.size == 0:
        return arr
    return 1.0 / (1.0 + np.exp(-arr / temperature))


def normalize_minmax(scores: ScoreArray) -> np.ndarray:
    """Min-max normalization scaling to [0, 1].

    Returns zeros if all scores are identical.
    """
    arr = _to_array(scores)
    if arr.size == 0:
        return arr
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def normalize_zscore(scores: ScoreArray) -> np.ndarray:
    """Z-score normalization (zero mean, unit variance).

    Returns zeros if standard deviation is zero.
    """
    arr = _to_array(scores)
    if arr.size == 0:
        return arr
    std = arr.std()
    if std == 0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


# ---------------------------------------------------------------------------
# Fusion Strategies
# ---------------------------------------------------------------------------


def fuse_weighted_sum(
    signal_scores: dict[str, ScoreArray],
    weights: dict[str, float],
    normalize: bool = True,
) -> np.ndarray:
    """Weighted linear combination of multiple signals.

    Args:
        signal_scores: Dict mapping signal name → score array.
            All arrays must have the same length.
        weights: Dict mapping signal name → weight. Unspecified signals
            get weight 0. Weights are auto-normalized to sum to 1.0.
        normalize: If True, apply minmax normalization to each signal
            before combining (recommended when signals have different scales).

    Returns:
        Fused score array.
    """
    if not signal_scores:
        return np.array([], dtype=np.float32)

    # Determine length from first signal
    first_key = next(iter(signal_scores))
    n = len(signal_scores[first_key])
    if n == 0:
        return np.array([], dtype=np.float32)

    # Auto-normalize weights
    total_weight = sum(weights.get(k, 0.0) for k in signal_scores)
    if total_weight <= 0:
        raise ValueError("At least one signal must have a positive weight")

    fused = np.zeros(n, dtype=np.float32)
    for name, scores in signal_scores.items():
        w = weights.get(name, 0.0) / total_weight
        if w == 0:
            continue
        arr = _to_array(scores)
        if len(arr) != n:
            raise ValueError(f"Signal '{name}' has length {len(arr)}, expected {n}")
        if normalize:
            arr = normalize_minmax(arr)
        fused += w * arr

    return fused


def fuse_rrf(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
) -> np.ndarray:
    """Reciprocal Rank Fusion for combining ranked lists.

    Ideal for heterogeneous signals where score magnitudes are incomparable
    (e.g., BM25 raw scores vs cosine similarity).

    Args:
        rankings: List of ranked index lists. Each inner list contains
            document indices ordered by relevance (best first).
        k: Smoothing constant. Higher k compresses score differences.
            Default 60 is standard from Cormack et al.

    Returns:
        RRF scores array indexed by document position. Length equals
        max(max(rankings)) + 1.
    """
    if not rankings:
        return np.array([], dtype=np.float32)

    max_idx = max(
        (max(r) for r in rankings if r),
        default=-1,
    )
    if max_idx < 0:
        return np.array([], dtype=np.float32)

    scores = np.zeros(max_idx + 1, dtype=np.float32)
    for ranking in rankings:
        for rank_pos, doc_idx in enumerate(ranking):
            scores[doc_idx] += 1.0 / (k + rank_pos + 1)

    return scores


def fuse_max(
    signal_scores: dict[str, ScoreArray],
    normalize: bool = True,
) -> np.ndarray:
    """Element-wise maximum across signals.

    Useful for conservative retrieval where any strong signal should
    surface a document.

    Args:
        signal_scores: Dict mapping signal name → score array.
        normalize: If True, apply minmax normalization first.

    Returns:
        Max-fused score array.
    """
    if not signal_scores:
        return np.array([], dtype=np.float32)

    arrays = []
    for scores in signal_scores.values():
        arr = _to_array(scores)
        if normalize:
            arr = normalize_minmax(arr)
        arrays.append(arr)

    return np.maximum.reduce(arrays)


# ---------------------------------------------------------------------------
# Vector Similarity (moved from scoring_utils)
# ---------------------------------------------------------------------------


def cosine_similarity(a: ScoreArray, b: ScoreArray) -> float:
    """Cosine similarity between two vectors."""
    va = _to_array(a).flatten()
    vb = _to_array(b).flatten()
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))
