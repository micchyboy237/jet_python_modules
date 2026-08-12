import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def sigmoid_normalize_scores(
    scores: list[float],
    temperature: float = 1.0,
) -> list[float]:
    """
    Apply sigmoid normalization to convert raw scores to 0–1 range.

    Uses sigmoid function: normalized = 1 / (1 + exp(-score / temperature))

    Args:
        scores: List of raw scores (can be negative or positive).
        temperature: Controls the steepness of the sigmoid curve.
                     Lower = more extreme (closer to 0 or 1).
                     Higher = more moderate (closer to 0.5).
                     Default 1.0 works well for most cross-encoders.

    Returns:
        List of normalized scores in range (0, 1).
    """
    if not scores:
        return []

    normalized = []
    for score in scores:
        # Apply sigmoid with temperature scaling
        norm_score = 1.0 / (1.0 + np.exp(-score / temperature))
        normalized.append(float(norm_score))

    return normalized
