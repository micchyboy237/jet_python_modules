import numpy as np
from jet.audio.audio_waveform.vad.vad_config import (
    DEFAULT_PROB_WEIGHT,
    DEFAULT_RMS_WEIGHT,
)
from jet.audio.helpers.config import (
    HOP_SIZE,
)

# ---------------------------------------------------------------------------
# Reusable hybrid probability computation
# ---------------------------------------------------------------------------


def compute_hybrid_probs(
    probs: np.ndarray,
    audio_np: np.ndarray,
    prob_weight: float = DEFAULT_PROB_WEIGHT,
    rms_weight: float = DEFAULT_RMS_WEIGHT,
    frame_samples: int = HOP_SIZE,
) -> np.ndarray:
    """
    Compute hybrid scores by combining speech probabilities with normalised RMS energy.

    The hybrid score per frame is:
        score = prob_weight * smoothed_prob + rms_weight * rms_norm

    RMS is normalised using the 99th-percentile of the segment's RMS values.

    Args:
        probs:         Speech probability array (one value per 10ms frame).
        audio_np:      Corresponding audio signal as numpy array.
        prob_weight:   Weight for the speech probability component.
        rms_weight:    Weight for the RMS energy component.
        frame_samples: Number of audio samples per frame (160 @ 16kHz = 10ms).

    Returns:
        Numpy array of hybrid scores, same length as probs.
    """
    n_frames = len(probs)
    if n_frames == 0:
        return np.array([], dtype=np.float32)

    # Compute per-frame RMS aligned to probs
    n_audio_frames = len(audio_np) // frame_samples
    n_common = min(n_frames, n_audio_frames)
    if n_common == 0:
        return np.array([], dtype=np.float32)

    frames = audio_np[: n_common * frame_samples].reshape(n_common, frame_samples)
    rms_arr = np.sqrt(np.mean(frames**2, axis=1))
    rms_ceil = np.percentile(rms_arr, 99) + 1e-10
    rms_norm = np.clip(rms_arr / rms_ceil, 0.0, 1.0)

    hybrid = prob_weight * probs[:n_common] + rms_weight * rms_norm

    # If probs is longer than available audio frames, pad with prob-only values
    if n_frames > n_common:
        pad = prob_weight * probs[n_common:]
        hybrid = np.concatenate([hybrid, pad])

    return hybrid.astype(np.float32)
