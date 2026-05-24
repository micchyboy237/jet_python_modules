# jet.audio.helpers.loudness


import numpy as np
from jet.audio.helpers.config import (
    FRAME_LENGTH_SAMPLE,
    FRAME_SHIFT_SAMPLE,
    NORMAL_MAX,
)
from jet.audio.helpers.energy import compute_frame_rms, compute_rms


def get_peak_rms(
    audio: np.ndarray,
    frame_length: int = FRAME_LENGTH_SAMPLE,
    hop_length: int = FRAME_SHIFT_SAMPLE,
) -> float:
    """Get the peak RMS value across all frames in the audio.

    This is used to determine if the audio is already loud enough
    and doesn't need amplification.

    Parameters
    ----------
    audio : np.ndarray
        Input audio samples (1D array, float32/float64).
    frame_length : int
        Frame length in samples for RMS computation.
    hop_length : int
        Hop length in samples between frames.

    Returns
    -------
    float
        Peak RMS value across all frames.
    """
    if len(audio) == 0:
        return 0.0

    frame_rms = compute_frame_rms(audio, frame_length, hop_length)
    if len(frame_rms) == 0:
        return compute_rms(audio)  # fallback to overall RMS

    return float(np.max(frame_rms))


def increase_loudness(
    audio: np.ndarray,
    target_rms: float = NORMAL_MAX,
    max_gain_db: float = 12.0,
) -> np.ndarray:
    """Simple version: increase overall loudness if below target.

    This is a simpler alternative that applies uniform gain to the entire
    audio signal instead of per-frame processing. Faster but less precise.

    Parameters
    ----------
    audio : np.ndarray
        Input audio samples (1D array).
    target_rms : float
        Target RMS value. Audio below this will be amplified.
    max_gain_db : float
        Maximum gain in dB to apply.

    Returns
    -------
    np.ndarray
        Audio with increased loudness if needed.
    """
    if len(audio) == 0:
        return audio.copy()

    # Quick check if audio is already loud enough
    current_rms = compute_rms(audio)
    if current_rms >= target_rms:
        return audio.copy()

    # Calculate gain needed
    required_gain = target_rms / max(current_rms, 1e-10)  # avoid division by zero

    # Limit gain
    max_gain_linear = 10 ** (max_gain_db / 20.0)
    gain = min(required_gain, max_gain_linear)

    # Apply gain
    output = audio * gain

    # Clip to prevent distortion
    output = np.clip(output, -1.0, 1.0)

    return output.astype(audio.dtype)
